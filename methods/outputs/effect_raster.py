#!/usr/bin/env python3
import argparse
import os
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask


def _resolve_xy_columns(df):
    x_candidates = ["k_lng", "lng", "lon"]
    y_candidates = ["k_lat", "lat"]
    x_col = next((candidate for candidate in x_candidates if candidate in df.columns), None)
    y_col = next((candidate for candidate in y_candidates if candidate in df.columns), None)
    if x_col is None or y_col is None:
        raise KeyError(
            f"Could not find coordinate columns. Tried X={x_candidates}, Y={y_candidates}. "
            f"Available columns include: {list(df.columns)[:20]}"
        )
    return x_col, y_col

def _compute_effect_df(all_pairs_df):
    # List all columns in the DataFrame
    cols = all_pairs_df.columns.tolist()
    print(f"Columns in all_pairs_df: {cols[:20]}")
    x_col, y_col = _resolve_xy_columns(all_pairs_df)
    print(f"Identified coordinate columns: X={x_col}, Y={y_col}")
    # Determine if there are cols starting with s_prop_ or s_luc_ to identify kind and year
    s_prop_cols = [col for col in cols if re.match(r"s_prop_\d{4}", col)]
    s_luc_cols = [col for col in cols if re.match(r"s_luc_\d{4}", col)]
    if s_prop_cols:
        kind = "prop"
        year = s_prop_cols[0].split("_")[-1]
    elif s_luc_cols:
        kind = "luc"
        year = s_luc_cols[0].split("_")[-1]
    else:
        raise KeyError("Could not find columns matching s_prop_YYYY or s_luc_YYYY patterns.")
    
    # Find the years in k_luc_YYYY
    years = []
    for col in cols:
        match = re.match(r"k_luc_(\d{4})", col)
        if match:
            years.append(match.group(1))

    year = sorted(years, reverse=True)[0] if years else [-1]
    print(f"Most recent year found in k_luc_YYYY columns: {year}")

    k_col = f"k_luc_{year}"
    s_col = f"s_{kind}_{year}"

    k_vals = pd.to_numeric(all_pairs_df[k_col], errors="coerce")
    s_vals = pd.to_numeric(all_pairs_df[s_col], errors="coerce")

    k_vals = k_vals.where(k_vals <= 1, 0)
    effect = (k_vals-s_vals).clip(-1, 1)

    out_df = pd.DataFrame(
        {
            "k_lat": pd.to_numeric(all_pairs_df[y_col], errors="coerce"),
            "k_lng": pd.to_numeric(all_pairs_df[x_col], errors="coerce"),
            "effect": effect,
        }
    ).dropna(subset=["k_lat", "k_lng", "effect"])

    print(f"Using columns: {k_col} and {s_col}")
    print(f"Rows in effect parquet: {len(out_df)}")
    return out_df


def _rasterize_effect(effect_df, countries_tif, buffered_geojson, output_raster):
    buffered_gdf = gpd.read_file(buffered_geojson)

    with rasterio.open(countries_tif) as src:
        buffered_gdf = buffered_gdf.to_crs(src.crs)
        shapes = [geom for geom in buffered_gdf.geometry if geom is not None and not geom.is_empty]
        if not shapes:
            raise ValueError("No valid geometries found in buffered_project.geojson")

        cropped, out_transform = mask(src, shapes, crop=True, filled=False)
        valid_mask = ~cropped[0].mask

        height, width = valid_mask.shape
        effect_raster = np.full((height, width), np.nan, dtype=np.float32)

        points_gdf = gpd.GeoDataFrame(
            effect_df,
            geometry=gpd.points_from_xy(effect_df["k_lng"], effect_df["k_lat"]),
            crs="EPSG:4326",
        ).to_crs(src.crs)

        xs = points_gdf.geometry.x.to_numpy()
        ys = points_gdf.geometry.y.to_numpy()
        rows, cols = rasterio.transform.rowcol(out_transform, xs, ys)
        rows = np.asarray(rows)
        cols = np.asarray(cols)

        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)

        assigned = pd.DataFrame(
            {
                "row": rows[inside],
                "col": cols[inside],
                "effect": points_gdf.loc[inside, "effect"].to_numpy(),
            }
        )

        assigned = assigned[np.isfinite(assigned["effect"])]
        if len(assigned) > 0:
            grouped = assigned.groupby(["row", "col"], as_index=False)["effect"].mean()
            rr = grouped["row"].to_numpy(dtype=int)
            cc = grouped["col"].to_numpy(dtype=int)
            vv = grouped["effect"].to_numpy(dtype=np.float32)

            valid_cells = valid_mask[rr, cc]
            effect_raster[rr[valid_cells], cc[valid_cells]] = vv[valid_cells]

        nodata = -9999.0
        out_data = np.where(np.isnan(effect_raster), nodata, effect_raster).astype(np.float32)

        profile = src.profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": 1,
                "dtype": "float32",
                "transform": out_transform,
                "nodata": nodata,
                "compress": "lzw",
            }
        )

    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(out_data, 1)

    filled_cells = int(np.isfinite(effect_raster).sum())
    print(f"Rasterized effect to {output_raster}")
    print(f"Filled raster cells: {filled_cells}")


def _aggregate_raster_by_block_size(input_raster: str, output_raster: str, block_size: int = 16):
    """Aggregate `input_raster` by averaging each non-overlapping `block_size x block_size` pixel block.

    The output raster will have dimensions ceil(height/block_size) x ceil(width/block_size).
    Empty blocks (no valid pixels) are set to nodata.
    """
    with rasterio.open(input_raster) as src:
        data = src.read(1)
        nodata = src.nodata
        transform = src.transform
        profile = src.profile.copy()

    if nodata is None:
        nodata = np.nan

    height, width = data.shape

    out_h = int(np.ceil(height / block_size))
    out_w = int(np.ceil(width / block_size))
    out_arr = np.full((out_h, out_w), nodata, dtype=np.float32)

    for i in range(out_h):
        r0 = i * block_size
        r1 = min((i + 1) * block_size, height)
        for j in range(out_w):
            c0 = j * block_size
            c1 = min((j + 1) * block_size, width)
            block = data[r0:r1, c0:c1]
            if nodata is not None:
                valid = block != nodata
            else:
                valid = np.isfinite(block)
            if np.any(valid):
                out_arr[i, j] = float(np.mean(block[valid]))
            else:
                out_arr[i, j] = nodata

    # Build new transform: scale pixel sizes by block_size
    new_a = transform.a * block_size
    new_e = transform.e * block_size
    new_transform = rasterio.Affine(new_a, transform.b, transform.c, transform.d, new_e, transform.f)

    profile.update({
        "height": out_h,
        "width": out_w,
        "transform": new_transform,
        "dtype": "float32",
        "count": 1,
        "nodata": nodata,
        "compress": profile.get("compress", "lzw"),
    })

    write_arr = np.where(np.isfinite(out_arr), out_arr, nodata).astype(np.float32)

    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(write_arr, 1)

    print(f"Wrote aggregated raster ({block_size}x{block_size} blocks) to {output_raster}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build last-year effect (s-k) from all_pairs.parquet and rasterize onto countries.tif "
            "masked to buffered_project.geojson"
        )
    )
    parser.add_argument("--project-folder", required=True, help="Path to project folder")
    parser.add_argument(
        "--all-pairs",
        default=None,
        help="Path to all_pairs.parquet (default: <project-folder>/all_pairs.parquet)",
    )
    parser.add_argument(
        "--countries-tif",
        default=None,
        help="Path to countries.tif (default: <project-folder>/countries.tif)",
    )
    parser.add_argument(
        "--buffered-geojson",
        default=None,
        help="Path to buffered_project.geojson (default: <project-folder>/buffered_project.geojson)",
    )
    parser.add_argument(
        "--output-parquet",
        default=None,
        help="Output parquet path (default: <project-folder>/effect.parquet)",
    )
    parser.add_argument(
        "--output-raster",
        default=None,
        help="Output raster path (default: <project-folder>/effect.tif)",
    )

    args = parser.parse_args()

    all_pairs_path = args.all_pairs or os.path.join(args.project_folder, "all_pairs.parquet")
    countries_tif = args.countries_tif or os.path.join(args.project_folder, "countries.tif")
    buffered_geojson = args.buffered_geojson or os.path.join(args.project_folder, "buffered_project.geojson")
    output_parquet = args.output_parquet or os.path.join(args.project_folder, "effect.parquet")
    output_raster = args.output_raster or os.path.join(args.project_folder, "effect.tif")

    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    os.makedirs(os.path.dirname(output_raster), exist_ok=True)

    all_pairs_df = pd.read_parquet(all_pairs_path)
    effect_df = _compute_effect_df(all_pairs_df)
    effect_df.to_parquet(output_parquet, index=False)
    print(f"Saved effect parquet to {output_parquet}")

    _rasterize_effect(effect_df, countries_tif, buffered_geojson, output_raster)

    # Also produce a downsampled aggregated raster (16x16) by averaging pixels inside each block
    agg_raster_16 = os.path.splitext(output_raster)[0] + "_16x16.tif"
    try:
        _aggregate_raster_by_block_size(output_raster, agg_raster_16, block_size=16)
    except Exception as e:
        print(f"Warning: failed to produce aggregated 16x16 raster: {e}")


if __name__ == "__main__":
    main()
