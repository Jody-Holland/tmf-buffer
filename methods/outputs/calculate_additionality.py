#!/usr/bin/env python3
"""
Calculate year-by-year undisturbed forest change since start year for project and buffer.

Workflow:
1) Load all k-grids and classify pixels into project vs buffer.
2) Build K-based undisturbed forest series from luc_YYYY (value == 1), convert to hectares,
   and compute proportion/deforestation/change since start year.
3) Load non-matchless pairs parquet files, classify rows by k_lat/k_lng into project vs buffer,
   build S-based yearly forest proportions (s_prop_YYYY or s_luc_YYYY with values > 1 set to 0),
   then convert to hectares using K start-year baseline areas.
4) Export combined outputs.
"""
import os
import argparse
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

PIXEL_AREA_HA = 0.09


def _extract_years(columns, prefix):
    years = []
    for col in columns:
        if col.startswith(prefix):
            try:
                years.append(int(col.split("_")[-1]))
            except ValueError:
                continue
    return sorted(set(years))


def _resolve_xy_columns(df, x_candidates, y_candidates):
    x_col = next((candidate for candidate in x_candidates if candidate in df.columns), None)
    y_col = next((candidate for candidate in y_candidates if candidate in df.columns), None)
    if x_col is None or y_col is None:
        raise KeyError(
            f"Could not find coordinate columns. Tried X={x_candidates}, Y={y_candidates}. "
            f"Available columns include: {list(df.columns)[:20]}"
        )
    return x_col, y_col

def main():
    parser = argparse.ArgumentParser(description="Calculate project/buffer undisturbed forest change")
    parser.add_argument("--project-folder", required=True, help="Path to project folder")
    parser.add_argument("--og-geojson", required=True, help="Path to original project GeoJSON")
    parser.add_argument(
        "--pairs-folder",
        default="pairs_1000",
        help="Pairs folder under project-folder (e.g. pairs_1000, pairs_prop, pairs_propensity)",
    )
    args = parser.parse_args()

    project_folder = args.project_folder
    og_geojson_path = args.og_geojson

    # Load buffered + original project geometry and build outer buffer (buffered minus original)
    candidates_buffer_geojson = [
        os.path.join(project_folder, "buffered_project.geojson"),
        os.path.join(project_folder, "buffer.geojson"),
    ]
    buffered_project_geojson_path = next(path for path in candidates_buffer_geojson if os.path.isfile(path))
    buffered_project_gdf = gpd.read_file(buffered_project_geojson_path)
    project_gdf = gpd.read_file(og_geojson_path).to_crs(buffered_project_gdf.crs)

    project_geom = unary_union(project_gdf.geometry.values)
    buffer_geom = buffered_project_gdf.geometry.difference(project_geom)
    buffer_gdf = gpd.GeoDataFrame(geometry=buffer_geom, crs=buffered_project_gdf.crs)
    project_area_gdf = gpd.GeoDataFrame(geometry=project_gdf.geometry, crs=buffered_project_gdf.crs)

    # Load all k-grids parquet
    kgrids_parquet_path = os.path.join(project_folder, "k_grids", "all_k_grids.parquet")
    all_k_grids_df = pd.read_parquet(kgrids_parquet_path)

    years = _extract_years(all_k_grids_df.columns, "luc_")
    if not years:
        raise ValueError("No luc_YYYY columns found in all_k_grids.parquet")

    start_year = years[0] + 10
    start_year_col = f"luc_{start_year}"
    if start_year_col not in all_k_grids_df.columns:
        raise ValueError(
            f"Computed start year column {start_year_col} not present in all_k_grids.parquet"
        )

    years_since_start = [year for year in years if year >= start_year]

    # Build K-grid points and classify into buffer/project
    k_lng_col, k_lat_col = _resolve_xy_columns(
        all_k_grids_df,
        x_candidates=["lng", "lon", "k_lng"],
        y_candidates=["lat", "k_lat"],
    )

    all_k_grids_gdf = gpd.GeoDataFrame(
        all_k_grids_df,
        geometry=gpd.points_from_xy(all_k_grids_df[k_lng_col], all_k_grids_df[k_lat_col]),
        crs="EPSG:4326"
    )

    buffer_gdf = buffer_gdf.to_crs(epsg=4326)
    project_area_gdf = project_area_gdf.to_crs(epsg=4326)

    buffer_kgrids_gdf = gpd.sjoin(all_k_grids_gdf, buffer_gdf, how="inner", predicate='within')
    project_kgrids_gdf = gpd.sjoin(all_k_grids_gdf, project_area_gdf, how="inner", predicate='within')

    print(f"K-grid pixels in buffer: {len(buffer_kgrids_gdf)}")
    print(f"K-grid pixels in project: {len(project_kgrids_gdf)}")

    # K start-year undisturbed forest area: count(luc_start_year == 1) * 0.09 ha
    buffer_start_count = (buffer_kgrids_gdf[start_year_col] == 1).sum()
    project_start_count = (project_kgrids_gdf[start_year_col] == 1).sum()
    buffer_start_area_ha = buffer_start_count * PIXEL_AREA_HA
    project_start_area_ha = project_start_count * PIXEL_AREA_HA
    print(f"Start year: {start_year}")
    print(f"Buffer start undisturbed forest: {buffer_start_area_ha:.2f} ha")
    print(f"Project start undisturbed forest: {project_start_area_ha:.2f} ha")

    # K series for each year since start year
    k_rows = []
    for year in years_since_start:
        year_col = f"luc_{year}"
        buffer_count = (buffer_kgrids_gdf[year_col] == 1).sum()
        project_count = (project_kgrids_gdf[year_col] == 1).sum()

        buffer_area_ha_year = buffer_count * PIXEL_AREA_HA
        project_area_ha_year = project_count * PIXEL_AREA_HA

        buffer_proportion = (buffer_area_ha_year / buffer_start_area_ha) if buffer_start_area_ha > 0 else 0
        project_proportion = (project_area_ha_year / project_start_area_ha) if project_start_area_ha > 0 else 0

        buffer_deforestation_ha = buffer_start_area_ha - buffer_area_ha_year
        project_deforestation_ha = project_start_area_ha - project_area_ha_year
        buffer_change_ha = buffer_area_ha_year - buffer_start_area_ha
        project_change_ha = project_area_ha_year - project_start_area_ha

        k_rows.append({
            "year": year,
            "buffer_proportion_k": buffer_proportion,
            "project_proportion_k": project_proportion,
            "buffer_area_k": buffer_area_ha_year,
            "project_area_k": project_area_ha_year,
            "buffer_deforestation_k": buffer_deforestation_ha,
            "project_deforestation_k": project_deforestation_ha,
            "buffer_change_k": buffer_change_ha,
            "project_change_k": project_change_ha,
        })
    k_df = pd.DataFrame(k_rows)

    # Load all non-matchless pair parquets
    pairs_folder = os.path.join(project_folder, args.pairs_folder)
    if not os.path.isdir(pairs_folder):
        raise FileNotFoundError(f"Pairs folder not found: {pairs_folder}")

    parquet_files = [
        f for f in os.listdir(pairs_folder)
        if f.endswith(".parquet") and not f.endswith("_matchless.parquet")
    ]

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in pairs folder: {pairs_folder}")

    pairs_dfs = []
    for pq_file in parquet_files:
        pq_path = os.path.join(pairs_folder, pq_file)
        df = pd.read_parquet(pq_path)
        pairs_dfs.append(df)

    all_pairs_df = pd.concat(pairs_dfs, ignore_index=True)
    print(f"Total pairs rows loaded: {len(all_pairs_df)}")
    
    # Round all s_prop_YYYY and s_luc_YYYY columns to 4 decimal places to reduce file size (these are proportions so 4 decimals should be sufficient)
    for col in all_pairs_df.columns:
        if col.startswith("s_prop_") or col.startswith("s_luc_"):
            all_pairs_df[col] = all_pairs_df[col].round(4)
    
    # Save all pairs to paruqet
    all_pairs_parquet_path = os.path.join(project_folder, "all_pairs.parquet")
    all_pairs_df.to_parquet(all_pairs_parquet_path, index=False)
    

    
    print(f"Saved all pairs to {all_pairs_parquet_path}")

    pair_lng_col, pair_lat_col = _resolve_xy_columns(
        all_pairs_df,
        x_candidates=["k_lng", "lng", "lon"],
        y_candidates=["k_lat", "lat"],
    )

    # Classify pairs rows into project vs buffer using k_lat/k_lng
    points_gdf = gpd.GeoDataFrame(
        all_pairs_df,
        geometry=gpd.points_from_xy(all_pairs_df[pair_lng_col], all_pairs_df[pair_lat_col]),
        crs="EPSG:4326"
    )

    points_in_buffer = gpd.sjoin(points_gdf, buffer_gdf, how="inner", predicate='within')
    points_in_project = gpd.sjoin(points_gdf, project_area_gdf, how="inner", predicate='within')
    print(f"Pairs rows in buffer: {len(points_in_buffer)}")
    print(f"Pairs rows in project: {len(points_in_project)}")

    # S series: yearly mean proportion since start year, values >1 set to 0
    s_rows = []
    for year in years_since_start:
        print(f"Processing S values for year {year}...")
        s_prop_col = f"s_prop_{year}"
        s_luc_col = f"s_luc_{year}"

        if s_prop_col in points_in_buffer.columns and s_prop_col in points_in_project.columns:
            buffer_vals = points_in_buffer[s_prop_col].copy()
            project_vals = points_in_project[s_prop_col].copy()
        elif s_luc_col in points_in_buffer.columns and s_luc_col in points_in_project.columns:
            buffer_vals = points_in_buffer[s_luc_col].copy()
            project_vals = points_in_project[s_luc_col].copy()
        else:
            continue
        buffer_vals = buffer_vals[buffer_vals <= 1]
        project_vals = project_vals[project_vals <= 1]
        buffer_proportion = buffer_vals.mean(skipna=True) if len(points_in_buffer) > 0 else 0
        print(f"Year {year}: Buffer S proportion: {buffer_proportion:.4f}")
        project_proportion = project_vals.mean(skipna=True) if len(points_in_project) > 0 else 0
        print(f"Year {year}: Project S proportion: {project_proportion:.4f}")
        if pd.isna(buffer_proportion):
            buffer_proportion = 0
        if pd.isna(project_proportion):
            project_proportion = 0

        buffer_area_ha_year = buffer_proportion * buffer_start_area_ha
        project_area_ha_year = project_proportion * project_start_area_ha
        buffer_deforestation_ha = buffer_start_area_ha - buffer_area_ha_year
        project_deforestation_ha = project_start_area_ha - project_area_ha_year
        buffer_change_ha = buffer_area_ha_year - buffer_start_area_ha
        project_change_ha = project_area_ha_year - project_start_area_ha

        s_rows.append({
            "year": year,
            "buffer_proportion_s": buffer_proportion,
            "project_proportion_s": project_proportion,
            "buffer_area_s": buffer_area_ha_year,
            "project_area_s": project_area_ha_year,
            "buffer_deforestation_s": buffer_deforestation_ha,
            "project_deforestation_s": project_deforestation_ha,
            "buffer_change_s": buffer_change_ha,
            "project_change_s": project_change_ha,
        })

    s_df = pd.DataFrame(s_rows)

    # Merge K and S outputs by year
    merged_df = pd.merge(k_df, s_df, on="year", how="outer").sort_values("year")

    if "buffer_area_k" in merged_df.columns and "buffer_area_s" in merged_df.columns:
        merged_df["leakage_area_ha"] = merged_df["buffer_area_k"] - merged_df["buffer_area_s"]
    if "project_area_k" in merged_df.columns and "project_area_s" in merged_df.columns:
        merged_df["additionality_area_ha"] = merged_df["project_area_k"] - merged_df["project_area_s"]

    cols = merged_df.columns.tolist()
    if "year" in cols:
        cols.remove("year")
        merged_df = merged_df[["year"] + sorted(cols)]

    # Export full output
    output_csv_path = os.path.join(project_folder, "forest_cover_change.csv")
    merged_df.to_csv(output_csv_path, index=False)
    print(f"Exported full forest cover change to {output_csv_path}")

    # Export just undisturbed forest change since start year (what was requested)
    change_cols = [
        col for col in [
            "year",
            "buffer_change_k",
            "project_change_k",
            "buffer_change_s",
            "project_change_s",
        ]
        if col in merged_df.columns
    ]
    summary_df = merged_df[change_cols] if change_cols else pd.DataFrame({"year": merged_df["year"]})
    summary_csv_path = os.path.join(project_folder, "summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Exported undisturbed forest change summary to {summary_csv_path}")

if __name__ == "__main__":
    main()