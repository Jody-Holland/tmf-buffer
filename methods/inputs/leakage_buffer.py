#!/usr/bin/env python3
"""
Crop a project's boundary to the country containing its centroid:
 - read project vector (any file fiona supports)
 - compute centroid (in EPSG:4326)
 - choose a metric CRS (UTM) based on centroid
 - buffer the project by 10000 m (configurable)
 - intersect buffer with the country polygon containing (or nearest to) the centroid
 - write cropped geometry to output file
"""
from pathlib import Path
import argparse

import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Point
from pyproj import CRS


def utm_crs_for_lonlat(lon, lat):
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def load_countries(countries_path=None):
    if countries_path:
        return gpd.read_file(countries_path)
    return gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))


def main():
    p = argparse.ArgumentParser(description="Buffer a project and crop to centroid country")
    p.add_argument("--project", help="project vector file (shp/geojson/gpkg etc.)")
    p.add_argument("-o", "--out", default=None, help="output path (default: <project>_cropped.geojson)")
    p.add_argument("-b", "--buffer", type=float, default=10000.0, help="buffer in meters (default 10000)")
    p.add_argument("--countries", help="optional country vector file to use instead of naturalearth")
    p.add_argument("--layer", help="layer name (for files with multiple layers)")
    args = p.parse_args()

    project_path = Path(args.project)
    if not project_path.exists():
        raise SystemExit(f"Project file not found: {project_path}")

    proj_gdf = gpd.read_file(args.project, layer=args.layer)  # read file
    if proj_gdf.empty:
        raise SystemExit("Project file contains no geometries")

    # unify geometry and compute centroid in WGS84
    proj_wgs = proj_gdf.to_crs(epsg=4326)
    unified = unary_union(proj_wgs.geometry.values)
    centroid_wgs = Point(unified.centroid.x, unified.centroid.y)

    # select metric CRS (UTM based on centroid)
    utm_crs = utm_crs_for_lonlat(centroid_wgs.x, centroid_wgs.y)

    # reproject to metric CRS and buffer
    proj_metric = gpd.GeoSeries([unified], crs="EPSG:4326").to_crs(utm_crs.to_string())
    buffered = proj_metric.geometry.values[0].buffer(args.buffer)

    # load countries and find the matching country polygon
    countries = load_countries(args.countries)
    countries_wgs = countries.to_crs(epsg=4326)

    # try contains first
    containing = countries_wgs[countries_wgs.contains(centroid_wgs)]
    if not containing.empty:
        country = containing.iloc[[0]]
    else:
        # none contains the centroid (e.g., centroid in ocean) -> pick nearest
        # compute distances in metric CRS to be accurate
        countries_metric = countries_wgs.to_crs(utm_crs.to_string())
        centroid_metric = gpd.GeoSeries([centroid_wgs], crs="EPSG:4326").to_crs(utm_crs.to_string()).geometry.values[0]
        countries_metric["dist_to_centroid"] = countries_metric.geometry.distance(centroid_metric)
        country = countries_metric.sort_values("dist_to_centroid").iloc[[0]].to_crs(epsg=4326)

    # intersect buffered area with chosen country (do all ops in metric CRS)
    country_metric = country.to_crs(utm_crs.to_string())
    intersection = buffered.intersection(country_metric.geometry.values[0])

    # prepare output GeoDataFrame in WGS84
    out_gdf = gpd.GeoDataFrame({"source_project": [project_path.name]}, geometry=[intersection], crs=utm_crs.to_string()).to_crs(epsg=4326)

    out_path = args.out or (project_path.with_suffix("") .name + "_cropped.geojson")
    out_path = Path(out_path)
    out_gdf.to_file(out_path, driver="GeoJSON")

    print(f"Centroid (lon,lat): {centroid_wgs.x:.6f},{centroid_wgs.y:.6f}")
    print(f"Used country: {country.iloc[0].get('name','(unknown)')}")
    print(f"Buffered by: {args.buffer} m, written: {out_path}")


if __name__ == "__main__":
    main()