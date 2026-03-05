import argparse
import json

from geopandas import gpd  # type: ignore
from shapely.ops import unary_union
from shapely.geometry import Point

def generate_country_from_centroid(
    project_geojson: str,
    countries_vector_filename: str,
    output_filename: str
) -> None:
    project_gdf = gpd.read_file(project_geojson)
    if project_gdf.empty:
        raise ValueError("Project geojson is empty")

    # Compute centroid
    unified = unary_union(project_gdf.geometry.values)
    centroid = Point(unified.centroid.x, unified.centroid.y)

    countries = gpd.read_file(countries_vector_filename)

    # Ensure same CRS
    if project_gdf.crs != countries.crs:
        centroid_gdf = gpd.GeoDataFrame(geometry=[centroid], crs=project_gdf.crs).to_crs(countries.crs)
        centroid = centroid_gdf.geometry.values[0]

    # Find containing country
    containing = countries[countries.contains(centroid)]
    if containing.empty:
        # Find nearest country
        countries_copy = countries.copy()
        countries_copy['distance'] = countries_copy.geometry.distance(centroid)
        nearest = countries_copy.loc[countries_copy['distance'].idxmin()]
        country_code = nearest['ISO_A2']
        print(f"Centroid not in any country, using nearest: {country_code}")
    else:
        country_code = containing.iloc[0]['ISO_A2']

    with open(output_filename, "w", encoding="utf-8") as outfd:
        outfd.write(json.dumps([country_code]))

def main() -> None:
    parser = argparse.ArgumentParser(description="Finds the country code for the centroid of the project")
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        dest="project_geojson",
        help="GeoJSON File of project boundary."
    )
    parser.add_argument(
        "--countries",
        type=str,
        required=True,
        dest="countries_vector_filename",
        help="File of country vector shapes."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="output_filename",
        help="JSON file listing country code."
    )
    args = parser.parse_args()

    generate_country_from_centroid(
        args.project_geojson,
        args.countries_vector_filename,
        args.output_filename,
    )

if __name__ == "__main__":
    main()