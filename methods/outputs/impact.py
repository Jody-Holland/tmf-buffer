#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import argparse
import os
from tqdm.auto import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_folder')
    parser.add_argument('--project_geojson') 
    parser.add_argument('--start_year', type=int)
    parser.add_argument('--end_year', type=int)
    parser.add_argument('--output_folder')
    args = parser.parse_args()
    
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Load data
    project_gdf = gpd.read_file(args.project_geojson)
    buffered_gdf = gpd.read_file(os.path.join(args.project_folder, "buffered_project.geojson"))
    # Extract Project name from geojson filename (assumes format like "projectname.geojson")
    project_geojson = os.path.splitext(os.path.basename(args.project_geojson))[-2]
    print(f"Loaded project '{project_geojson}' and buffered geometries.")
    print(f"Loaded project and buffered geometries.")
    print(f"Project area (ha): {project_gdf.geometry.to_crs(epsg=6933).area.sum() / 10000:,.2f}")
    print(f"Buffered area (ha): {buffered_gdf.geometry.to_crs(epsg=6933).area.sum() / 10000:,.2f}")

    pairs_files = [f for f in os.listdir(os.path.join(args.project_folder, "pairs_1000")) 
                   if f.endswith(".parquet") and not f.endswith("_matchless.parquet")]
    pairs_dfs = []
    for file in tqdm(pairs_files, desc="Reading pairs", unit="file"):
        table = pq.read_table(os.path.join(args.project_folder, "pairs_1000", file))
        pairs_dfs.append(table.to_pandas())
    pairs_df = pd.concat(pairs_dfs, ignore_index=True)
    
    # Process spatial data
    buffered_cropped_gdf = gpd.overlay(buffered_gdf, project_gdf, how='difference')
    
    geometry = gpd.points_from_xy(pairs_df['k_lng'], pairs_df['k_lat'])
    pairs_gdf = gpd.GeoDataFrame(pairs_df, geometry=geometry, crs=project_gdf.crs)
    
    buffer_pairs_gdf = gpd.sjoin(pairs_gdf, buffered_cropped_gdf, how='inner', predicate='within')
    project_pairs_gdf = gpd.sjoin(pairs_gdf, project_gdf, how='inner', predicate='within')
    
    # Classify outcomes
    def classify_outcome(row):
        k_start = f'k_luc_{args.start_year}'
        k_end = f'k_luc_{args.end_year}'
        s_end = f's_luc_{args.end_year}'
        
        if row[k_start] == 1 and row[k_end] == 1 and row[s_end] != 1:
            return 'avoided_def'
        elif row[k_start] == 1 and row[k_end] != 1 and row[s_end] == 1:
            return 'excess_def'
        else:
            return "no_effect"
    
    # progress_apply isn't present on GeoDataFrame in some environments; use tqdm over iterrows
    buffer_pairs_gdf['outcome'] = [
        classify_outcome(row) for _, row in tqdm(buffer_pairs_gdf.iterrows(), total=len(buffer_pairs_gdf), desc="Classify buffer")
    ]
    print(f"Buffer area outcomes:\n{buffer_pairs_gdf['outcome'].value_counts()}")
    
    project_pairs_gdf['outcome'] = [
        classify_outcome(row) for _, row in tqdm(project_pairs_gdf.iterrows(), total=len(project_pairs_gdf), desc="Classify project")
    ]
    print(f"Project area outcomes:\n{project_pairs_gdf['outcome'].value_counts()}")
    
    # PLOT 1 & 2: Points-only map (omit 'no_effect' points to avoid overplotting) and Points + Project-boundary overlay
    combined_points = pd.concat([buffer_pairs_gdf, project_pairs_gdf], ignore_index=True)
    
    # Drop the "duff" / no_effect points to avoid crashing the plot
    avoided = combined_points[combined_points['outcome'] == 'avoided_def']
    excess = combined_points[combined_points['outcome'] == 'excess_def']
    
    # larger legend icons and font
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=12, label='Avoided Deforestation'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=12, label='Excess Deforestation'),
    ]
    legend_fontsize = 18
    title_fontsize = 24
    
    # Map A: points only (no project/buffer boundaries)
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    avoided.plot(ax=ax, color='blue', markersize=0.5, label='avoided_def', alpha=0.5)
    excess.plot(ax=ax, color='orange', markersize=0.5, label='excess_def', alpha=0.5)
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=legend_fontsize)
    plt.title(f'Project {project_geojson}, Conservation Outcomes ({args.start_year}-{args.end_year})', fontsize=title_fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_folder, 'spatial_map_points_only.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Map B: same points, with project boundaries overlaid as dotted line
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    avoided.plot(ax=ax, color='blue', markersize=0.2, label='avoided_def', alpha=0.5)
    excess.plot(ax=ax, color='orange', markersize=0.2, label='excess_def', alpha=0.5)
    project_gdf.boundary.plot(ax=ax, linewidth=1.0, edgecolor='black', alpha =0.7, linestyle='--', label='Project Boundary')
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=legend_fontsize)
    plt.title(f'Project {project_geojson}, Conservation Outcomes (with project boundary) ({args.start_year}-{args.end_year})', fontsize=title_fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_folder, 'spatial_map_boundaries_overlay.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()