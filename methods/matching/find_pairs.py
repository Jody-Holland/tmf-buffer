#!/usr/bin/env python3

import argparse
import glob
import os
import logging
import re
import gc
import datetime
import psutil
import sys
import json
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count, set_start_method
from shapely.geometry import shape
from pyproj import Geod
from numba import jit
from methods.common.luc import luc_matching_columns
from methods.common import LandUseClass
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed

# Constants
HARD_COLUMN_COUNT = 5  # Number of categorical variables requiring exact match

# Continuous covariates used in Euclidean distance
DISTANCE_COLUMNS = [
    "elevation", "slope", "access",
    "fcc0_u", "fcc0_d",
    "fcc5_u", "fcc5_d",
    "fcc10_u", "fcc10_d"
]

def calculate_processors_for_k_grid_size(k_grid_size: int) -> int:
    """
    Calculate the number of processors to use based on K grid size.
    
    The goal is to ensure no more than ~320,000 pixels are processed concurrently,
    while scaling smoothly and capping at 32 processors.
    
    Args:
        k_grid_size: Number of rows in the first K grid
        
    Returns:
        Number of processors to use (1-32)
    """
    # Target maximum concurrent pixels across all processors
    max_concurrent_pixels = 320000
    
    # Calculate ideal processor count
    processors = max_concurrent_pixels // k_grid_size
    
    # Cap at 32 and ensure at least 1
    processors = max(1, min(32, processors))
    
    return processors

DEBUG = False
MEMORY_LOG_FILE = None


def _worker_init():
    """Initializer for multiprocessing workers — sets up logging and unbuffered stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [PID %(process)d] %(message)s",
        stream=sys.stdout,
    )
    # Force line-buffered stdout so print() shows up immediately
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    elif not sys.stdout.line_buffering:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)


def setup_memory_logging(output_folder):
    """Set up memory logging to file in output directory."""
    global MEMORY_LOG_FILE
    MEMORY_LOG_FILE = os.path.join(output_folder, "memory_usage_log.txt")
    
    # Initialise log file with header
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(MEMORY_LOG_FILE, 'w') as f:
        f.write(f"Memory Usage Log - Started at {timestamp}\n")
        f.write("="*50 + "\n\n")

def log_memory_usage(label="", detailed=False):
    """Log memory usage of the process to both console and file."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        rss_mb = memory_info.rss / 1024 / 1024
        vms_mb = memory_info.vms / 1024 / 1024
        
        log_msg = f"Memory {label}: RSS {rss_mb:.1f} MB"
        logging.info(log_msg)
        
        if detailed:
            detailed_msg = f"Memory {label}: RSS {rss_mb:.1f} MB, VMS {vms_mb:.1f} MB"
            logging.info(detailed_msg)
        
        # Write to memory log file
        if MEMORY_LOG_FILE:
            timestamp = datetime.datetime.now().strftime('%H:%M:%S')
            with open(MEMORY_LOG_FILE, 'a') as f:
                f.write(f"[{timestamp}] PID {os.getpid()}: {log_msg}\n")
                
        return rss_mb
    except:
        return 0

def log_memory_change(label_before, label_after, memory_before=None):
    """Log memory change between two points."""
    memory_after = psutil.Process().memory_info().rss / 1024 / 1024
    
    if memory_before is not None:
        change = memory_after - memory_before
        sign = "+" if change >= 0 else ""
        change_msg = f"Memory {label_after}: {memory_after:.1f} MB ({sign}{change:.1f} MB from {label_before})"
    else:
        change_msg = f"Memory {label_after}: {memory_after:.1f} MB"
        
    logging.info(change_msg)
    
    # Write to memory log file
    if MEMORY_LOG_FILE:
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        with open(MEMORY_LOG_FILE, 'a') as f:
            f.write(f"[{timestamp}] PID {os.getpid()}: {change_msg}\n")
    
    return memory_after

def log_dataframe_memory(df, df_name):
    """Log memory usage of a DataFrame."""
    total_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
    shape_info = f"{df.shape[0]:,} rows × {df.shape[1]} cols"
    memory_msg = f"{df_name} DataFrame: {total_memory:.1f} MB ({shape_info})"
    
    logging.info(memory_msg)
    
    # Write to memory log file
    if MEMORY_LOG_FILE:
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        with open(MEMORY_LOG_FILE, 'a') as f:
            f.write(f"[{timestamp}] PID {os.getpid()}: {memory_msg}\n")

def log_array_memory(arr, arr_name):
    """Log memory usage of a numpy array."""
    if hasattr(arr, 'nbytes'):
        memory_mb = arr.nbytes / 1024 / 1024
        if memory_mb > 1:  # Only log arrays > 1MB
            memory_msg = f"{arr_name} array: {memory_mb:.1f} MB ({arr.shape})"
            logging.info(memory_msg)
            
            if MEMORY_LOG_FILE:
                timestamp = datetime.datetime.now().strftime('%H:%M:%S')
                with open(MEMORY_LOG_FILE, 'a') as f:
                    f.write(f"[{timestamp}] PID {os.getpid()}: {memory_msg}\n")

def log_system_memory():
    """Log overall system memory usage."""
    try:
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        system_msg = (f"System Memory: {memory.used / 1024**3:.1f}GB used / "
                     f"{memory.total / 1024**3:.1f}GB total ({memory.percent:.1f}%), "
                     f"Swap: {swap.used / 1024**3:.1f}GB / {swap.total / 1024**3:.1f}GB")
        
        logging.info(system_msg)
        
        # Write to memory log file
        if MEMORY_LOG_FILE:
            timestamp = datetime.datetime.now().strftime('%H:%M:%S')
            with open(MEMORY_LOG_FILE, 'a') as f:
                f.write(f"[{timestamp}] SYSTEM: {system_msg}\n")
    except:
        pass

def find_match_iteration(
    m_sample_filename: str,
    start_year: int,
    evaluation_year: int,
    output_folder: str,
    k_grid_filepath_and_seed: tuple,
    shuffle_seed: int,
    max_potential_matches: int
) -> str:
    """
    Process a single K grid file to find matching S pixels from the M set.
    
    Returns:
        K grid ID
    """
    k_grid_filepath, seed = k_grid_filepath_and_seed
    k_grid_filename = os.path.basename(k_grid_filepath)

    # Extract identifier from k_grid filename
    match = re.search(r'\d+', k_grid_filename)
    k_grid_id = match.group(0) if match else k_grid_filename.replace('.parquet', '')

    print(f"Starting iteration for K grid: {k_grid_filename} (ID: {k_grid_id})", flush=True)
    logging.info(f"Starting iteration for K grid: {k_grid_filename} (ID: {k_grid_id}) with seed {seed}")
    log_memory_usage(f"iteration start for {k_grid_id}", detailed=True)
    
    rng = np.random.default_rng(seed)

    # Load K grid
    logging.info(f"Loading K grid from {k_grid_filepath}")
    k_subset = pd.read_parquet(k_grid_filepath).reset_index(drop=True)
    log_dataframe_memory(k_subset, "K grid")
    print(f"Loaded K grid with {len(k_subset)} rows", flush=True)

    # Load pre-sampled M set
    logging.info(f"Loading pre-sampled M set from {m_sample_filename}")
    m_set = pd.read_parquet(m_sample_filename)
    log_dataframe_memory(m_set, "M set (pre-sampled)")
    print(f"Loaded M set with {len(m_set)} rows", flush=True)
    
    # Shuffle M set with the provided shuffle seed
    logging.info(f"Shuffling M set for iteration {k_grid_id} with shuffle seed {shuffle_seed}...")
    shuffle_rng = np.random.default_rng(shuffle_seed)
    m_set = m_set.sample(frac=1, random_state=shuffle_rng).reset_index(drop=True)
    print("Shuffled M set", flush=True)
    logging.info("Shuffled M set")
    
    # Thresholds for normalising continuous variables
    thresholds_for_columns = np.array([
        200.0,  # Elevation (metres)
        2.5,    # Slope (degrees)
        10.0,   # Accessibility (cost units)
        0.1,    # Forest cover change metrics
        0.1,    # FCCs are normalised with smaller thresholds
        0.1,    # as they're already on a 0-1 scale
        0.1,
        0.1,
        0.1,
    ])

    # Normalise continuous variables by dividing by thresholds
    m_dist_thresholded_df = m_set[DISTANCE_COLUMNS] / thresholds_for_columns
    k_subset_dist_thresholded_df = k_subset[DISTANCE_COLUMNS] / thresholds_for_columns
    print("Normalized continuous variables", flush=True)
    logging.info("Normalized continuous variables")

    # Convert to contiguous arrays for faster numba processing
    m_dist_thresholded = np.ascontiguousarray(m_dist_thresholded_df, dtype=np.float32)
    k_subset_dist_thresholded = np.ascontiguousarray(k_subset_dist_thresholded_df, dtype=np.float32)

    # Get land use class column names for exact matching
    luc0, luc5, luc10 = luc_matching_columns(start_year)
    luc_columns = [x for x in m_set.columns if x.startswith('luc')]

    # Columns requiring exact matches (categorical variables)
    hard_match_columns = ['country', 'ecoregion', luc10, luc5, luc0]

    # Prepare categorical variables as integer arrays
    m_dist_hard = np.ascontiguousarray(m_set[hard_match_columns].to_numpy()).astype(np.int32)
    k_subset_dist_hard = np.ascontiguousarray(k_subset[hard_match_columns].to_numpy()).astype(np.int32)

    # Free intermediate dataframes
    del m_dist_thresholded_df, k_subset_dist_thresholded_df
    gc.collect()

    logging.info("Running make_s_set_mask...")
    # Create random starting positions to avoid bias in candidate selection
    starting_positions = rng.integers(0, int(m_dist_thresholded.shape[0]), int(k_subset_dist_thresholded.shape[0]))
    s_set_mask_true, no_potentials = make_s_set_mask(
        m_dist_thresholded,
        k_subset_dist_thresholded,
        m_dist_hard,
        k_subset_dist_hard,
        starting_positions,
        max_potential_matches
    )
    print(f"Created S set mask: {np.sum(s_set_mask_true)} potential matches, {np.sum(no_potentials)} K pixels with no matches", flush=True)
    logging.info(f"Created S set mask: {np.sum(s_set_mask_true)} potential matches, {np.sum(no_potentials)} K pixels with no matches")

    # Create S set (candidate matches) from M pixels that passed filtering
    s_set = m_set[s_set_mask_true]
    # Remove K pixels that have no potential matches
    potentials = np.invert(no_potentials)
    excluded_k = k_subset[no_potentials]  # Collect excluded K pixels
    k_subset = k_subset[potentials]
    print(f"S set has {len(s_set)} rows, K subset has {len(k_subset)} rows, {len(excluded_k)} K pixels excluded in potential sweep", flush=True)
    logging.info(f"S set has {len(s_set)} rows, K subset has {len(k_subset)} rows, {len(excluded_k)} K pixels excluded in potential sweep")

    log_dataframe_memory(s_set, "S set")
    log_dataframe_memory(k_subset, "K subset (filtered)")

    results = []
    matchless = []

    # Calculate covariance matrix for Mahalanobis distance
    logging.info("Calculating covariance matrix...")
    s_set_for_cov = s_set[DISTANCE_COLUMNS]
    covarience = np.cov(s_set_for_cov, rowvar=False)
    invconv = np.linalg.inv(covarience).astype(np.float32)
    print("Calculated inverse covariance matrix", flush=True)
    logging.info("Calculated inverse covariance matrix")

    # Free covariance intermediate data
    del s_set_for_cov
    gc.collect()

    # Prepare data arrays for matching algorithm
    s_set_match = s_set[hard_match_columns + DISTANCE_COLUMNS].to_numpy(dtype=np.float32)
    s_set_match = np.ascontiguousarray(s_set_match)

    k_subset_match = k_subset[hard_match_columns + DISTANCE_COLUMNS].to_numpy(dtype=np.float32)
    k_subset_match = np.ascontiguousarray(k_subset_match)
    print("Prepared categorical and continuous arrays", flush=True)
    logging.info("Prepared arrays for matching")

    logging.info("Starting greedy matching...")
    # Perform greedy matching with random order of K pixels
    add_results, k_idx_matchless = greedy_match_with_shuffled_k(
        k_subset_match,
        s_set_match,
        invconv,
        rng
    )
    print(f"Greedy matching completed: {len(add_results)} matches, {len(k_idx_matchless)} matchless", flush=True)
    logging.info(f"Greedy matching completed: {len(add_results)} matches, {len(k_idx_matchless)} matchless")

    # Store match results
    for result in add_results:
        (k_idx, s_idx) = result
        k_row = k_subset.iloc[k_idx]
        match = s_set.iloc[s_idx]

        # Verify hard matches if debug is enabled
        if DEBUG:
            for hard_match_column in hard_match_columns:
                if k_row[hard_match_column] != match[hard_match_column]:
                    raise ValueError("Hard match inconsistency")

        # Collect matched pairs data
        results.append(
            [k_row.lat, k_row.lng] + [k_row[x] for x in luc_columns + DISTANCE_COLUMNS] + \
            [match.lat, match.lng] + [match[x] for x in luc_columns + DISTANCE_COLUMNS]
        )

    # After building the results list, add this before saving matchless:
    if results:
        # Define column names for matched pairs (adjust based on your data structure)
        columns = (
            ['k_lat', 'k_lng'] + [f'k_{x}' for x in luc_columns + DISTANCE_COLUMNS] +
            ['s_lat', 's_lng'] + [f's_{x}' for x in luc_columns + DISTANCE_COLUMNS]
        )
        results_df = pd.DataFrame(results, columns=columns)
        results_df.to_parquet(os.path.join(output_folder, f'{k_grid_id}.parquet'))  # Save matched pairs
        print(f"Saved {len(results_df)} matched pairs to {os.path.join(output_folder, f'{k_grid_id}.parquet')}", flush=True)
        logging.info(f"Saved {len(results_df)} matched pairs to {os.path.join(output_folder, f'{k_grid_id}.parquet')}")
    else:
        print(f"No matched pairs for K grid {k_grid_id}", flush=True)
        logging.info(f"No matched pairs for K grid {k_grid_id}")

    # Existing matchless saving code remains unchanged
    matchless_df = pd.DataFrame(matchless, columns=k_subset.columns)
    matchless_df.to_parquet(os.path.join(output_folder, f'{k_grid_id}_matchless.parquet'))
    print(f"Saved {len(matchless_df)} matchless (including {len(excluded_k)} from potential sweep) to {os.path.join(output_folder, f'{k_grid_id}_matchless.parquet')}", flush=True)
    logging.info(f"Saved {len(matchless_df)} matchless (including {len(excluded_k)} from potential sweep) to {os.path.join(output_folder, f'{k_grid_id}_matchless.parquet')}")

    # Clean up large dataframes to free memory
    del m_set, s_set, k_subset, results_df, matchless_df
    del m_dist_thresholded, k_subset_dist_thresholded, m_dist_hard, k_subset_dist_hard
    del s_set_match, k_subset_match, invconv
    gc.collect()
    
    print(f"Completed iteration for K grid {k_grid_id}", flush=True)
    logging.info(f"Completed iteration for K grid {k_grid_id}")
    
    return k_grid_id

@jit(nopython=True, fastmath=True, error_model="numpy")
def make_s_set_mask(
    m_dist_thresholded: np.ndarray,
    k_subset_dist_thresholded: np.ndarray,
    m_dist_hard: np.ndarray,
    k_subset_dist_hard: np.ndarray,
    starting_positions: np.ndarray,
    max_potential_matches: int
):
    """Create a boolean mask for M pixels that are potential matches for K pixels."""
    m_size = m_dist_thresholded.shape[0]
    k_size = k_subset_dist_thresholded.shape[0]

    # Boolean masks to track which M pixels to include and which K pixels have no matches
    s_include = np.zeros(m_size, dtype=np.bool_)
    k_miss = np.zeros(k_size, dtype=np.bool_)

    # Process each K pixel
    for k in range(k_size):
        matches = 0
        k_row = k_subset_dist_thresholded[k, :]
        k_hard = k_subset_dist_hard[k]

        # Loop through M pixels, starting from a random position
        for index in range(m_size):
            # Use modulo to wrap around to start of array when reaching the end
            m_index = (index + starting_positions[k]) % m_size

            m_row = m_dist_thresholded[m_index, :]
            m_hard = m_dist_hard[m_index]

            should_include = True

            # Check for exact match on categorical variables
            hard_equals = True
            for j in range(m_hard.shape[0]):
                if m_hard[j] != k_hard[j]:
                    hard_equals = False

            if not hard_equals:
                should_include = False
            else:
                # Check for threshold match on continuous variables
                for j in range(m_row.shape[0]):
                    if abs(m_row[j] - k_row[j]) > 1.0:
                        should_include = False

            # If all criteria are met, include this M pixel
            if should_include:
                s_include[m_index] = True
                matches += 1

            # Stop once we've found enough matches for this K pixel
            if matches == max_potential_matches:
                break

        # Mark K pixels that have no potential matches
        k_miss[k] = matches == 0

    return s_include, k_miss

@jit(nopython=True, fastmath=True, error_model="numpy")
def rows_all_true(rows: np.ndarray):
    """Check if all values in each row are True."""
    all_true = np.ones((rows.shape[0],), dtype=np.bool_)
    for row_idx in range(rows.shape[0]):
        for col_idx in range(rows.shape[1]):
            if not rows[row_idx, col_idx]:
                all_true[row_idx] = False
                break
    return all_true

def build_categorical_index(s_set: np.ndarray, k_subset: np.ndarray):
    """Pre-compute which S indices match each K pixel's categorical values.
    
    Returns:
        A list where each element i contains an array of S indices that match 
        k_subset[i]'s categorical columns.
    """
    print("Building categorical index for fast matching...", flush=True)
    logging.info("Building categorical index for fast matching...")
    
    # Extract categorical columns
    s_categorical = s_set[:, :HARD_COLUMN_COUNT].astype(np.int32)
    k_categorical = k_subset[:, :HARD_COLUMN_COUNT].astype(np.int32)
    
    # Build dictionary mapping categorical tuples to S indices
    categorical_dict = {}
    for s_idx in range(s_categorical.shape[0]):
        key = tuple(s_categorical[s_idx])
        if key not in categorical_dict:
            categorical_dict[key] = []
        categorical_dict[key].append(s_idx)
    
    # Convert lists to numpy arrays for faster access
    for key in categorical_dict:
        categorical_dict[key] = np.array(categorical_dict[key], dtype=np.int32)
    
    # Build index for each K pixel
    k_to_s_indices = []
    for k_idx in range(k_categorical.shape[0]):
        key = tuple(k_categorical[k_idx])
        if key in categorical_dict:
            k_to_s_indices.append(categorical_dict[key])
        else:
            k_to_s_indices.append(np.array([], dtype=np.int32))
    
    print(f"Built categorical index: {len(categorical_dict)} unique categorical combinations", flush=True)
    logging.info(f"Built categorical index: {len(categorical_dict)} unique categorical combinations")
    
    return k_to_s_indices

@jit(nopython=True, fastmath=True, error_model="numpy")
def greedy_match_core_optimized(
    k_subset: np.ndarray,
    s_set: np.ndarray,
    invcov: np.ndarray,
    k_order: np.ndarray,
    k_to_s_indices_flat: np.ndarray,
    k_to_s_offsets: np.ndarray
):
    """Optimized core greedy matching algorithm using pre-computed categorical indices.
    
    Args:
        k_subset: K pixel features
        s_set: S pixel features
        invcov: Inverse covariance matrix
        k_order: Order in which to process K pixels
        k_to_s_indices_flat: Flattened array of all S indices
        k_to_s_offsets: Offset array where [k_to_s_offsets[i]:k_to_s_offsets[i+1]] 
                        gives S indices for K pixel i
    """
    # Track which S pixels are still available for matching
    s_available = np.ones((s_set.shape[0],), dtype=np.bool_)
    total_available = s_set.shape[0]

    results = []
    matchless = []

    # Process K pixels in the specified order
    for k_idx_original in k_order:
        k_row = k_subset[k_idx_original, :]
        
        # Get pre-computed S indices that match this K pixel's categorical values
        start_offset = k_to_s_offsets[k_idx_original]
        end_offset = k_to_s_offsets[k_idx_original + 1]
        candidate_s_indices = k_to_s_indices_flat[start_offset:end_offset]
        
        if len(candidate_s_indices) == 0 or total_available == 0:
            # No matches available
            matchless.append(k_idx_original)
            continue
        
        # Filter to only available S pixels
        available_candidates = candidate_s_indices[s_available[candidate_s_indices]]
        
        if len(available_candidates) == 0:
            # No available matches
            matchless.append(k_idx_original)
            continue
        
        # Calculate Mahalanobis distances only for available candidates
        candidate_features = s_set[available_candidates, HARD_COLUMN_COUNT:]
        distances = batch_mahalanobis_squared(
            candidate_features, 
            k_row[HARD_COLUMN_COUNT:], 
            invcov
        )
        
        # Find S pixel with minimum distance
        min_dist_local_idx = np.argmin(distances)
        s_idx = available_candidates[min_dist_local_idx]
        
        # Store the match and remove S pixel from available pool
        results.append((k_idx_original, s_idx))
        s_available[s_idx] = False
        total_available -= 1

    return (results, matchless)

def greedy_match_with_shuffled_k(
    k_subset: np.ndarray,
    s_set: np.ndarray,
    invcov: np.ndarray,
    rng: np.random.Generator
):
    """Wrapper for greedy matching with shuffled K pixel order."""
    k_order = np.arange(k_subset.shape[0])
    rng.shuffle(k_order)
    
    # Build categorical index for optimization
    k_to_s_indices = build_categorical_index(s_set, k_subset)
    
    # Flatten the list of arrays into a single array with offsets
    # This allows passing to numba jitted function
    k_to_s_indices_flat = np.concatenate([arr for arr in k_to_s_indices if len(arr) > 0] + [np.array([], dtype=np.int32)])
    k_to_s_offsets = np.zeros(len(k_to_s_indices) + 1, dtype=np.int32)
    
    offset = 0
    for i, indices in enumerate(k_to_s_indices):
        k_to_s_offsets[i] = offset
        offset += len(indices)
    k_to_s_offsets[-1] = offset
    
    return greedy_match_core_optimized(k_subset, s_set, invcov, k_order, k_to_s_indices_flat, k_to_s_offsets)

@jit(nopython=True, fastmath=True, error_model="numpy")
def batch_mahalanobis_squared(rows, vector, invcov):
    """Calculate squared Mahalanobis distances between multiple rows and a single vector."""
    # Calculate difference between each row and the reference vector
    diff = rows - vector
    # Calculate Mahalanobis distance using the inverse covariance matrix
    dists = (np.dot(diff, invcov) * diff).sum(axis=1)
    return dists

def calculate_smd(k_values: np.ndarray, s_values: np.ndarray) -> float:
    """Calculate Standardised Mean Difference (SMD) between K and S values."""
    if len(k_values) == 0 or len(s_values) == 0:
        return np.nan
        
    # Calculate means
    mean_k = np.mean(k_values)
    mean_s = np.mean(s_values)
    
    # Calculate variances with n-1 degrees of freedom
    var_k = np.var(k_values, ddof=1) if len(k_values) > 1 else 0
    var_s = np.var(s_values, ddof=1) if len(s_values) > 1 else 0
    
    n_k = len(k_values)
    n_s = len(s_values)
    
    # Calculate pooled variance
    pooled_var = ((n_k - 1) * var_k + (n_s - 1) * var_s) / (n_k + n_s - 2)
    pooled_std = np.sqrt(pooled_var)
    
    # Avoid division by zero
    if pooled_std == 0:
        return 0.0
        
    # SMD = (mean_treatment - mean_control) / pooled_std
    return (mean_k - mean_s) / pooled_std

def analyse_matching_balance(output_folder: str, evaluation_year: int, luc_columns: list, distance_columns: list) -> pd.DataFrame:
    """Analyse balance between treatment and control groups after matching."""
    logging.info("Starting matching balance analysis...")
    
    # Find all matched pair files (excluding matchless files)
    match_files = glob.glob(os.path.join(output_folder, "*.parquet"))
    match_files = [f for f in match_files if not f.endswith("_matchless.parquet")]
    if not match_files:
        logging.warning("No matched parquet files found for balance analysis")
        return pd.DataFrame()
        
    logging.info(f"Found {len(match_files)} matched parquet files")
    
    # Initialize dictionaries to collect all K and S values
    all_k_data = {var: [] for var in distance_columns}
    all_s_data = {var: [] for var in distance_columns}
    
    # Process each matched pair file
    for match_file in match_files:
        try:
            df = pd.read_parquet(match_file)
            if len(df) == 0:
                continue
                
            # Collect values for each continuous variable
            for var in distance_columns:
                k_col = f'k_{var}'
                s_col = f's_{var}'
                if k_col in df.columns and s_col in df.columns:
                    all_k_data[var].extend(df[k_col].dropna().values)
                    all_s_data[var].extend(df[s_col].dropna().values)
        except Exception as e:
            logging.warning(f"Error processing {match_file}: {e}")
            continue
            
    # Calculate SMD for each variable
    smd_results = []
    for var in distance_columns:
        k_vals = np.array(all_k_data[var])
        s_vals = np.array(all_s_data[var])
        
        if len(k_vals) > 0 and len(s_vals) > 0:
            smd = calculate_smd(k_vals, s_vals)
            smd_results.append({
                'variable': var,
                'n_k': len(k_vals),
                'n_s': len(s_vals),
                'mean_k': np.mean(k_vals),
                'mean_s': np.mean(s_vals),
                'std_k': np.std(k_vals, ddof=1),
                'std_s': np.std(s_vals, ddof=1),
                'smd': smd,
                'abs_smd': abs(smd)
            })
        else:
            smd_results.append({
                'variable': var,
                'n_k': len(k_vals),
                'n_s': len(s_vals),
                'mean_k': np.nan,
                'mean_s': np.nan,
                'std_k': np.nan,
                'std_s': np.nan,
                'smd': np.nan,
                'abs_smd': np.nan
            })
            
    smd_df = pd.DataFrame(smd_results)
    
    # Add interpretation of SMD values
    def interpret_smd(smd_val):
        if np.isnan(smd_val):
            return "No data"
        abs_smd = abs(smd_val)
        if abs_smd < 0.1:
            return "Negligible"
        elif abs_smd < 0.2:
            return "Small"
        elif abs_smd < 0.5:
            return "Medium"
        elif abs_smd < 0.8:
            return "Large"
        else:
            return "Very large"
            
    smd_df['smd_interpretation'] = smd_df['smd'].apply(interpret_smd)
    logging.info("Completed matching balance analysis")
    return smd_df

def extract_k_grid_number(filepath: str) -> int:
    """Extract the numeric part from K grid filenames for proper sorting."""
    basename = os.path.basename(filepath)
    match = re.search(r'k_(\d+)\.parquet$', basename)
    if match:
        return int(match.group(1))
    else:
        logging.warning(f"Could not extract grid number from {basename}. Placing it at the end.")
        return float('inf')

def create_single_m_sample(args):
    """Create a single M sample file - designed for multiprocessing."""
    m_parquet_filename, sample_index, sample_size, seed, m_samples_dir = args
    
    logging.info(f"Creating sample {sample_index+1} with seed {seed}")
    
    # Load M set inside the process to avoid passing large DataFrame
    m_set = pd.read_parquet(m_parquet_filename)
    
    rng = np.random.default_rng(seed)
    
    # Use numpy for faster sampling
    should_replace = len(m_set) < sample_size
    if should_replace:
        indices = rng.choice(len(m_set), size=sample_size, replace=True)
    else:
        indices = rng.choice(len(m_set), size=sample_size, replace=False)
    
    sample = m_set.iloc[indices].reset_index(drop=True)
    
    # Shuffle the sampled rows using numpy permutation
    shuffle_indices = rng.permutation(len(sample))
    sample = sample.iloc[shuffle_indices].reset_index(drop=True)
    
    # Save sample
    sample_filename = os.path.join(m_samples_dir, f"m_sample_{sample_index+1:03d}.parquet")
    sample.to_parquet(sample_filename)
    
    # Clean up sample from memory
    del sample, m_set
    gc.collect()
    
    return f"Completed sample {sample_index+1}"

def create_m_samples(
    m_parquet_filename: str,  # Changed from m_set: pd.DataFrame
    output_folder: str,
    num_samples: int,
    sample_size: int,
    seeds: np.ndarray,
    sample_processes: int = None
) -> None:
    """Create pre-sampled M set files for each matching iteration using multiprocessing.

    Improvements:
    - If M set is small enough to fit in memory (configurable threshold), load once and generate all samples
      from memory (much faster and avoids repeated disk reads).
    - Otherwise fall back to the per-sample multiprocessing approach (each worker reads file).
    """
    if sample_processes is None:
        sample_processes = min(cpu_count(), 8)  # Reduced default to 8 to avoid memory issues
    
    # Create m_samples directory
    m_samples_dir = os.path.join(output_folder, "m_samples")
    os.makedirs(m_samples_dir, exist_ok=True)
    
    # Check which samples already exist
    existing_samples = []
    missing_samples = []
    
    for i in range(num_samples):
        sample_filename = os.path.join(m_samples_dir, f"m_sample_{i+1:03d}.parquet")
        if os.path.exists(sample_filename):
            existing_samples.append(i+1)
        else:
            missing_samples.append(i)
    
    if existing_samples:
        logging.info(f"Found {len(existing_samples)} existing M samples")
    
    if not missing_samples:
        logging.info("All M samples already exist. Skipping sample creation.")
        return

    logging.info(f"Preparing to create {len(missing_samples)} missing M set samples of {sample_size:,} rows each...")
    
    # Heuristic: if parquet file size is small enough, load into memory once
    try:
        file_bytes = os.path.getsize(m_parquet_filename)
    except Exception:
        file_bytes = float("inf")
    
    # Estimate safe in-memory threshold (adjustable); here ~6 GiB
    INMEM_THRESHOLD = 6 * 1024**3
    if file_bytes < INMEM_THRESHOLD:
        logging.info("Parquet file small enough to load into memory once. Using fast in-memory sampling.")
        m_set = pd.read_parquet(m_parquet_filename)
        n = len(m_set)
        log_dataframe_memory(m_set, "M set (loaded once for sampling)")

        # Pre-generate indices for missing samples to avoid repeated RNG overhead
        sample_jobs = []
        for i in missing_samples:
            seed = seeds[i]
            rng = np.random.default_rng(seed)
            replace = n < sample_size
            indices = rng.choice(n, size=sample_size, replace=replace)
            sample_jobs.append((i, indices))

        # Helper to write a single sample using pyarrow with multithreading
        def _write_sample(job):
            i, indices = job
            sample_filename = os.path.join(m_samples_dir, f"m_sample_{i+1:03d}.parquet")
            sample = m_set.iloc[indices].reset_index(drop=True)
            table = pa.Table.from_pandas(sample)

            # Try writing with threaded pyarrow; if it fails, retry without thread kw and finally fallback to pandas
            write_kwargs = {"compression": "snappy", "use_dictionary": False, "use_threads": True}
            try:
                pq.write_table(table, sample_filename, **write_kwargs)
            except TypeError as e:
                # Some builds don't accept 'use_threads' as an argument
                write_kwargs.pop("use_threads", None)
                try:
                    pq.write_table(table, sample_filename, **write_kwargs)
                except Exception:
                    # Last resort: pandas.to_parquet (safer)
                    sample.to_parquet(sample_filename, engine="pyarrow", compression=write_kwargs.get("compression"))
            finally:
                logging.info(f"Saved sample {i+1} -> {sample_filename}")
                del sample, table
                return sample_filename

        max_workers = min(8, len(sample_jobs), os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_write_sample, job): job[0] for job in sample_jobs}
            for fut in as_completed(futures):
                try:
                    _ = fut.result()
                except Exception as exc:
                    logging.error("Error writing sample %s: %s", futures[fut], exc)

        del m_set
        gc.collect()
        logging.info("Completed creating samples from in-memory M set.")
        return
    else:
        logging.warning(
            "Parquet file is large (~%.1f GB). Falling back to per-sample workers (will read file per-sample). "
            "Consider reducing sample_size or increasing INMEM_THRESHOLD if you have more RAM.",
            file_bytes / 1024**3
        )
    
    # Fallback: create samples in parallel, each worker reads the Parquet file (existing behaviour)
    sample_args = []
    for i in missing_samples:
        sample_args.append((m_parquet_filename, i, sample_size, seeds[i], m_samples_dir))
    
    with Pool(processes=sample_processes, initializer=_worker_init) as pool:
        results = pool.map(create_single_m_sample, sample_args)
    
    for result in results:
        logging.info(result)
    
    logging.info(f"Completed creating {len(missing_samples)} missing M set samples in {m_samples_dir}")

def iteration_func_wrapper(args):
    """Wrapper function for multiprocessing - must be at module level to be picklable."""
    m_sample_file, k_grid_file, iter_seed, shuf_seed, start_year, evaluation_year, output_folder, max_potential_matches = args
    return find_match_iteration(
        m_sample_file,
        start_year,
        evaluation_year,
        output_folder,
        (k_grid_file, iter_seed),
        shuf_seed,
        max_potential_matches
    )

def find_pairs(
    k_directory: str,
    m_parquet_filename: str,
    start_year: int,
    evaluation_year: int,
    seed: int,
    output_folder: str,
    batch_size: int,
    processes_count: int,
    max_potential_matches: int = 1000
) -> None:
    """Main matching function that processes all K grids."""
    logging.info("Starting find pairs")
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Setup memory logging
    setup_memory_logging(output_folder)
    log_memory_usage("main function start", detailed=True)
    log_system_memory()
    
    # Check if M set file exists
    if not os.path.exists(m_parquet_filename):
        logging.error(f"M set file not found: {m_parquet_filename}")
        return
    
    # Find all K grid files
    k_grid_files_unsorted = glob.glob(os.path.join(k_directory, "k_*.parquet"))
    if not k_grid_files_unsorted:
        logging.error(f"No k_*.parquet files found in directory: {k_directory}")
        return
    
    k_grid_files = sorted(k_grid_files_unsorted, key=extract_k_grid_number)
    num_k_grids = len(k_grid_files)
    logging.info(f"Found and numerically sorted {num_k_grids} K grid files to process.")
    
    # Load first K grid to determine optimal processor count
    first_k_grid = pd.read_parquet(k_grid_files[0])
    first_k_grid_size = len(first_k_grid)
    optimal_processors = calculate_processors_for_k_grid_size(first_k_grid_size)
    
    # Override processes_count with optimal value
    original_processes = processes_count
    processes_count = optimal_processors
    
    logging.info(f"First K grid has {first_k_grid_size:,} rows")
    logging.info(f"Calculated optimal processor count: {processes_count} (was {original_processes})")
    logging.info(f"Target concurrent pixels: {processes_count * first_k_grid_size:,}")
    
    # Free the first K grid from memory
    del first_k_grid
    gc.collect()
    
    # Generate all random seeds upfront
    rng = np.random.default_rng(seed)
    iteration_seeds = rng.integers(0, 2000000, num_k_grids)
    shuffle_seeds = rng.integers(0, 2000000, num_k_grids)
    
    # Load full M set to compute sample size
    logging.info(f"Loading full M set from {m_parquet_filename} to compute sample size")
    m_set = pd.read_parquet(m_parquet_filename)
    total_m_rows = len(m_set)
    sample_size = max(5_000_000, int(0.1 * total_m_rows))
    logging.info(f"M set has {total_m_rows:,} rows. Using sample size: {sample_size:,}")    
    
    # create M samples first using parallel processing
    sample_processes = processes_count
    logging.info("Creating M set samples...")
    create_m_samples(
        m_parquet_filename=m_parquet_filename,  # Pass filename
        output_folder=output_folder,
        num_samples=num_k_grids,
        sample_size=sample_size,
        seeds=iteration_seeds,
        sample_processes=min(8, processes_count)  # Limit to 8 for sampling
    )
    
    # Create mapping of grid files to their corresponding M samples
    m_samples_dir = os.path.join(output_folder, "m_samples")
    map_arguments = []
    for i, (k_grid_file, iter_seed, shuf_seed) in enumerate(zip(k_grid_files, iteration_seeds, shuffle_seeds)):
        m_sample_file = os.path.join(m_samples_dir, f"m_sample_{i+1:03d}.parquet")
        map_arguments.append((
            m_sample_file,
            k_grid_file,
            iter_seed,
            shuf_seed,
            start_year,
            evaluation_year,
            output_folder,
            max_potential_matches
        ))
    
    processed_k_grid_ids = []
    total_processed = 0
    
    log_memory_usage("before starting multiprocessing pool", detailed=True)
    
    with Pool(processes=processes_count, initializer=_worker_init) as pool:
        for i in range(0, num_k_grids, batch_size):
            batch_args = map_arguments[i:min(i + batch_size, num_k_grids)]
            if not batch_args:
                break
            
            logging.info(f"Processing batch {i//batch_size + 1}/{(num_k_grids + batch_size - 1)//batch_size}")
            
            log_memory_usage(f"before batch {i//batch_size + 1}", detailed=True)
            log_system_memory()
            
            # Use the module-level function directly
            batch_results = pool.map(iteration_func_wrapper, batch_args)
            
            log_memory_usage(f"after batch {i//batch_size + 1}", detailed=True)
            
            for k_grid_id in batch_results:
                processed_k_grid_ids.append(k_grid_id)
            
            total_processed = len(processed_k_grid_ids)
            logging.info(f"Processed {total_processed}/{num_k_grids} K grids.")
    
    log_memory_usage("after multiprocessing completed", detailed=True)
    logging.info(f"Finished processing. Total K grids processed: {total_processed}")
    
    # Calculate and save matching balance metrics
    logging.info("Calculating Standardised Mean Differences for matching balance...")
    log_memory_usage("before SMD analysis")
    
    # Get luc_columns from first matched file
    first_match = glob.glob(os.path.join(output_folder, "[0-9]*.parquet"))
    if first_match:
        luc_columns = [x for x in pd.read_parquet(first_match[0]).columns if x.startswith('luc')]
    else:
        luc_columns = []
    
    distance_columns = DISTANCE_COLUMNS
    smd_df = analyse_matching_balance(output_folder, evaluation_year, luc_columns, distance_columns)
    
    if not smd_df.empty:
        smd_output_file = os.path.join(output_folder, "matching_balance_smd.csv")
        smd_df.to_csv(smd_output_file, index=False)
        logging.info(f"SMD analysis saved to: {smd_output_file}")
        
        # Print SMD results table
        print("\n" + "="*80)
        print("MATCHING BALANCE ANALYSIS - Standardised Mean Differences (SMD)")
        print("="*80)
        print(f"{'Variable':<15} {'N_K':<8} {'N_S':<8} {'Mean_K':<10} {'Mean_S':<10} {'SMD':<8} {'Interpretation'}")
        print("-"*80)
        for _, row in smd_df.iterrows():
            print(f"{row['variable']:<15} {row['n_k']:<8} {row['n_s']:<8} "
                  f"{row['mean_k']:<10.3f} {row['mean_s']:<10.3f} "
                  f"{row['smd']:<8.3f} {row['smd_interpretation']}")
        print("-"*80)
        print(f"SMD Interpretation: |SMD| < 0.1 = Negligible, 0.1-0.2 = Small, 0.2-0.5 = Medium, 0.5-0.8 = Large, >0.8 = Very Large")
        print(f"Good matching typically has |SMD| < 0.1 for all variables")
        print("="*80)
    else:
        logging.warning("Could not calculate SMD analysis - no valid matched data found")
    
    log_memory_usage("end of main function", detailed=True)
    
    # final summary to memory log
    if MEMORY_LOG_FILE:
        with open(MEMORY_LOG_FILE, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"FINAL SUMMARY\n")
            f.write(f"Completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total K grids processed: {total_processed}\n")
            f.write(f"Processes used: {processes_count}\n")
            f.write(f"{'='*80}\n")

def main():
    """Main entry point for the matching algorithm."""
    parser = argparse.ArgumentParser(description="Match K→M without sampling or multiprocessing")
    parser.add_argument("--k_directory", required=True)
    parser.add_argument("--m_parquet_filename", required=True)
    parser.add_argument("--start_year", type=int, required=True)
    parser.add_argument("--evaluation_year", type=int, required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of K grids to process per batch")
    parser.add_argument("--processes_count", type=int, default=16,
                        help="Number of processes to use for parallel processing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--max_potential_matches", type=int, default=1000,
                        help="Maximum number of potential M matches to collect per K pixel (cap)")
    args = parser.parse_args()

    # logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Run the main matching process
    find_pairs(
        k_directory=args.k_directory,
        m_parquet_filename=args.m_parquet_filename,
        start_year=args.start_year,
        evaluation_year=args.evaluation_year,
        seed=args.seed,
        output_folder=args.output_folder,
        batch_size=args.batch_size,
        processes_count=args.processes_count,
        max_potential_matches=args.max_potential_matches
    )

 
if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    main()