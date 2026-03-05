#!/usr/bin/env python3
"""
Counterfactual proportion matching.

For each K pixel that is undisturbed (luc == 1) in the start year:
  1. Find candidate M pixels using hard + threshold filtering (same as find_pairs).
  2. For each year from start_year to evaluation_year, compute the proportion of
     those candidates that remain undisturbed (luc == 1).

Output per K grid: one parquet with columns
    k_lat, k_lng, k_luc_{start_year..eval_year},
    s_prop_{start_year..eval_year},   # proportion of candidates with luc==1
    n_candidates
"""

import argparse
import glob
import os
import logging
import re
import gc
import datetime
import time
import psutil
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count, set_start_method
from numba import jit
from methods.common.luc import luc_matching_columns
import logging.handlers
import multiprocessing as mp
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HARD_COLUMN_COUNT = 5

DISTANCE_COLUMNS = [
    "elevation", "slope", "access",
    "fcc0_u", "fcc0_d",
    "fcc5_u", "fcc5_d",
    "fcc10_u", "fcc10_d",
]

# Spatial balancing settings to reduce single-cluster dominance
SPATIAL_TILE_DEGREES = 0.1
MAX_CLUSTER_SHARE = 0.3

# ---------------------------------------------------------------------------
# Utility helpers (logging, memory, etc.)
# ---------------------------------------------------------------------------

MEMORY_LOG_FILE = None


def _worker_init():
    """Initializer for multiprocessing workers."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [PID %(process)d] %(message)s",
        stream=sys.stdout,
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    elif not sys.stdout.line_buffering:
        sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)


def setup_memory_logging(output_folder):
    global MEMORY_LOG_FILE
    MEMORY_LOG_FILE = os.path.join(output_folder, "memory_usage_log.txt")
    with open(MEMORY_LOG_FILE, "w") as f:
        f.write(f"Memory Usage Log - Started at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 50 + "\n\n")


def _log(msg, to_file=True):
    logging.info(msg)
    if to_file and MEMORY_LOG_FILE:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        with open(MEMORY_LOG_FILE, "a") as f:
            f.write(f"[{ts}] PID {os.getpid()}: {msg}\n")


def log_memory(label=""):
    try:
        rss = psutil.Process().memory_info().rss / 1024 / 1024
        _log(f"Memory {label}: RSS {rss:.1f} MB")
    except Exception:
        pass


def calculate_processors_for_k_grid_size(k_grid_size: int) -> int:
    # With grouped matching the bottleneck is I/O (reading M samples),
    # not compute. Cap at 8 to avoid saturating disk bandwidth.
    max_concurrent = 320_000
    return max(1, min(8, max_concurrent // k_grid_size))


# ---------------------------------------------------------------------------
# Numba kernel – identical to find_pairs_mode
# ---------------------------------------------------------------------------

@jit(nopython=True, fastmath=True, error_model="numpy", cache=True)
def _threshold_match_group(
    k_row_dist,
    group_dist,
    group_global_indices,
    start_offset,
    max_potential_matches,
):
    """
    For a single K pixel, scan only the M pixels in its hard-match group
    and return those that also pass the continuous threshold check.

    Returns (match_global_indices, n_matches).
    """
    g_size = group_dist.shape[0]
    out = np.empty(max_potential_matches, dtype=np.int64)
    matches = 0

    for idx in range(g_size):
        g_idx = (idx + start_offset) % g_size
        m_row = group_dist[g_idx, :]
        ok = True
        for j in range(m_row.shape[0]):
            if abs(m_row[j] - k_row_dist[j]) > 1.0:
                ok = False
                break
        if ok:
            out[matches] = group_global_indices[g_idx]
            matches += 1
            if matches == max_potential_matches:
                break

    return out[:matches], matches


def make_s_set_mask_grouped(
    m_dist_thresholded,
    k_subset_dist_thresholded,
    m_hard_df,
    k_hard_df,
    rng,
    max_potential_matches,
    grid_id="",
):
    """
    Group-based candidate matching.  O(K × M_group) instead of O(K × M).

    1. Group M rows by their hard-match key (tuple of categorical values).
    2. For each K pixel, look up its group and run a threshold-only Numba scan.
    """
    t_start = time.perf_counter()
    k_size = k_subset_dist_thresholded.shape[0]
    k_to_m_indices = np.full((k_size, max_potential_matches), -1, dtype=np.int64)
    k_match_counts = np.zeros(k_size, dtype=np.int64)
    k_miss = np.zeros(k_size, dtype=np.bool_)

    # --- Build group index from M hard columns (vectorised) ---
    n_hard = m_hard_df.shape[1]
    P = np.int64(100_000)
    m_keys = np.zeros(m_hard_df.shape[0], dtype=np.int64)
    for c in range(n_hard):
        m_keys += m_hard_df[:, c].astype(np.int64) * (P ** c)

    key_series = pd.Series(m_keys)
    grouped = key_series.groupby(key_series).indices

    t_group = time.perf_counter()
    group_sizes = [len(v) for v in grouped.values()]
    print(
        f"[prop] K grid {grid_id}: grouped M into {len(grouped)} hard-key groups "
        f"(median size {int(np.median(group_sizes))}, max {max(group_sizes)}) "
        f"in {t_group - t_start:.2f}s",
        flush=True,
    )

    # Pre-build contiguous arrays per group for Numba
    group_cache = {}
    for key_val, idx_arr in grouped.items():
        idx_arr = idx_arr.astype(np.int64)
        group_cache[key_val] = (
            np.ascontiguousarray(m_dist_thresholded[idx_arr, :]),
            idx_arr,
        )

    t_cache = time.perf_counter()
    print(f"[prop] K grid {grid_id}: built group cache in {t_cache - t_group:.2f}s", flush=True)

    # Pre-compute K keys the same way
    k_keys = np.zeros(k_size, dtype=np.int64)
    for c in range(n_hard):
        k_keys += k_hard_df[:, c].astype(np.int64) * (P ** c)

    # --- Match each K pixel against its group ---
    n_no_group = 0
    for k in range(k_size):
        cached = group_cache.get(k_keys[k])
        if cached is None:
            k_miss[k] = True
            n_no_group += 1
            continue

        group_dist, group_global_idx = cached
        g_size = group_dist.shape[0]
        start_offset = int(rng.integers(0, g_size)) if g_size > 0 else 0

        matched_idx, n = _threshold_match_group(
            k_subset_dist_thresholded[k, :],
            group_dist,
            group_global_idx,
            start_offset,
            max_potential_matches,
        )

        k_match_counts[k] = n
        if n == 0:
            k_miss[k] = True
        else:
            k_to_m_indices[k, :n] = matched_idx

        # Progress every 200 K pixels
        if (k + 1) % 200 == 0:
            print(
                f"[prop] K grid {grid_id}: matched {k + 1}/{k_size} K pixels "
                f"({time.perf_counter() - t_cache:.1f}s elapsed)",
                flush=True,
            )

    t_match = time.perf_counter()
    print(
        f"[prop] K grid {grid_id}: matching loop done in {t_match - t_cache:.2f}s "
        f"({n_no_group} K pixels had no group in M)",
        flush=True,
    )

    return k_miss, k_to_m_indices, k_match_counts


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

def find_proportion_iteration(
    m_sample_filename: str,
    start_year: int,
    evaluation_year: int,
    output_folder: str,
    k_grid_filepath_and_seed: tuple,
    shuffle_seed: int,
    max_potential_matches: int,
) -> str:
    k_grid_filepath, seed = k_grid_filepath_and_seed
    k_grid_filename = os.path.basename(k_grid_filepath)

    match = re.search(r"\d+", k_grid_filename)
    k_grid_id = match.group(0) if match else k_grid_filename.replace(".parquet", "")

    print(f"[prop] Starting K grid {k_grid_id}", flush=True)
    log_memory(f"start {k_grid_id}")

    rng = np.random.default_rng(seed)

    # ---- Load K grid & filter to undisturbed in start year ----
    k_all = pd.read_parquet(k_grid_filepath).reset_index(drop=True)
    start_luc_col = f"luc_{start_year}"
    k_subset = k_all[k_all[start_luc_col] == 1].reset_index(drop=True)
    n_dropped = len(k_all) - len(k_subset)
    del k_all
    gc.collect()

    if len(k_subset) == 0:
        print(f"[prop] K grid {k_grid_id}: no undisturbed pixels – skipping", flush=True)
        # Write empty parquet so downstream doesn't error
        pd.DataFrame().to_parquet(os.path.join(output_folder, f"{k_grid_id}.parquet"))
        pd.DataFrame().to_parquet(os.path.join(output_folder, f"{k_grid_id}_matchless.parquet"))
        return k_grid_id

    print(f"[prop] K grid {k_grid_id}: {len(k_subset)} undisturbed pixels ({n_dropped} dropped)", flush=True)

    # ---- Load & shuffle M sample ----
    print(f"[prop] K grid {k_grid_id}: reading M sample from {os.path.basename(m_sample_filename)}...", flush=True)
    t0 = time.perf_counter()
    m_set = pd.read_parquet(m_sample_filename)
    shuffle_rng = np.random.default_rng(shuffle_seed)
    m_set = m_set.sample(frac=1, random_state=shuffle_rng).reset_index(drop=True)
    print(f"[prop] K grid {k_grid_id}: loaded & shuffled M sample ({len(m_set):,} rows) in {time.perf_counter()-t0:.1f}s", flush=True)

    # ---- Prepare matching arrays ----
    thresholds = np.array([200.0, 2.5, 30.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])

    m_dist = np.ascontiguousarray((m_set[DISTANCE_COLUMNS] / thresholds).values, dtype=np.float32)
    k_dist = np.ascontiguousarray((k_subset[DISTANCE_COLUMNS] / thresholds).values, dtype=np.float32)

    luc0, luc5, luc10 = luc_matching_columns(start_year)
    hard_cols = ["country", "ecoregion", luc10, luc5, luc0]

    m_hard = np.ascontiguousarray(m_set[hard_cols].values, dtype=np.int32)
    k_hard = np.ascontiguousarray(k_subset[hard_cols].values, dtype=np.int32)

    # ---- Run grouped matching (O(K × M_group) instead of O(K × M)) ----
    t1 = time.perf_counter()
    k_miss, k_to_m_indices, k_match_counts = make_s_set_mask_grouped(
        m_dist, k_dist, m_hard, k_hard, rng, max_potential_matches, grid_id=k_grid_id
    )
    print(f"[prop] K grid {k_grid_id}: total matching phase in {time.perf_counter()-t1:.1f}s", flush=True)

    n_matched = int(np.sum(~k_miss))
    n_matchless = int(np.sum(k_miss))
    print(f"[prop] K grid {k_grid_id}: {n_matched} matched, {n_matchless} matchless", flush=True)

    # ---- Determine year range & pre-extract M LUC arrays ----
    luc_years = list(range(start_year, evaluation_year + 1))
    luc_col_names = [f"luc_{y}" for y in luc_years]
    # Only keep columns that actually exist in the M set
    existing_luc = [(y, c) for y, c in zip(luc_years, luc_col_names) if c in m_set.columns]

    m_luc_arrays = {}
    for y, c in existing_luc:
        m_luc_arrays[y] = m_set[c].values

    # Also grab k_luc values
    k_luc_arrays = {}
    for y, c in existing_luc:
        if c in k_subset.columns:
            k_luc_arrays[y] = k_subset[c].values

    k_lat = k_subset["lat"].values
    k_lng = k_subset["lng"].values
    m_lat = m_set["lat"].values
    m_lng = m_set["lng"].values

    # ---- Build results ----
    t2 = time.perf_counter()
    n_k = len(k_subset)
    # Pre-allocate result arrays
    res_lat = np.empty(n_k, dtype=np.float64)
    res_lng = np.empty(n_k, dtype=np.float64)
    res_n_cand = np.empty(n_k, dtype=np.int32)
    res_k_luc = {y: np.empty(n_k, dtype=np.int16) for y in k_luc_arrays}
    res_s_prop = {y: np.full(n_k, np.nan, dtype=np.float32) for y, _ in existing_luc}
    res_matched = np.ones(n_k, dtype=bool)

    for k_idx in range(n_k):
        res_lat[k_idx] = k_lat[k_idx]
        res_lng[k_idx] = k_lng[k_idx]

        for y, arr in k_luc_arrays.items():
            res_k_luc[y][k_idx] = arr[k_idx]

        n_cand = int(k_match_counts[k_idx])
        res_n_cand[k_idx] = n_cand

        if n_cand == 0:
            res_matched[k_idx] = False
            continue

        cand_idx = k_to_m_indices[k_idx, :n_cand]

        # Similarity weights from normalized covariate distance (closer candidates weighted higher)
        candidate_dist_vectors = m_dist[cand_idx, :] - k_dist[k_idx, :]
        candidate_distances = np.sqrt(np.sum(candidate_dist_vectors * candidate_dist_vectors, axis=1))

        # Spatial cluster balancing: cap contribution from any single spatial tile
        cand_lat = m_lat[cand_idx]
        cand_lng = m_lng[cand_idx]
        tile_lat = np.floor(cand_lat / SPATIAL_TILE_DEGREES).astype(np.int64)
        tile_lng = np.floor(cand_lng / SPATIAL_TILE_DEGREES).astype(np.int64)
        tile_ids = tile_lat * 10_000_000 + tile_lng

        max_per_cluster = max(1, int(np.floor(n_cand * MAX_CLUSTER_SHARE)))
        order = np.argsort(candidate_distances)

        selected_positions = []
        cluster_counts = {}
        for pos in order:
            cluster_id = int(tile_ids[pos])
            count = cluster_counts.get(cluster_id, 0)
            if count < max_per_cluster:
                selected_positions.append(int(pos))
                cluster_counts[cluster_id] = count + 1

        # Fallback: keep at least one candidate if cap is overly restrictive
        if len(selected_positions) == 0:
            selected_positions = [int(order[0])]

        selected_positions = np.array(selected_positions, dtype=np.int64)
        cand_idx = cand_idx[selected_positions]
        n_cand_selected = len(cand_idx)
        res_n_cand[k_idx] = n_cand_selected

        similarity_weights = 1.0 / (candidate_distances[selected_positions] + 1e-6)

        # Inverse cluster-density weighting among selected candidates
        selected_tile_ids = tile_ids[selected_positions]
        unique_tiles, tile_freq = np.unique(selected_tile_ids, return_counts=True)
        tile_freq_map = {int(tid): int(freq) for tid, freq in zip(unique_tiles, tile_freq)}
        density_weights = np.array([1.0 / tile_freq_map[int(tid)] for tid in selected_tile_ids], dtype=np.float64)

        combined_weights = similarity_weights * density_weights
        weights_sum = combined_weights.sum()
        if weights_sum > 0:
            combined_weights = combined_weights / weights_sum
        else:
            combined_weights = np.full(n_cand_selected, 1.0 / n_cand_selected, dtype=np.float64)

        for y, m_arr in m_luc_arrays.items():
            cand_vals = m_arr[cand_idx]
            is_undisturbed = (cand_vals == 1).astype(np.float64)
            res_s_prop[y][k_idx] = float(np.sum(combined_weights * is_undisturbed))

    # ---- Assemble DataFrame ----
    data = {"k_lat": res_lat, "k_lng": res_lng}
    for y in sorted(res_k_luc):
        data[f"k_luc_{y}"] = res_k_luc[y]
    for y in sorted(res_s_prop):
        data[f"s_prop_{y}"] = res_s_prop[y]
    data["n_candidates"] = res_n_cand

    results_df = pd.DataFrame(data)
    matched_df = results_df[res_matched].reset_index(drop=True)
    matchless_df = results_df[~res_matched].reset_index(drop=True)

    matched_df.to_parquet(os.path.join(output_folder, f"{k_grid_id}.parquet"))
    matchless_df.to_parquet(os.path.join(output_folder, f"{k_grid_id}_matchless.parquet"))

    print(
        f"[prop] K grid {k_grid_id}: saved {len(matched_df)} matched, "
        f"{len(matchless_df)} matchless "
        f"(proportions computed in {time.perf_counter()-t2:.2f}s)",
        flush=True,
    )

    del m_set, k_subset, m_dist, k_dist, m_hard, k_hard, m_lat, m_lng
    del k_to_m_indices, k_match_counts, m_luc_arrays, k_luc_arrays
    gc.collect()

    return k_grid_id


# ---------------------------------------------------------------------------
# Orchestration (identical plumbing to find_pairs_mode)
# ---------------------------------------------------------------------------

def extract_k_grid_number(filepath: str) -> int:
    basename = os.path.basename(filepath)
    m = re.search(r"k_(\d+)\.parquet$", basename)
    return int(m.group(1)) if m else float("inf")


def create_single_m_sample(args):
    m_parquet_filename, sample_index, sample_size, seed, m_samples_dir = args
    logging.info(f"Creating sample {sample_index + 1} with seed {seed}")
    m_set = pd.read_parquet(m_parquet_filename)
    rng = np.random.default_rng(seed)
    replace = len(m_set) < sample_size
    indices = rng.choice(len(m_set), size=sample_size, replace=replace)
    sample = m_set.iloc[indices].reset_index(drop=True)
    sample = sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)
    sample.to_parquet(os.path.join(m_samples_dir, f"m_sample_{sample_index + 1:03d}.parquet"))
    del sample, m_set
    gc.collect()
    return f"Completed sample {sample_index + 1}"


def create_m_samples(m_parquet_filename, output_folder, num_samples, sample_size, seeds, sample_processes=None):
    if sample_processes is None:
        sample_processes = min(cpu_count(), 8)
    m_samples_dir = os.path.join(output_folder, "m_samples")
    os.makedirs(m_samples_dir, exist_ok=True)

    missing = []
    for i in range(num_samples):
        if not os.path.exists(os.path.join(m_samples_dir, f"m_sample_{i + 1:03d}.parquet")):
            missing.append(i)

    if not missing:
        logging.info("All M samples already exist. Skipping.")
        return

    logging.info(f"Creating {len(missing)} M samples of {sample_size:,} rows...")
    args_list = [(m_parquet_filename, i, sample_size, seeds[i], m_samples_dir) for i in missing]
    with Pool(processes=sample_processes, initializer=_worker_init) as pool:
        for r in pool.map(create_single_m_sample, args_list):
            logging.info(r)


def _iteration_wrapper(args):
    (m_sample, k_grid, iter_seed, shuf_seed, start_year, eval_year, out, max_pm) = args
    return find_proportion_iteration(m_sample, start_year, eval_year, out, (k_grid, iter_seed), shuf_seed, max_pm)


def find_pairs(
    k_directory, m_parquet_filename, m_sample_folder,
    start_year, evaluation_year, seed,
    output_folder, batch_size, processes_count, max_potential_matches=100,
):
    logging.info("Starting find_pairs_prop")
    os.makedirs(output_folder, exist_ok=True)
    setup_memory_logging(output_folder)
    log_memory("main start")

    if not os.path.exists(m_parquet_filename):
        logging.error(f"M set not found: {m_parquet_filename}")
        return

    k_files = sorted(glob.glob(os.path.join(k_directory, "k_*.parquet")), key=extract_k_grid_number)
    if not k_files:
        logging.error(f"No k_*.parquet in {k_directory}")
        return

    num_k = len(k_files)
    logging.info(f"Found {num_k} K grid files")

    first_size = len(pd.read_parquet(k_files[0]))
    processes_count = calculate_processors_for_k_grid_size(first_size)

    
    rng = np.random.default_rng(seed)
    iter_seeds = rng.integers(0, 2_000_000, num_k)
    shuf_seeds = rng.integers(0, 2_000_000, num_k)

    # M samples
    if m_sample_folder and os.path.exists(m_sample_folder):
        m_samples_dir = m_sample_folder
        logging.info(f"Using existing M samples from {m_samples_dir}")
    else:
        m_set = pd.read_parquet(m_parquet_filename)
        sample_size = max(5_000_000, int(0.1 * len(m_set)))
        logging.info(f"M set {len(m_set):,} rows, sample size {sample_size:,}")
        del m_set; gc.collect()
        create_m_samples(m_parquet_filename, output_folder, num_k, sample_size, iter_seeds, min(8, processes_count))
        m_samples_dir = os.path.join(output_folder, "m_samples")

    map_args = []
    for i, (kf, iseed, sseed) in enumerate(zip(k_files, iter_seeds, shuf_seeds)):
        msf = os.path.join(m_samples_dir, f"m_sample_{i + 1:03d}.parquet")
        map_args.append((msf, kf, iseed, sseed, start_year, evaluation_year, output_folder, max_potential_matches))

    processed = []
    with Pool(processes=processes_count, initializer=_worker_init) as pool:
        for i in range(0, num_k, batch_size):
            batch = map_args[i : i + batch_size]
            logging.info(f"Batch {i // batch_size + 1}/{(num_k + batch_size - 1) // batch_size}")
            log_memory(f"before batch {i // batch_size + 1}")

            # Submit tasks with a small stagger between starts to avoid many
            # workers simultaneously reading large M sample parquet files.
            futures = []
            for args in batch:
                futures.append(pool.apply_async(_iteration_wrapper, (args,)))
                time.sleep(0.25)

            # Collect results for this batch
            results = [f.get() for f in futures]
            processed.extend(results)
            logging.info(f"Processed {len(processed)}/{num_k}")

    logging.info(f"Done. {len(processed)} K grids processed.")

    if MEMORY_LOG_FILE:
        with open(MEMORY_LOG_FILE, "a") as f:
            f.write(f"\nCompleted: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"Total K grids: {len(processed)}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Counterfactual proportion matching: for undisturbed K pixels, "
                    "compute the proportion of candidate M pixels remaining undisturbed each year."
    )
    parser.add_argument("--k_directory", required=True)
    parser.add_argument("--m_parquet_filename", required=True)
    parser.add_argument("--m_sample_folder", default=None)
    parser.add_argument("--start_year", type=int, required=True)
    parser.add_argument("--evaluation_year", type=int, required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--processes_count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_potential_matches", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    find_pairs(
        k_directory=args.k_directory,
        m_parquet_filename=args.m_parquet_filename,
        m_sample_folder=args.m_sample_folder,
        start_year=args.start_year,
        evaluation_year=args.evaluation_year,
        seed=args.seed,
        output_folder=args.output_folder,
        batch_size=args.batch_size,
        processes_count=args.processes_count,
        max_potential_matches=args.max_potential_matches,
    )


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()
