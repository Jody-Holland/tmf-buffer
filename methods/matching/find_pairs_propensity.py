#!/usr/bin/env python3
"""
Counterfactual propensity-weighted proportion matching.

For each K pixel that is undisturbed (luc == 1) in the start year:
    1. Find candidate M pixels using hard + threshold filtering (same as find_pairs).
    2. For each year from start_year to evaluation_year, compute a propensity-weighted
       proportion of candidates that remain undisturbed (luc == 1).

Output per K grid: one parquet with columns
    k_lat, k_lng, k_luc_{start_year..eval_year},
    s_prop_{start_year..eval_year},   # weighted proportion of candidates with luc==1
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
from sklearn.linear_model import LogisticRegression
from methods.common.luc import luc_matching_columns
import logging.handlers
import multiprocessing as mp
from typing import Optional
try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

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
    # not compute. Cap at 32 to avoid saturating disk bandwidth while
    # allowing higher parallelism on machines with sufficient resources.
    max_concurrent_pixels = 500_000
    num_processors = min(32, (max_concurrent_pixels // k_grid_size))
    print(f"Calculated {num_processors} processors for K grid size {k_grid_size:,}")
    return num_processors


def _parquet_columns(path: str):
    """Return column names for a parquet file without loading full data when possible."""
    if pq is not None:
        try:
            pf = pq.ParquetFile(path)
            return list(pf.schema.names)
        except Exception:
            pass
    # Fallback — read with pandas (costly)
    try:
        return list(pd.read_parquet(path).columns)
    except Exception:
        return []


def _parquet_num_rows(path: str) -> Optional[int]:
    """Return parquet row count from metadata when available."""
    if pq is not None:
        try:
            pf = pq.ParquetFile(path)
            return int(pf.metadata.num_rows)
        except Exception:
            pass
    return None


def _unique_preserve_order(items):
    """Return items with duplicates removed, preserving first-seen order."""
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _drop_duplicate_columns(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Drop duplicate column labels (keep first) to make boolean indexing robust."""
    if df.columns.has_duplicates:
        dup_cols = pd.Index(df.columns)[pd.Index(df.columns).duplicated()].unique().tolist()
        preview = ", ".join(map(str, dup_cols[:5]))
        more = "..." if len(dup_cols) > 5 else ""
        logging.warning(
            f"{label}: dropping {len(dup_cols)} duplicate column labels: {preview}{more}"
        )
        return df.loc[:, ~df.columns.duplicated(keep="first")].copy()
    return df


# ---------------------------------------------------------------------------
# Numba kernel – identical to find_pairs_mode
# ---------------------------------------------------------------------------

@jit(nopython=True, fastmath=True, error_model="numpy", cache=True)
def _threshold_match_group(
    k_row_dist,
    group_dist,
    group_global_indices,
    start_offset,
):
    """
    For a single K pixel, scan only the M pixels in its hard-match group
    and return those that also pass the continuous threshold check.

    Returns (match_global_indices, n_matches).
    """
    g_size = group_dist.shape[0]
    out = np.empty(g_size, dtype=np.int64)
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

    return out[:matches], matches


def compute_propensity_scores(
    m_set: pd.DataFrame,
    k_subset: pd.DataFrame,
    hard_cols: list,
):
    """Compute propensity-like scores for M and K rows using logistic regression.

    The model predicts membership in K (1) vs M (0) using hard + continuous covariates.
    """
    # Use only features that exist in both M and K to avoid KeyErrors
    features = [f for f in (hard_cols + DISTANCE_COLUMNS) if f in m_set.columns and f in k_subset.columns]
    if len(features) == 0:
        raise KeyError("No overlapping features found between M and K for propensity computation")
    x_m = m_set[features].to_numpy(dtype=np.float32)
    x_k = k_subset[features].to_numpy(dtype=np.float32)

    x = np.vstack([x_m, x_k])
    y = np.concatenate([
        np.zeros(x_m.shape[0], dtype=np.int8),
        np.ones(x_k.shape[0], dtype=np.int8),
    ])

    # Standardise for stable logistic fitting
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    sigma[sigma == 0] = 1.0
    x_scaled = (x - mu) / sigma

    model = LogisticRegression(max_iter=200, solver="lbfgs", class_weight="balanced")
    model.fit(x_scaled, y)
    scores = model.predict_proba(x_scaled)[:, 1].astype(np.float32)

    m_scores = scores[:x_m.shape[0]]
    k_scores = scores[x_m.shape[0]:]
    return m_scores, k_scores


def make_s_set_mask_grouped(
    m_dist_thresholded,
    k_subset_dist_thresholded,
    m_hard_df,
    k_hard_df,
    m_propensity,
    k_propensity,
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
        )

        if n == 0:
            k_miss[k] = True
        else:
            # Propensity stratification: keep candidates closest in propensity score to K pixel
            if n > max_potential_matches:
                k_score = k_propensity[k]
                score_diffs = np.abs(m_propensity[matched_idx] - k_score)
                keep_idx = np.argsort(score_diffs)[:max_potential_matches]
                selected = matched_idx[keep_idx]
            else:
                selected = matched_idx

            n_sel = selected.shape[0]
            k_to_m_indices[k, :n_sel] = selected
            k_match_counts[k] = n_sel

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

    # Determine luc years and matching columns early so we can read only needed columns
    luc_years = list(range(start_year, evaluation_year + 1))
    luc_col_names = [f"luc_{y}" for y in luc_years]

    luc0, luc5, luc10 = luc_matching_columns(start_year)
    k_hard_cols = ["country", "ecoregion", luc10, luc5, luc0]

    # ---- Load K grid & filter to undisturbed in start year (read only needed columns) ----
    k_available = _parquet_columns(k_grid_filepath)
    needed_k_cols = _unique_preserve_order(
        [c for c in (DISTANCE_COLUMNS + ["lat", "lng"] + k_hard_cols + luc_col_names) if c in k_available]
    )
    if needed_k_cols:
        k_all = pd.read_parquet(k_grid_filepath, columns=needed_k_cols).reset_index(drop=True)
    else:
        k_all = pd.read_parquet(k_grid_filepath).reset_index(drop=True)
    k_all = _drop_duplicate_columns(k_all, f"K grid {k_grid_id}")

    start_luc_col = f"luc_{start_year}"
    if start_luc_col not in k_all.columns:
        raise KeyError(f"K grid {k_grid_id}: required column '{start_luc_col}' not found")
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
    # Read only needed columns from M sample to reduce memory
    m_available = _parquet_columns(m_sample_filename)
    needed_m_cols = _unique_preserve_order(
        [c for c in (DISTANCE_COLUMNS + k_hard_cols + luc_col_names) if c in m_available]
    )
    if needed_m_cols:
        m_set = pd.read_parquet(m_sample_filename, columns=needed_m_cols)
    else:
        m_set = pd.read_parquet(m_sample_filename)
    m_set = _drop_duplicate_columns(m_set, f"K grid {k_grid_id} M sample")

    # Filter to rows undisturbed in start year if present
    if start_luc_col in m_set.columns:
        m_set = m_set[m_set[start_luc_col] == 1].reset_index(drop=True)

    # Downcast numeric columns to reduce memory
    for c in DISTANCE_COLUMNS:
        if c in m_set.columns:
            try:
                m_set[c] = m_set[c].astype(np.float32)
            except Exception:
                pass
    for c in luc_col_names:
        if c in m_set.columns:
            try:
                m_set[c] = m_set[c].astype(np.int8)
            except Exception:
                pass
    for c in ["country", "ecoregion"]:
        if c in m_set.columns:
            try:
                m_set[c] = m_set[c].astype(np.int32)
            except Exception:
                pass

    shuffle_rng = np.random.default_rng(shuffle_seed)
    m_set = m_set.sample(frac=1, random_state=shuffle_rng).reset_index(drop=True)
    print(f"[prop] K grid {k_grid_id}: loaded & shuffled M sample ({len(m_set):,} rows) in {time.perf_counter()-t0:.1f}s", flush=True)

    # ---- Prepare matching arrays ----
    thresholds = np.array([100.0, 10, 30.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    # Build distance arrays (ensure columns exist)
    present_dist = [c for c in DISTANCE_COLUMNS if c in m_set.columns and c in k_subset.columns]
    m_dist = np.ascontiguousarray((m_set[present_dist] / thresholds[: len(present_dist)]).values, dtype=np.float32)
    k_dist = np.ascontiguousarray((k_subset[present_dist] / thresholds[: len(present_dist)]).values, dtype=np.float32)

    hard_cols = k_hard_cols

    # ---- Compute propensity scores for stratified candidate selection ----
    m_propensity, k_propensity = compute_propensity_scores(m_set, k_subset, hard_cols)

    m_hard = np.ascontiguousarray(m_set[hard_cols].values, dtype=np.int32)
    k_hard = np.ascontiguousarray(k_subset[hard_cols].values, dtype=np.int32)

    # ---- Run grouped matching (O(K × M_group) instead of O(K × M)) ----
    t1 = time.perf_counter()
    k_miss, k_to_m_indices, k_match_counts = make_s_set_mask_grouped(
        m_dist,
        k_dist,
        m_hard,
        k_hard,
        m_propensity,
        k_propensity,
        rng,
        max_potential_matches,
        grid_id=k_grid_id,
    )
    print(f"[prop] K grid {k_grid_id}: total matching phase in {time.perf_counter()-t1:.1f}s", flush=True)

    n_matched = int(np.sum(~k_miss))
    n_matchless = int(np.sum(k_miss))
    print(f"[prop] K grid {k_grid_id}: {n_matched} matched, {n_matchless} matchless", flush=True)

    # ---- Pre-extract M LUC arrays and then free DataFrame to save memory ----
    existing_luc = [(y, f"luc_{y}") for y in luc_years if f"luc_{y}" in m_set.columns]
    m_luc_arrays = {y: m_set[c].values for y, c in existing_luc}
    # m_set no longer needed beyond this point
    del m_set
    gc.collect()

    # Also grab k_luc values
    k_luc_arrays = {}
    for y, c in existing_luc:
        if c in k_subset.columns:
            k_luc_arrays[y] = k_subset[c].values

    k_lat = k_subset["lat"].values
    k_lng = k_subset["lng"].values

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

        # Propensity-based candidate weights (closer propensity => higher weight)
        cand_prop = m_propensity[cand_idx]
        k_prop = k_propensity[k_idx]
        prop_diff = np.abs(cand_prop - k_prop)
        weights = 1.0 / (prop_diff + 1e-6)
        weights_sum = np.sum(weights)
        if weights_sum <= 0:
            weights = np.ones(n_cand, dtype=np.float32) / float(n_cand)
        else:
            weights = weights / weights_sum

        for y, m_arr in m_luc_arrays.items():
            cand_vals = m_arr[cand_idx]
            is_undisturbed = (cand_vals == 1).astype(np.float32)
            res_s_prop[y][k_idx] = float(np.sum(weights * is_undisturbed))

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
        f"(weighted proportions computed in {time.perf_counter()-t2:.2f}s)",
        flush=True,
    )

    del m_set, k_subset, m_dist, k_dist, m_hard, k_hard
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
    (
        m_parquet_filename,
        sample_index,
        sample_size,
        seed,
        m_samples_dir,
        start_year,
        parquet_compression,
        parquet_compression_level,
    ) = args
    logging.info(f"Creating sample {sample_index + 1} with seed {seed} (filter luc_{start_year}==1)")
    # Read only needed columns where possible to reduce memory
    luc_years = [start_year]
    luc_cols = [f"luc_{y}" for y in luc_years]
    luc0, luc5, luc10 = luc_matching_columns(start_year)
    needed = _unique_preserve_order(DISTANCE_COLUMNS + ["country", "ecoregion", luc0, luc5, luc10] + luc_cols)
    available = _parquet_columns(m_parquet_filename)
    to_read = [c for c in needed if c in available]
    if to_read:
        m_set = pd.read_parquet(m_parquet_filename, columns=to_read)
    else:
        m_set = pd.read_parquet(m_parquet_filename)
    m_set = _drop_duplicate_columns(m_set, f"M source sample {sample_index + 1}")
    # Filter to only rows undisturbed in the start year if the column exists
    start_luc_col = f"luc_{start_year}"
    if start_luc_col in m_set.columns:
        m_set = m_set[m_set[start_luc_col] == 1].reset_index(drop=True)

    # Downcast numeric columns to save memory
    for c in DISTANCE_COLUMNS:
        if c in m_set.columns:
            try:
                m_set[c] = m_set[c].astype(np.float32)
            except Exception:
                pass
    for c in luc_cols:
        if c in m_set.columns:
            try:
                m_set[c] = m_set[c].astype(np.int8)
            except Exception:
                pass
    for c in ["country", "ecoregion", luc0, luc5, luc10]:
        if c in m_set.columns:
            try:
                m_set[c] = m_set[c].astype(np.int32)
            except Exception:
                pass
    rng = np.random.default_rng(seed)
    replace = len(m_set) < sample_size
    indices = rng.choice(len(m_set), size=sample_size, replace=replace)
    sample = m_set.iloc[indices].reset_index(drop=True)
    sample = sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)
    sample.to_parquet(
        os.path.join(m_samples_dir, f"m_sample_{sample_index + 1:03d}.parquet"),
        index=False,
        compression=parquet_compression,
        compression_level=parquet_compression_level,
    )
    del sample, m_set
    gc.collect()
    return f"Completed sample {sample_index + 1}"


def create_m_samples(
    m_parquet_filename,
    output_folder,
    num_samples,
    sample_size,
    seeds,
    sample_processes=None,
    start_year: Optional[int] = None,
    parquet_compression: str = "zstd",
    parquet_compression_level: int = 3,
):
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
    args_list = [
        (
            m_parquet_filename,
            i,
            sample_size,
            seeds[i],
            m_samples_dir,
            start_year,
            parquet_compression,
            parquet_compression_level,
        )
        for i in missing
    ]
    with Pool(processes=sample_processes, initializer=_worker_init) as pool:
        for r in pool.map(create_single_m_sample, args_list):
            logging.info(r)


def _iteration_wrapper(args):
    (m_sample, k_grid, iter_seed, shuf_seed, start_year, eval_year, out, max_pm) = args
    return find_proportion_iteration(m_sample, start_year, eval_year, out, (k_grid, iter_seed), shuf_seed, max_pm)


def _get_k_grid_id_from_path(k_filepath: str) -> str:
    """Return a stable K grid id extracted from the K filepath (used for output names)."""
    k_grid_filename = os.path.basename(k_filepath)
    match = re.search(r"\d+", k_grid_filename)
    return match.group(0) if match else k_grid_filename.replace(".parquet", "")


def _k_outputs_exist(k_filepath: str, output_folder: str) -> bool:
    """Return True if both expected output parquet files exist and are non-empty."""
    k_id = _get_k_grid_id_from_path(k_filepath)
    p1 = os.path.join(output_folder, f"{k_id}.parquet")
    p2 = os.path.join(output_folder, f"{k_id}_matchless.parquet")
    return os.path.exists(p1) and os.path.getsize(p1) > 0 and os.path.exists(p2) and os.path.getsize(p2) > 0


def find_pairs(
    k_directory, m_parquet_filename, m_sample_folder,
    start_year, evaluation_year, seed,
    output_folder, batch_size, processes_count, max_potential_matches=100,
    m_sample_count: Optional[int] = None,
    m_sample_size: int = 2_000_000,
    m_sample_compression: str = "zstd",
    m_sample_compression_level: int = 3,
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
        m_sample_files = sorted(glob.glob(os.path.join(m_samples_dir, "m_sample_*.parquet")))
        if not m_sample_files:
            raise FileNotFoundError(f"No m_sample_*.parquet files found in {m_samples_dir}")
        logging.info(f"Found {len(m_sample_files)} existing M sample files")
    else:
        # Create a bounded reusable pool of M samples (instead of one sample per K grid)
        if m_sample_count is None:
            effective_m_sample_count = max(1, min(num_k, max(8, min(32, processes_count * 2))))
        else:
            effective_m_sample_count = max(1, min(num_k, int(m_sample_count)))

        total_rows = _parquet_num_rows(m_parquet_filename)
        if total_rows is None:
            logging.info(
                f"Creating {effective_m_sample_count} M samples, sample size {m_sample_size:,} rows"
            )
        else:
            logging.info(
                f"M set {total_rows:,} rows; creating {effective_m_sample_count} reusable M samples "
                f"of {m_sample_size:,} rows"
            )

        create_m_samples(
            m_parquet_filename,
            output_folder,
            effective_m_sample_count,
            m_sample_size,
            iter_seeds,
            min(8, processes_count),
            start_year=start_year,
            parquet_compression=m_sample_compression,
            parquet_compression_level=m_sample_compression_level,
        )
        m_samples_dir = os.path.join(output_folder, "m_samples")
        m_sample_files = [
            os.path.join(m_samples_dir, f"m_sample_{i + 1:03d}.parquet")
            for i in range(effective_m_sample_count)
        ]

    # Build map args but skip K grids whose outputs already exist (both files present & non-empty)
    map_args = []
    skipped = 0
    for i, (kf, iseed, sseed) in enumerate(zip(k_files, iter_seeds, shuf_seeds)):
        if _k_outputs_exist(kf, output_folder):
            skipped += 1
            continue
        msf = m_sample_files[i % len(m_sample_files)]
        map_args.append((msf, kf, iseed, sseed, start_year, evaluation_year, output_folder, max_potential_matches))

    if skipped:
        logging.info(f"Skipped {skipped} K grids because outputs already exist")

    processed = []
    with Pool(processes=processes_count, initializer=_worker_init) as pool:
        for i in range(0, len(map_args), batch_size):
            batch = map_args[i : i + batch_size]
            logging.info(f"Batch {i // batch_size + 1}/{(len(map_args) + batch_size - 1) // batch_size}")
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
        description="Counterfactual propensity-weighted proportion matching: for undisturbed K pixels, "
                    "compute weighted proportion of candidate M pixels with luc==1 each year."
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
    parser.add_argument("--m_sample_count", type=int, default=None, help="Number of reusable M sample parquet files to maintain (default: bounded by CPU/K count)")
    parser.add_argument("--m_sample_size", type=int, default=2_000_000, help="Rows per M sample parquet")
    parser.add_argument("--m_sample_compression", type=str, default="zstd", help="Compression codec for generated M sample parquet files")
    parser.add_argument("--m_sample_compression_level", type=int, default=3, help="Compression level for generated M sample parquet files")
    parser.add_argument("--use-fork", action="store_true", help="Use fork start method for multiprocessing (Linux only). Allows copy-on-write sharing to reduce memory per worker.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Choose multiprocessing start method before creating pools
    try:
        if args.use_fork:
            set_start_method("fork")
            logging.info("Using multiprocessing start method: fork")
        else:
            set_start_method("spawn")
            logging.info("Using multiprocessing start method: spawn")
    except RuntimeError:
        # already set; ignore
        pass

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
        m_sample_count=args.m_sample_count,
        m_sample_size=args.m_sample_size,
        m_sample_compression=args.m_sample_compression,
        m_sample_compression_level=args.m_sample_compression_level,
    )


if __name__ == "__main__":
    main()
