#!/usr/bin/env python3

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DISTANCE_COLUMNS = [
    "elevation", "slope", "access",
    "fcc0_u", "fcc0_d",
    "fcc5_u", "fcc5_d",
    "fcc10_u", "fcc10_d",
]


def luc_matching_columns(start_year: int):
    return (
        f"luc_{start_year}",
        f"luc_{start_year - 5}",
        f"luc_{start_year - 10}",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Fast M sample generation for Julia matcher")
    parser.add_argument("--m_parquet_filename", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--sample_size", type=int, required=True)
    parser.add_argument("--start_year", type=int, required=True)
    parser.add_argument("--seeds", required=True, help="Comma-separated integer seeds")
    parser.add_argument("--writer_threads", type=int, default=8)
    return parser.parse_args()


def parse_seeds(text: str):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main():
    args = parse_args()
    seeds = parse_seeds(args.seeds)

    if len(seeds) < args.num_samples:
        raise ValueError(f"Need at least {args.num_samples} seeds, got {len(seeds)}")

    m_samples_dir = os.path.join(args.output_folder, "m_samples")
    os.makedirs(m_samples_dir, exist_ok=True)

    missing = []
    for i in range(1, args.num_samples + 1):
        p_parquet = os.path.join(m_samples_dir, f"m_sample_{i:03d}.parquet")
        p_arrow = os.path.join(m_samples_dir, f"m_sample_{i:03d}.arrow")
        if not os.path.isfile(p_parquet) and not os.path.isfile(p_arrow):
            missing.append(i)

    if not missing:
        print("[python-sampler] All M samples already exist. Skipping.")
        return

    luc0, luc5, luc10 = luc_matching_columns(args.start_year)
    start_luc_col = f"luc_{args.start_year}"

    all_cols = list(pq.ParquetFile(args.m_parquet_filename).schema.names)
    luc_cols_all = [c for c in all_cols if c.startswith("luc_")]
    prop_cols_all = [c for c in all_cols if "prop" in c.lower()]

    needed = []
    for c in (
        DISTANCE_COLUMNS
        + ["lat", "lng"]
        + ["country", "ecoregion", luc0, luc5, luc10, start_luc_col]
        + luc_cols_all
        + prop_cols_all
    ):
        if c in all_cols and c not in needed:
            needed.append(c)

    print(f"[python-sampler] Loading source M parquet once with {len(needed)} columns...")
    m_source = pd.read_parquet(args.m_parquet_filename, columns=needed, engine="pyarrow")

    if start_luc_col in m_source.columns:
        m_source = m_source[m_source[start_luc_col] == 1].reset_index(drop=True)

    n_m = len(m_source)
    if n_m == 0:
        raise ValueError(f"No rows available in M after filtering {start_luc_col}==1")

    print(f"[python-sampler] Creating {len(missing)} samples from {n_m:,} rows")

    def write_one(sample_num: int):
        seed = seeds[sample_num - 1]
        rng = np.random.default_rng(seed)

        replace = n_m < args.sample_size
        idx = rng.choice(n_m, size=args.sample_size, replace=replace)
        sample = m_source.iloc[idx].reset_index(drop=True)

        sample_path = os.path.join(m_samples_dir, f"m_sample_{sample_num:03d}.parquet")
        table = pa.Table.from_pandas(sample, preserve_index=False)
        pq.write_table(table, sample_path, compression="snappy", use_dictionary=False)
        return sample_num

    workers = max(1, min(args.writer_threads, len(missing), os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(write_one, i) for i in missing]
        for fut in as_completed(futures):
            i = fut.result()
            print(f"[python-sampler] Completed sample {i:03d}")

    print("[python-sampler] All requested samples created.")


if __name__ == "__main__":
    main()
