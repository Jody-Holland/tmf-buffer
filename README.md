# TMF Buffer Pipeline

This repository provides a reproducible pipeline for constructing **ex post counterfactual dynamic baselines** for conservation interventions and their surrounding landscapes in tropical moist forest regions.

This is a fork of http://github.com/quantifyearth/tmf-implementation/ by Dr Michael Dales and the Cambridge Centre for Carbon Credits.

The workflow prepares project and landscape covariates, identifies comparable control pixels, builds matched treatment–control pairs, and generates impact outputs (including additionality summaries and effect rasters).

## What this pipeline does

- Builds project boundary products (including surrounding buffers and matching areas).
- Assembles biophysical and contextual covariates (e.g. elevation, slope, accessibility, forest cover change).
- Calculates candidate treatment/control sets (`K` and `M` tables).
- Runs hybrid matching in Julia (propensity + Mahalanobis weighting).
- Produces analysis-ready outputs for ex post impact and dynamic baseline assessment.

## Repository layout

- `pipeline.sh`: interactive/CLI orchestration for end-to-end runs.
- `pipeline.conf`: data paths and runtime configuration.
- `methods/inputs`: data preparation and raster/vector preprocessing.
- `methods/matching`: matching and pair-generation steps.
- `methods/outputs`: additionality/effect outputs.
- `methods/common`, `methods/utils`: shared utilities.

## Requirements

### Python

Install dependencies from `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Some environments in this project use `tmfpython3`; if that alias is not configured on your system, use `python3` with the same module commands.

### Julia

The matching stage (step 13) uses Julia and the project environment in this repository.

From the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Main Julia packages used by the matcher include:

- `DataFrames`
- `Parquet2`
- `Arrow`
- `CodecZstd`
- `GLM`
- `Distributed`

## Configuration

Edit `pipeline.conf` to set:

- input and output directories,
- project/vector/raster source paths,
- run-time controls such as `PROCESSES`, `BATCH_SIZE`, and `K_NUM_TO_KEEP`.

## Running the pipeline

Start from repository root:

```bash
bash pipeline.sh
```

The script supports:

- running all or selected steps,
- single-project mode,
- batch mode from a CSV (`project,start_year,evaluation_year`).

## Key outputs

Outputs are written under `${OUTPUT_DIR}/${PROJECT_ID}` and typically include:

- buffered project geometry,
- matching area and country mask layers,
- `k_grids` and `matches.parquet`,
- pair outputs from the hybrid matcher,
- additionality and effect products (`effect.parquet`, `effect.tif`).

## Notes

- This repository assumes access to large geospatial input datasets defined in `pipeline.conf`.
- Runtime and memory use can be substantial for large projects; tune `PROCESSES`/`BATCH_SIZE` as needed.