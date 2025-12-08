# Temporal CT Generation and Progression Modeling with Latent Flow Matching Diffusion

## Setup Environment
To set up the environment, run:

```bash
mamba env create -f environment.yml
```
To update the environment after changes to `environment.yml`, run:

```bash
mamba env update -n ct_venv -f environment.yml
```
If you don't have cuda installed or are not on a linux system you can build without cuda support by running:

```bash
mamba env create -f environment_no_build.yml
```

Next, install the package in editable mode (from the repo root directory where README.md is located):

**Note:** `requirements.in` should be all the top level installed packages.

```bash
pip install -e .
```

## Loading Data

In docs process.md will give the full overview of the data loading process. Here is a brief summary of the process and parquet files used for loading data.

## Parquet Files Overview (Most of these are created when running the data construction scripts)

### Note exact names of parquet files may vary slightly

- **all_exams.parquet** – One row per exam, raw data.
- **single_timepoints.parquet** – One row per exam, cleaned data with only one exam at each timepoint
- **patient_timelines.parquet** – One row per patient with all their exams
- **paired_exams.parquet** – One row per consecutive exams
- **nlst_nodule_tracking.parquet** – Nodule tracking information. One nodule per row.


## Workflow


### Notes for Zack
The path to all the encoded data is: /data/rbg/scratch/nlst_encoded/train5

## Acknowledgments

This project builds on code from:
- SybilX, MIT License (c) 2021 Peter Mikhael & Jeremy Wohlwend

Special thanks to Benny Grey for contributions to the upstream project
that informed evaluation of the models.

