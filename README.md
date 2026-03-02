# Temporal CT Generation and Progression Modeling with Latent Flow Matching Diffusion

## Overview

This repository implements generative modeling of lung CT scans with a focus on **temporal progression modeling of pulmonary nodules** using latent Flow Matching.

The core objective of this project is:

> **To generate realistic future CT scans conditioned on prior scans and time intervals**, enabling modeling of longitudinal disease progression.

The repository contains:

* Unconditional 2D latent generative models
* Unconditional 3D latent generative models
* Conditional 3D temporal generative models
* A full NLST data construction pipeline
* Modular reusable modeling components

Generative models operate both in **latent space** learned via a 3D VAE and in pixel space for high-resolution image generation.


---

## Conditional 3D Temporal Generation (Primary Contribution)

### Conditional Flow Matching with Time

Input:

* CT scan at time ( $t_0$ )
* Time delta ( $\Delta t$ )

Output:

* Predicted CT scan at time ( $t_1$ )

This allows:

* Modeling nodule growth
* Learning disease progression
* Generating plausible future scans
* Studying longitudinal latent dynamics

Time conditioning is implemented using time embeddings injected into a 3D UNet-based architecture.

---

## Flow Matching

We use Flow Matching rather than traditional diffusion:

* Deterministic ODE formulation
* Direct velocity prediction
* Stable training
* Efficient Euler sampling

Both unconditional and conditional variants are implemented in reusable modules.

---

## Repository Layout

A full layout can be found in the docs folder. A brief layout of key code is as follows:

```
scripts/       # Training & evaluation entry points
src/CTFM/
│
├── data/          # NLST dataset handling, bounding boxes, metadata
├── eval/          # Segmentation model and evaluation metrics
├── models/        # VAE, 2D/3D UNet, Flow Matching implementations
├── utils/         # Logging, visualization, training utilities
```

All modules are importable from:

```
CTFM.data
CTFM.models
CTFM.utils
```


---

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

### Note: All the modules are importable from CTFM.data/models/utils

---

## Loading Data

In `docs/process.md` you will find a full overview of the NLST data construction pipeline.

Below is a high-level summary.

---

## NLST Data Pipeline

This repository contains a complete workflow from raw NLST DICOM scans to training-ready tensors.

### Pipeline Steps

1. Load raw DICOM exams
2. Clean metadata
3. Construct patient timelines
4. Pair consecutive exams
5. Perform image registration
6. Extract nodules
7. Generate fixed bounding boxes
8. Encode volumes using trained VAE
9. Cache latent tensors for Flow Matching

---

## Parquet Files Overview

(Most of these are created when running the data construction scripts)

### Note: exact names may vary slightly

* **all_exams.parquet** – One row per exam, raw data.
* **single_timepoints.parquet** – One row per exam, cleaned data with only one exam at each timepoint
* **patient_timelines.parquet** – One row per patient with all their exams
* **paired_exams.parquet** – One row per consecutive exams
* **nlst_nodule_tracking.parquet** – Nodule tracking information. One nodule per row.
* **nodules_with_fixed_bboxes.parquet** – Nodule information with fixed bounding boxes for registered images.
* **paired_nodules.parquet** – One row per consecutive nodules.

---

## Encoded Data

Encoded latent tensors are saved separately to enable fast training.

---

## Training

Training scripts are located in:

```
scripts/
```

Includes:

* 2D unconditional Flow Matching
* 3D unconditional Flow Matching
* 3D conditional Flow Matching
* VAE training
* Evaluation scripts

---

## Evaluation

Utilities include:

* Reconstruction metrics
* Bounding box overlays
* Slice montage visualization
* Temporal consistency checks

Evaluation code lives in:

```
CTFM/eval/
```

---

## Research Contributions

This repository enables:

* Latent 3D Flow Matching for CT
* Temporal conditioning for volumetric generation
* Longitudinal NLST processing
* Nodule-aware training objectives
* Modular generative modeling infrastructure

---

## Acknowledgments

This data pipeline from this project builds on code from:
- SybilX, MIT License (c) 2021 Peter Mikhael & Jeremy Wohlwend

Special thanks to Peter Mikhael for all his mentorship in completing this project.

Special thanks to Benny Grey for contributions to the upstream project
that informed evaluation of the models.



