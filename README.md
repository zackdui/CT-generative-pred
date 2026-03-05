# Temporal CT Generation and Progression Modeling with Latent Flow Matching Diffusion

## Overview

This repository implements generative modeling of lung CT scans with a focus on **temporal progression modeling of pulmonary nodules** using Flow Matching.

The core objective of this project is:

> **To generate realistic future CT scans conditioned on prior scans and time intervals**, enabling modeling of longitudinal disease progression.

The repository contains:

* 3D VAE training code
* Unconditional 2D latent generative models
* Unconditional 3D latent generative models
* Conditional 3D temporal generative models
* A full NLST data construction pipeline
* Modular reusable modeling components

Generative models operate both in latent space and in pixel space for high-resolution image generation.


---

## Conditional 3D Temporal Generation

The central modeling task in this repository is conditional 3D temporal generation. Given a CT volume acquired at time $t_0$ and a time interval $\Delta t$, the model predicts a future volume at time $t_1$. Time information is incorporated through learned temporal embeddings that are injected into a 3D UNet-based architecture, enabling the model to learn structured longitudinal transformations directly in volumetric space.

Generation is formulated using Flow Matching rather than traditional diffusion. The model learns a deterministic velocity field and produces samples via numerical integration, allowing stable training, efficient sampling, and a natural mechanism for conditioning. Both unconditional and conditional variants are implemented within the same modular framework.

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
CTFM.eval
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

### Note: All the modules are importable from CTFM.data/eval/models/utils

---

## Loading Data

In `docs/process.md` you will find a full overview of the NLST data construction pipeline which  contains a complete workflow from raw NLST DICOM scans to training-ready tensors.

Below is a high-level summary.

### Pipeline Steps

1. Load raw DICOM exams
2. Clean metadata
3. Construct patient timelines
4. Pair consecutive exams
5. Perform image registration
6. Extract nodules
7. Encode volumes using trained VAE
8. Cache latent tensors for Flow Matching

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

* Bounding box overlays
* Slice montage visualization
* Nodule Segmentation and volume calculations

Evaluation code lives in:

```
CTFM/eval/
```

---

## Research Findings

This work investigates whether Flow Matching–based generative models can model longitudinal changes in chest CT scans, with a specific focus on pulmonary nodule growth. A complete 2D and 3D generative pipeline was developed, including unconditional and conditional models operating in both latent and pixel space, along with a tailored 3D variational autoencoder and spatially weighted objectives to emphasize clinically relevant regions. While the models successfully preserve global anatomical structure, accurately modeling small, localized volumetric growth remains challenging, highlighting the difficulty of inducing meaningful change within predominantly static anatomy.

---

## Acknowledgments

The data pipeline from this project builds on code from:
- SybilX, MIT License (c) 2021 Peter Mikhael & Jeremy Wohlwend

Special thanks to Peter Mikhael for all his mentorship in completing this project.

Special thanks to Benny Grey for contributions to the upstream project
that informed evaluation of the models.



