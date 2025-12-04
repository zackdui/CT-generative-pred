This is the full layout of this git repo.

---

## Layout of this ML repo

```text
CT-generative-pred/
├── pyproject.toml 
├── environment.yaml
├── environment-no-build.yaml          
├── README.md
├── AUTHORS.md
├── LICENSE                  
├── .gitignore
├── configs/                  # YAML or JSON configs for experiments
│   ├── nlst_large.yaml
│   ├── nlst_small.yaml
│   └── paths.yaml
├── scripts/                  # CLI entrypoints / experiment scripts
│   ├── train_ct_ae.py
│   ├── train_classifier.py
│   ├── eval_ct_ae.py
│   └── inspect_dataset.py
├── src/
│   └── CTFM/         # importable as `import your_project`
│       ├── __init__.py
│       ├── data/
│       │   ├── create_parquets.py       # Create all the metadata files for easy loading
│       │   ├── datasets.py       # PyTorch Dataset & DataLoader wrappers
│       │   ├── transforms.py     # augmentations, TorchIO pipelines, etc.
│       │   └── utils.py          # small helpers for data paths, splits
│       ├── models/
│       │   ├── unet.py
│       │   ├── autoencoder.py
│       │   ├── diffusion.py
│       │   └── __init__.py       # convenient exports
│       ├── training/
│       │   ├── trainer.py        # training loop / LightningModule / etc.
│       │   ├── callbacks.py      # checkpointing, early stopping, logging
│       │   └── optimizers.py     # schedulers, custom optimizers
│       ├── evaluation/
│       │   ├── metrics.py        # metrics: dice, SSIM, AUROC, etc.
│       │   └── eval_loops.py     # evaluation pipelines / scripts
│       ├── utils/
│       │   ├── registration_logging.py        # Save registration logs to update parquet files
│       │   ├── config.py         # config loading 
│       │   ├── __init__.py
│       │   ├── plot_lr.py
│       │   ├── pixel_conversions.py
│       │   └── data_size.py
│       └── cli.py                # optional: a unified CLI (typer/click)
├── notebooks/                # Jupyter exploratory stuff (not core logic)
│   └── 01_data_exploration.ipynb
├── tests/                    # unit / integration tests (pytest)
│   ├── test_datasets.py
│   ├── test_models.py
│   └── test_training_loop.py
├── data/                     # local data mount, kept out of git
│   └── (empty, gitignored)
├── metadata/                     # local data mount, kept out of git
│   └── (empty, gitignored)
├── docs/                     # Extra documention and information
│   ├── process.md
│   └── repo_layout.md
└── experiments/              # logs, checkpoints, outputs (gitignored)
    ├── runs/
    └── debug/
```

### Key principles behind this layout

* **All importable code lives under `src/CTFM/`**

  * This makes it easy to install with:

    ```bash
    pip install -e .
    ```

    and use from anywhere: `from CTFM.models import unet`.

* **Separate library code from scripts**

  * `src/your_project/...` contains reusable stuff.
  * `scripts/` calls into that library with configs / CLI args.
  * This keeps your training logic reusable across experiments.

* **Configs instead of hardcoding**

  * Use `configs/*.yaml` for:

    * model hyperparams (channels, depths, etc.)
    * data paths
    * training params (lr, epochs, batch size)
  * Scripts read a config path:

    ```bash
    python scripts/train_ct_ae.py --config configs/experiment_01.yaml
    ```

* **Keep big/volatile stuff out of Git**

  * Add to `.gitignore`:

    * `data/`
    * `experiments/` or `runs/`
    * `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`
    * large temp artifacts

* **Tests live separately**

  * Even a few small tests for datasets and models can save you from silent bugs.

