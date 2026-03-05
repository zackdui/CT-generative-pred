This is the full layout of this git repo. Also contains files that should be created when
running the code in the repo.

---

## Layout of this ML repo

```text
CT-generative-pred/
├── pyproject.toml
├── environment.yml
├── environment_no_build.yml
├── requirements.in
├── README.md
├── TODO.md
├── AUTHORS.md
├── LICENSE
├── .gitignore
│
├── configs/
│   ├── eval_3d_vae.yaml
│   ├── fm_3d_paired.yaml
│   ├── fm_3d_pretraining.yaml
│   ├── nlst_large.yaml
│   ├── nlst_small.yaml
│   ├── paths.yaml
│   ├── train_vae_3d_nodule.yaml
│   ├── train_vae_3d.yaml
│   ├── unet_2d_cfm.yaml
│   ├── unet_3d_cfm_raw.yaml
│   ├── unet_3d_cfm.yaml
│   └── vae_3d_model.yaml
│
├── extra_functional_code_files/      # This is for code that was used for small tasks
├── final_saved_models/
│
├── metadata/
│   ├── train/                # (.gitignored) This is built locally when running create_parquets script
│       ├── nlst_full.parquet
│       ├── nodules_with_fixed_bboxes.parquet
│       ├── paired_nodules.parquet
│       ├── full_data_single_timepoints.parquet
│       ├── full_data_timelines.parquet
│       └── full_data_paired_exams.parquet
│   ├── val/              # Same structure as train also (.gitignored)
│   ├── test/             # Same structure as train also (.gitignored)
│   ├── mapping_registered_exams.pkl
│   ├── nlst_nodule_tracking.parquet
│   └── zack_exam_to_nifti.pkl
│
├── output_metrics/
│
├── docs/
│   ├── parquets_explanations.md
│   ├── process.md
│   └── repo_layout.md
│
├── scripts/
│   ├── eval_2D_fm.py
│   ├── eval_3D_fm_raw.py
│   ├── eval_3D_fm.py
│   ├── eval_3d_vae.py
│   ├── model_diagram_save.py
│   ├── nodules_pt_to_parquet.py
│   ├── pretraining_2D_fm.py
│   ├── pretraining_3D_fm.py
│   ├── run_create_parquets.py
│   ├── save_encoded_images.py
│   ├── save_mapping.py
│   ├── save_nodule_cache.pys
│   ├── save_registrations.py
│   ├── save_small_bboxes_encoded.py
│   ├── save_small_bboxes.py
│   ├── train_pairs_2D_fm.py
│   ├── train_pairs_3D_fm.py
│   ├── train_vae_3d_nodules.py
│   └── train_vae_3d.py
│
├── src/
│   └── CTFM/
│       ├── __init__.py
│       │
│       ├── data/
│       │   ├── datasets/
│       │   │   ├── CT_orig_data.py
│       │   │   ├── cached_tensors_data.py
│       │   │   └── nlst_base.py
│       │   ├── bounding_boxes.py
│       │   ├── cache_encoded.py
│       │   ├── create_parquets.py
│       │   ├── flatten_nodules.py
│       │   ├── initial_exam_to_nifti.py
│       │   ├── processing.py
│       │   ├── utils.py
│       │   └── __init__.py
│       │
│       ├── models/
│       │   └── unet_2d/
│       │       ├── __init__.py
│       │       ├── auto_encoder_2d.py
│       │       ├── lightning_3d_cfm.py
│       │       └── lightning_2d_cfm.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── config.py
│           ├── custom_loggers.py
│           ├── data_size.py
│           ├── pixel_conversions.py
│           └── plot_lr.py
├── model_images/ 
├── final_saved_models/        # (.gitignored - maybe) Saved final models after training
└── experiments/              # logs, checkpoints, outputs (gitignored)
    ├── fm_2d_paired/
    ├── fm_2d_pretrain/
    ├── fm_3d_paired/
    ├── fm_3d_pretrain/
    ├── vae_2d/
    └── vae_3d/
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

* **Configs instead of hardcoding**

  * Use `configs/*.yaml` for:

    * model hyperparams (channels, depths, etc.)
    * data paths
    * training params (lr, epochs, batch size)


