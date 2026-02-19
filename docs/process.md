# Full Process for Data Creation and Experimentation

This document describes the full end-to-end process used to generate datasets and run experiments.

Unless otherwise stated, **all scripts are located in the `scripts/` directory**.

---

## 1. Data Creation

### 1.1 Create Exam ↔ NIfTI Mappings

1. Run `save_mapping_from_exams_to_nifti` in `save_mapping`  
   - Output:  
     - `metadata/zack_exam_to_nifti.pkl`

2. Within `save_mapping`, also run `save_exam_registration_mapping`  
   - Output:  
     - `metadata/mapping_registered_exams.pkl`

---

### 1.2 Convert Nodule Annotations

3. Run `nodules_pt_to_parquet`  
   - Converts nodule `.pt` files into a parquet format

---

### 1.3 Create Dataset Parquets

4. Run `run_create_parquets.py`  
   - Run this for train / val / test to create all the parquets

---

### 1.4 Registration Outputs and Updates

5. *(Optional)* Run `write_new_nifti_files` in `save_registrations`  
   - I only ran this **once** to generate test files

6. Run `write_registrations` for:
   - train
   - val
   - test  

   This step saves updates parquets corresponding to each update.

7. If the process stops early:
   - Run `update_parquet` in `scripts/save_registrations`

8. Run `remove_bad_exams_flag` in `save_registrations`


---

### 1.5 Bounding Boxes in Transformed Space

9. Run `save_bounding_boxes` in `save_registrations`  
    - Creates bounding boxes in **transformed space**

---

### 1.6 Notes on File Versions

⚠️ **Important:**  
Every time a script is run, ensure that **all file names and paths are updated** to point to the most recent versions of intermediate outputs.

---

### 1.7 Additional Registration Testing

10. Run `write_new_nifti_files` in `processing`  
    - Saved **100 transformed files** for testing  
    - Used by `save_registrations`

11. *(Optional – not used in final version)*  
   - Run `save_transformed_mapping` in `save_mapping`  
   - Output:
     - `metadata/exam_to_nifti_transformed.pkl`


---

### 1.8 Paired Nodule Parquets

12. In the parquet creation stage, run  for train/val/test
    `paired_exams_by_pid_nodule_group`  
    - Creates **paired nodule parquets** for:
      - train
      - val
      - test
    - Location: `scripts/save_nodule_cache`

---

### 1.9 Nodule Cache Creation

13. Save and consolidate cached nodule tensors for:
    - train
    - val
    - test

    Location: `scripts/save_nodule_cache`

---

### 1.10 Optional 2D Encoded Outputs

14. *(Optional)* Save 2D encoded images  
    - Script: `save_encoded_images`

---

### 1.11 End of Initial Dataset Setup

At this point, the dataset creation pipeline is complete.

---

### 1.12 Create and Save Small Bounding Boxes

15. Run `save_small_bboxes` or `save_small_bboxes_encoded`  
    - Creates small bounding boxes for nodules. Bounding boxes around the nodule after extracting the 3D boxes.

---

## 2. Variational Autoencoder (VAE)

15. Train a 3D VAE on nodules:
    - `train_vae_3d_nodules`

16. *(Alternative)* Train on full volumes:
    - `train_vae_3d`

17. Save encoded representations:
    - Encoded nodules:
      - `scripts/save_nodule_cache`
    - Fully encoded 3D volumes:
      - `scripts/save_encoded_images`

---

## 3. Unconditional Flow Matching Pretraining

As a pretraining step, flow matching models are trained unconditionally.

18. Run:
    - `scripts/pretraining_2d_fm.py`
    - `scripts/pretraining_3d_fm.py`

---

## 4. Paired Conditional Flow Matching

19. Train paired conditional flow matching models:
    - `train_pairs_2d_fm`
    - `train_pairs_3d_fm`

---

## 5. Evaluation (Run for Every Test Experiment)

20. Run evaluation scripts:
    - `eval_2D_fm.py`
    - `eval_3D_fm.py`

---



