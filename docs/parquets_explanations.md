
# Parquet File Explanations

---

## Parquet File Description

**File Name(s):**

* `full_data.parquet` (can be train, val, test splits as `full_data_train.parquet`, etc.)


**Created By:** `create_full_data_parquet(...)`

**Row Granularity:** One row per **exam instance** (i.e., one CT screening exam for a given patient, with associated metadata and file-location info).

Each row represents a single exam, with labels, timing information, and pointers to either a NIfTI file or raw DICOM paths.

---

### Columns

| **Column Name**         | **Type**              | **Description**                                                                                                                                                | **Example Value**                      |
| ----------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `cancer`                | `int`                 | Binary cancer label for the exam (`0` = no cancer, `1` = cancer), derived from `sample["y"]`.                                                                  | `0`                                    |
| `exam_id`               | `str`                 | **Primary unique exam identifier**, used for saved NIfTI files and registrations.                                                                              | `"NLST_000123_T1"`                     |
| `exam`                  | `str`                 | Internal exam index/ID (cast to `str`), may mirror `exam_str` indexing used downstream.                                                                        | `"1"`                                  |
| `num_slices`            | `int`                 | Number of original slices in the source volume (`sample["num_original_slices"]`).                                                                              | `320`                                  |
| `time_at_event`         | `float`               | Continuous time-to-event value (e.g., time until cancer or censoring) in the chosen time units.                                                                | `820.0`                                |
| `days_at_event`         | `float`               | Days from exam date to event date or censoring.                                                                                                                | `820.0`                                |
| `exam_str`              | `str`                 | Human-readable exam identifier string (e.g., `<PID>_T<k>`).                                                                                                    | `"NLST_000123_T1"`                     |
| `pid`                   | `str`                 | Unique patient identifier.                                                                                                                                     | `"NLST_000123"`                        |
| `series`                | `str`                 | Series identifier or DICOM series name/path associated with this exam.                                                                                         | `"1.3.6.1.4.1.5962.1.1.0.12345.1.1.1"` |
| `y_seq`                 | `list[float]`         | Sequence-level targets or features (converted to `float32` then `list`).                                                                                       | `[0.1, 0.3, 0.5, 0.7]`                 |
| `y_mask`                | `list[float]`         | Mask for `y_seq` (stored as float32 list, typically `0.0`/`1.0`), indicating valid positions.                                                                  | `[1.0, 1.0, 1.0, 0.0]`                 |
| `pixel_spacing`         | `list[float]` (len=3) | Pixel spacing values in mm, converted to `float32` then `list`. **Convention:** `[dx, dy, slice_thickness]`. The **last element is the true slice thickness**. | `[0.73, 0.73, 1.25]`                   |
| `accession`             | `str`                 | Accession number for the study/exam.                                                                                                                           | `"ACCN_987654"`                        |
| `screen_timepoint`      | `int`                 | Screening timepoint index (`0` = baseline, `1` = first follow-up, etc.).                                                                                       | `1`                                    |
| `slice_thickness_class` | `float`               | Slice-thickness value or binned slice-thickness numeric code (from `sample["slice_thickness"]`), used for grouping scans by thickness.                         | `1.0`     |
| `split`                 | `str`                 | Dataset split membership for this exam (`"train"`, `"val"`, `"test"`).                                                                                         | `"train"`                              |
| `nifti_path`            | `str` or `None`       | Absolute path to the saved NIfTI file for this exam if available (`exam_id` is used in naming).                                                                | `"/data/.../NLST_000123_T1.nii.gz"`    |
| `has_nifti`             | `bool`                | Whether this exam has a valid NIfTI file (`True` if `exam_id` in `exam_to_nifti` and not in `bad_exams`).                                                      | `True`                                 |
| `sorted_paths`          | `str` or `None`       | **JSON-encoded list** of DICOM file paths used for this exam when no NIfTI is available. Stored as `json.dumps(list_of_paths)` Return the list with `json.loads(list_of_paths)`.                                | `"[""/data/.../IM-0001.dcm"", ...]"`   |
| `first_dicom_path`      | `str`                 | The path to the first DICOM file in the exam's sorted paths.                                                                                                   | `"/data/.../IM-0001.dcm"`              |
| `nifti_label`           | `str` or `None`       | "ITK", "NIBABEL_RAW" or "UNKOWN". This is to make sure the nifti was saved in the proper format and used to correct it if not.                                                       | `"ITK"`                                |

---

### Notes

* `exam_id` and `exam` are both unique identifiers at the exam-level; `exam_id` is the **canonical** ID used for NIfTI filenames and registration outputs.
* Either `nifti_path` **or** `sorted_paths` is populated:

  * If `exam_id` is in `exam_to_nifti` and not in `bad_exams`: `nifti_path` is set, `has_nifti = True`, `sorted_paths = None`.
  * Otherwise, `sorted_paths` holds a JSON string of normalized paths via `fix_repeated_shared`, `nifti_path = None`, `has_nifti = False`.
* `pixel_spacing` explicitly encodes slice thickness in the third element; `slice_thickness_class` is used for grouping but retains a numeric representation.

---

## Parquet File Description

**File Name:** `single_timepoints.parquet`

**Created By:** `dedup_timepoints(...)`

**This is the updated ground truth for all metadata!!**

**Row Granularity:** One row per **unique patient–timepoint exam**, after deduplication by minimum slice thickness.

This file is the **canonical per-exam table** used for all downstream longitudinal processing (timelines, pairs, registrations). It is derived from the full-data Parquets and filtered such that each patient/timepoint has exactly one “best” exam (e.g., with minimal slice thickness).

---

### Relationship to `full_data_*.parquet`

`single_timepoints.parquet`:

* **Shares all core columns** from the `full_data_*` Parquets listed above:

  * `cancer`, `exam_id`, `exam`, `num_slices`, `time_at_event`, `days_at_event`,
    `exam_str`, `pid`, `series`, `y_seq`, `y_mask`, `pixel_spacing`,
    `accession`, `screen_timepoint`, `slice_thickness_class`, `split`,
    `nifti_path`, `has_nifti`, `sorted_paths`.
* Is **deduplicated** so that:

  * For each `(pid, screen_timepoint)` pair, exactly **one** exam remains.
  * The chosen exam is typically the one with **minimum slice thickness** (or your current selection criterion).

---

### Additional / Processing Columns

When registration and preprocessing are applied, the following columns are **added** (if not already present):

**note:** They will not be added when a pid only has one timepoint associated with it.

| **Column Name**          | **Type**                               | **Description**                                                                                  | **Example Value**                       |
| ------------------------ | -------------------------------------- | ------------------------------------------------------------------------------------------------ | --------------------------------------- |
| `registration_exists`    | `bool`                                 | Whether a successful registration transform exists for this exam. Initialized as `False`.        | `True`                                  |
| `registration_file`      | `str` or `pd.NA`                       | Path to the registration transform file (e.g., ANTs `.mat` file) if registration exists.         | `"/data/.../forward_fT0_mT1.mat"`       |
| `reverse_transform`      | `bool` or `pd.NA`                       | Whether or not to reverse the transform. (If True the transform is really moving to fixed)         | `True`                                  |
| `fixed_shape`            | `object` (e.g. list[int] or `pd.NA`)   | Shape of the **fixed/reference** image used during registration (e.g., `[Z, H, W]`).             | `[208, 512, 512]`                       |
| `fixed_spacing`          | `object` (e.g. list[float] or `pd.NA`) | Spacing (mm) of the fixed image used for registration.                                           | `[1.0, 1.0, 1.0]`                       |
| `fixed_origin`           | `object` (e.g. list[float] or `pd.NA`) | Origin of the fixed image in physical coordinates.                                               | `[-128.0, -128.0, -128.0]`              |
| `fixed_direction`        | `object` (e.g. list[float] or `pd.NA`) | Direction cosines of the fixed image (flattened orientation matrix). 9 dimentional array that frequently must be reshaped to 3 x 3                            | `[1.0, 0.0, 0.0, 0.0, ...]`             |
| `original_shape`            | `object` (e.g. list[int] or `pd.NA`)   | Shape of the **current** image used during registration (e.g., `[Z, H, W]`).             | `[208, 512, 512]
| `original_spacing`          | `object` (e.g. list[float] or `pd.NA`) | Spacing (mm) of the current/original image used for registration.                                           | `[1.0, 1.0, 1.0]`                       |
| `original_origin`           | `object` (e.g. list[float] or `pd.NA`) | Origin of the current/original image in physical coordinates.                                               | `[-128.0, -128.0, -128.0]`              |
| `original_direction`        | `object` (e.g. list[float] or `pd.NA`) | Direction cosines of the current/original image (flattened orientation matrix). 9 dimentional array that frequently must be reshaped to 3 x 3                            | `[1.0, 0.0, 0.0, 0.0, ...]`             |
| `transformed_nifti_path` | `str` or `pd.NA`                       | Path to the **registered/resampled** NIfTI volume for this exam, if it has been written to disk. | `"/data/.../NLST_000123_T1_reg.nii.gz"` |

---

### Notes

* `single_timepoints.parquet` is considered the **“truest” authoritative table**:

  * All downstream Parquets (`patient_timelines.parquet`, `paired_exam.parquet`) conceptually derive from this table.
  * `nifti_path`, `transformed_nifti_path`, and registration metadata are expected to be **kept up-to-date** here as processing progresses. It will not be up to date in patient timelines or paired exams.

---

## Parquet File Description

**File Name:** `patient_timelines.parquet`

**Created By:** `write_patient_timelines(...)`

**Row Granularity:** One row per **patient**, containing their full exam history aggregated into ordered lists.

Each row summarizes a patient’s longitudinal trajectory over multiple screening timepoints. It is built from `single_timepoints.parquet` but **does not include** registration-specific columns; instead, it aggregates the logical exam-level metadata.

---

### Relationship to `single_timepoints.parquet`

For each `pid`, this file:

* **Groups** all rows from `single_timepoints.parquet` for that patient.
* Sorts exams by `screen_timepoint`.
* Aggregates **non-processing columns** (i.e., excludes `registration_exists`, `registration_file`, `fixed_*`, `transformed_nifti_path`) into per-patient lists.

Concretely, it reuses the same semantics as your original timelines spec, but now based on the updated, deduplicated `single_timepoints` table.

---

### Columns

All list-valued columns below are **aligned** and sorted by `screen_timepoint`.

| **Column Name**         | **Type**                   | **Description**                                                                 | **Example Value**                      |
| ----------------------- | -------------------------- | ------------------------------------------------------------------------------- | -------------------------------------- |
| `pid`                   | `str`                      | Unique patient identifier.                                                      | `"NLST_000123"`                        |
| `exam_strs`             | `list[str]`                | Exam identifiers (from `exam_str`) in chronological order.                      | `["NLST_000123_T0", "NLST_000123_T1"]` |
| `screen_timepoints`     | `list[int]`                | Screening timepoints corresponding to `exam_strs`.                              | `[0, 1]`                               |
| `days_at_events`        | `list[float]`              | `days_at_event` values for each exam.                                           | `[820.0, 530.0]`                       |
| `series`                | `list[str]`                | Series identifiers for each exam.                                               | `["series_001", "series_002"]`         |
| `device`                | `list[str]` or `list[int]` | Scanner/device identifiers for each exam (same type as in original exam table). | `["GE_LIGHTSPEED", "SIEMENS_SOMATOM"]` |
| `cancer`                | `list[int]`                | Cancer labels for each exam (0/1).                                              | `[0, 1]`                               |
| `time_at_event`         | `list[float]`              | `time_at_event` values for each exam.                                           | `[820.0, 530.0]`                       |
| `y_seq`                 | `list[list[float]]`        | Sequence-level data per exam.                                                   | `[[0.0, 0.1], [0.2, 0.4]]`             |
| `y_mask`                | `list[list[float]]`        | Masks for `y_seq` per exam (same float 0/1 convention as `single_timepoints`).  | `[[1.0, 1.0], [1.0, 0.0]]`             |
| `pixel_spacing`         | `list[list[float]]`        | Pixel spacing (including slice thickness in last element) per exam.             | `[[0.7, 0.7, 1.25], [0.8, 0.8, 1.0]]`  |
| `slice_thickness_class` | `list[float]`              | Slice thickness values / codes per exam.                                        | `[1.25, 1.0]`                          |
| `split`                 | `list[str]`                | Dataset splits for each exam.                                                   | `["train", "train"]`                   |

#### Additional Timeline-Specific Columns

| **Column Name** | **Type** | **Description**                                                                      | **Example Value** |
| --------------- | -------- | ------------------------------------------------------------------------------------ | ----------------- |
| `n_exams`       | `int`    | Number of exams (length of `exam_strs`) for this patient.                            | `2`               |
| `cancer_any`    | `int`    | Patient-level indicator: `1` if **any** exam in `cancer` list is positive, else `0`. | `1`               |

---

### Notes

* It intentionally omits registration-specific detail, which remains in `single_timepoints.parquet`.

---

## Parquet File Description

**File Name:** `paired_exam.parquet`

**Created By:** `paired_exams_parquet(...)`

**Row Granularity:** One row per **pair of consecutive exams** (`exam_a` → `exam_b`) for the same patient.

This file is the updated pairing table, now derived from the **deduplicated** `single_timepoints.parquet`. It is used for longitudinal pairwise modeling (e.g., “how did the exam change from T0 to T1?”).

---

### Relationship to `single_timepoints.parquet`

* Built by:

  * Taking `single_timepoints.parquet`,
  * Sorting per patient by `screen_timepoint`,
  * Forming **consecutive pairs** `(exam_a, exam_b)` for that patient.
* For each base column in `single_timepoints.parquet` (excluding registration-related columns), two suffixed versions are created:

  * `<column>_a` – value from the **earlier** exam in the pair.
  * `<column>_b` – value from the **later** exam in the pair.

Examples (non-exhaustive):

* `exam_id_a`, `exam_id_b`
* `exam_str_a`, `exam_str_b`
* `time_at_event_a`, `time_at_event_b`
* `days_at_event_a`, `days_at_event_b`
* `series_a`, `series_b`
* `device_a`, `device_b`
* `cancer_a`, `cancer_b`
* `pixel_spacing_a`, `pixel_spacing_b`
* `slice_thickness_class_a`, `slice_thickness_class_b`
* `split_a`, `split_b`
* etc.

The `pid` is typically kept **unsuffixed** as the patient key for the pair.

---

### Pair-Specific Columns

In addition to suffixed base columns, the pairing function adds:

| **Column Name** | **Type** | **Description**                                                                                                                         | **Example Value** |
| --------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| `pid`           | `str`    | Patient identifier for this exam pair (shared by both exams).                                                                           | `"000123"`   |
| `delta_days`    | `float`  | Difference in event days between the two exams, computed as `days_at_event_a - days_at_event_b`. Positive if exam B is closer to event. | `290.0`           |

---

### Notes

* Only **consecutive** exams are paired (e.g., T0→T1, T1→T2). There is no T0→T2 skip unless explicitly added.
* The semantics of all suffixed columns (`*_a`, `*_b`) match those in `single_timepoints.parquet`, just split by earlier/later exam.
* This file is ideal for **pairwise change** modeling, contrastive setups, or delta-based prediction tasks.

---

## Parquet description for NLST nodule truth data

**File name:** `nlst_nodule_tracking.parquet`
**Row granularity:** One row per *physical nodule in a specific exam*.

Each row corresponds to a single annotated nodule instance for a given patient and exam, with basic geometry and tracking metadata.

**Key columns:**

* `pid` *(str)* – Patient ID.
* `nodule_group` *(int)* – Index of the physical nodule for that patient (groups together the same nodule across multiple exams).
* `exam_idx` *(int)* – Time index of the exam for that nodule (e.g. 0 = earliest exam, 1 = follow-up, 2 = later follow-up).
* `exam` *(str)* – Exam ID (matches your other NLST metadata).
* `nodid_in_segmentation` *(float/int)* – Label index of this nodule in the segmentation volume for that exam.
* `volume` *(float)* – Nodule volume in mm³ (computed in the original exam’s voxel space).

**Additional geometric / tracking fields:**

* `abnid_in_nlst` *(list[int] or null)* – NLST abnormality ID(s) for this nodule.
* `coords` *(list[int] or null)* – Bounding box in original exam IJK space, as `(i_min, i_max, j_min, j_max, k_min, k_max)`.
* `center_i`, `center_j`, `center_k` *(int or null)* – Nodule center in original exam IJK space.
* `centers_past_i`, `centers_past_j`, `centers_past_k` *(int or null)* – Center of this same nodule mapped into the *previous* exam’s IJK space (when available).
* `coords_past` *(list[int] or null)* – Corresponding bounding box in the previous exam’s IJK space (when available).
* `abn_prexist` *(bool or null)* – Whether the abnormality pre-existed at baseline.
* `screen_detected` *(bool or null)* – Whether it was screen-detected.

This table gives you a clean way to:

* Get all truth nodules for a specific `(pid, exam)`.
* Track the same physical nodule across multiple exams via `nodule_group` and `exam_idx`.
* Join with your predicted results via `(pid, exam, nodid_in_segmentation)` or via spatial proximity in IJK.

## Parquet description for NLST nodule bounding boxes in fixed space

**File name:** `nodules_with_fixed_bboxes.parquet`
**Row granularity:** One row per *physical nodule in a specific exam*.

Same as `nlst_nodule_tracking.parquet`, with additional fields for the bounding boxes in the transformed space after registration to a fixed image.

**Additional fields:**
* `coords_fixed` *(list[int] or null)* – Bounding box in fixed image space, as `(i_min, i_max, j_min, j_max, k_min, k_max)`. Any bounding box that could not be created was skipped.

## Parquet description for NLST paired nodules

**File name:** `paired_nodules.parquet`
**Row granularity:** One row per *pair of physical nodules* for the same patient. Order will be earlier exam nodule first, later exam nodule second.

All the same categories as `nodules_with_fixed_bboxes.parquet` except the collumns will have the suffix _a or _b depending on if it is the earlier exam nodule or later exam nodule. Besides for pid and nodule_group which are not suffixed.
