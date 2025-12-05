
Here’s a compact writeup plus some ideas on matching.

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

