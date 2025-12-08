# General imports
from pathlib import Path
import os
import pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import json

# Module imports
from CTFM.utils.config import load_config
from .datasets.nlst_base import NLST_Survival_Dataset
from CTFM.data.utils import fix_repeated_shared, get_exam_id


def create_full_data_parquet(nlst_dataset, 
                             exam_to_nifti, 
                             output_path: str, 
                             split: str, 
                             bad_exams: list = ["207121029408161841061", "11251907744137342542"]):
    """
    Create a full metadata parquet file for the NLST dataset that is passed in.
    The dataset should be a list with each entry being a dictionary containing the
    relevant information for each exam.
    Additionally the function will check if a nifti file exists for each exam
    based on a pre-defined mapping (exam_to_nifti). If a nifti file exists, it will
    store the path to the nifti file and set has_nifti to True. If not, it will
    store the sorted paths to the DICOM files and set has_nifti to False.

    For parquets the sorted_paths will be stored as a JSON string. So load it with json.loads(row["sorted_paths"]).

    output_path: str
        The path where the parquet file will be saved.
    exam_to_nifti: dict
        A dictionary mapping exam IDs to their corresponding nifti file paths.
    bad_exams: list
        A list of exam strings that should be excluded from the parquet file because of known issues.
    Returns:
        output_path
    """

    new_parquet = []

    for index, sample in enumerate(nlst_dataset):
        new_sample = {}

        exam_id = get_exam_id(sample)
        if exam_id in exam_to_nifti.keys() and exam_id not in bad_exams:
            new_sample["nifti_path"] = exam_to_nifti[exam_id]
            new_sample["has_nifti"] = True
            new_sample["sorted_paths"] = None
        else:
            sorted_paths = [fix_repeated_shared(p) for p in sample["paths"]]
            new_sample["sorted_paths"] = json.dumps(sorted_paths)
            new_sample["nifti_path"] = None
            new_sample["has_nifti"] = False
        
        # ask chat about storing them and what their type should be for lists and stuff
        new_sample["cancer"] = int(sample["y"])
        new_sample["exam_id"] = str(exam_id)
        new_sample["exam"] = str(sample["exam"])
        new_sample["num_slices"] = int(sample["num_original_slices"])
        new_sample["time_at_event"] = float(sample["time_at_event"])
        new_sample["days_at_event"] = float(sample["days_at_event"])
        new_sample["exam_str"] = str(sample["exam_str"])
        new_sample["pid"] = str(sample["pid"])
        new_sample["series"] = sample["series"]
        new_sample["y_seq"] = np.asarray(sample["y_seq"], dtype=np.float32).tolist()
        new_sample["y_mask"] = np.asarray(sample["y_mask"], dtype=np.float32).tolist()
        new_sample["pixel_spacing"] = np.asarray(sample["pixel_spacing"], dtype=np.float32).tolist()
        new_sample["accession"] = str(sample["accession"])
        new_sample["screen_timepoint"] = int(sample["screen_timepoint"])
        new_sample["slice_thickness_class"] = float(sample["slice_thickness"])
        new_sample["split"] = str(split)

        if sample["exam"] in bad_exams:
            continue

        new_parquet.append(new_sample)

    df = pd.DataFrame(new_parquet)
    dtype_map = {
                    "cancer": "int8",
                    "exam_id": "string",
                    "exam": "string",
                    "num_slices": "int16",
                    "time_at_event": "float32",
                    "days_at_event": "float32",
                    "exam_str": "string",
                    "pid": "string",
                    "series": "string",
                    "accession": "string",
                    "screen_timepoint": "int8",
                    "slice_thickness_class": "float32",
                    "split": "string",
                }
    
    df = df.astype(dtype_map)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)
    return output_path

def dedup_timepoints(parquet_path: str, new_file_name: str = "single_timepoints.parquet") -> str:
    """
    This function will only leave one scan for each timepoint and create a new parquet file
    It will choose the minimum slice thickness and if there are multiple then it will choose randomly
    """
    df = pd.read_parquet(parquet_path).copy()

    def _unwrap(x):
        if isinstance(x, (list, np.ndarray)):
            return x[0] if len(x) == 1 else tuple(x)
        return x

    df["pid"] = df["pid"].apply(_unwrap).astype(str)
    df["series"] = df["series"].apply(_unwrap).astype(str)

    # ensure numeric for comparisons (minimal safety)
    df["slice_thickness_class"] = pd.to_numeric(df["slice_thickness_class"], errors="coerce")

    def pick_one(g: pd.DataFrame) -> pd.Series:
        mn = g["slice_thickness_class"].min()
        cand = g[g["slice_thickness_class"] == mn]
        return cand.sample(n=1).iloc[0]  # random if multiple with same min

    out = (
        df.groupby(["pid", "screen_timepoint"], group_keys=False)
          .apply(pick_one)
          .reset_index(drop=True)
    )

    # same directory, new filename
    dirpath = os.path.dirname(parquet_path)
    new_path = os.path.join(dirpath, new_file_name)
    out.to_parquet(new_path, index=False)
    return new_path

def write_patient_timelines(in_parquet: str, out_parquet: str = "patient_timelines.parquet") -> str:
    """
    Write a parquet file with each patient on one row and a list in order of their exams

    Within each patient row, each column (except pid, n_exams and cancer_any) is a list of values for all the exams
    for that patient, ordered by screen_timepoint ascending.
    """
    df = pd.read_parquet(in_parquet).copy()

    collapsed_rows = []

    # All columns except grouping key
    value_cols = [c for c in df.columns if c != "pid"]

    for pid, g in df.groupby("pid", sort=False):
        g = g.sort_values("screen_timepoint", ascending=True)

        row = {"pid": pid}

        # collapse each column into a list
        for col in value_cols:
            row[col] = g[col].tolist()

        # optionally add a derived field
        row["n_exams"] = len(g)

        row["cancer_any"] = int(any(g["cancer"] == 1))

        collapsed_rows.append(row)

    
    out = pd.DataFrame(collapsed_rows)
    # same directory, new filename
    dirpath = os.path.dirname(in_parquet)
    new_path = os.path.join(dirpath, out_parquet)
    out.to_parquet(new_path, index=False)
    return new_path

def paired_exams_parquet(full_data_parquet, output_name: str = "paired_exams.parquet") -> str:
    """
    This function takes in an original deduped full data parquet file and creates a new parquet file
    where each row corresponds to a pair of exams for the same patient. The exams are paired
    based on their screen_timepoint, with each exam being paired with the next exam in time.
    The resulting parquet file contains columns for each exam in the pair, with suffixes '_a' and '_b'
    to indicate the first and second exam in the pair, respectively. Additionally, a new column 'delta_days'
    is added to indicate the difference in days between the two exams.

    """
    df = pd.read_parquet(full_data_parquet)

    pairs_rows = []

    base_cols = [c for c in df.columns if c not in ["pid", "n_exams"]]

    for pid, g in df.groupby("pid", sort=False):
        g = g.sort_values("screen_timepoint", ascending=True)

        if len(g) < 2:
            continue  # no pairs for this patient

        for i in range(len(g) - 1):
            row_a = g.iloc[i]
            row_b = g.iloc[i + 1]

            pair = {"pid": pid}

            for col in base_cols:
                pair[f"{col}_a"] = row_a[col]
                pair[f"{col}_b"] = row_b[col]

            pair["delta_days"] = row_a["days_at_event"] - row_b["days_at_event"]

            pairs_rows.append(pair)

    dirpath = os.path.dirname(full_data_parquet)
    new_path = os.path.join(dirpath, output_name)
    out = pd.DataFrame(pairs_rows)
    out.to_parquet(new_path, index=False)
    return new_path

def full_parquet_creation(nlst_data,
                          exam_to_nifti,
                          split_group,
                          metadata_path,
                          debug: bool = False,):
    """
    This function will create the full data parquet, single timepoint parquet,
    paired exams parquet, and patient timelines parquet for the given NLST dataset
    and exam to nifti mapping.

    Parameters:
    nlst_data: list
        The NLST dataset as a list of samples.
    exam_to_nifti: dict
        A mapping from exam IDs to nifti file paths.
    split_group: str
        The split group (e.g., 'train', 'val', 'test') for the dataset.
    debug: bool
        If True, print debug information during the process.
    """
    if debug:
        print("Running in debug mode.")
        print("Starting Creation of full data parquet.")
    full_data_parquet = create_full_data_parquet(
        nlst_data,
        exam_to_nifti=exam_to_nifti,
        output_path=str(metadata_path / f"{split_group}" / f"nlst_full.parquet"),
        split=split_group,
    )
    if debug:
        print("Finished Creation of full data parquet.")
        print("Starting Creation of single timepoint parquet.")
    single_timepoint_parquet = dedup_timepoints(
        full_data_parquet,
        new_file_name="full_data_single_timepoints.parquet"
    )
    if debug:
        print("Finished Creation of single timepoint parquet.")
        print("Starting Creation of paired exams parquet.")
    paired_exams_file = paired_exams_parquet(single_timepoint_parquet, output_name="full_data_paired_exams.parquet")
    if debug:
        print("Finished Creation of paired exams parquet.")
        print("Starting Creation of patient timelines parquet.")
    patient_timelines = write_patient_timelines(single_timepoint_parquet, out_parquet="full_data_timelines.parquet")
    if debug:
        print("Finished Creation of patient timelines parquet.")

    print("Parquet files created:")
    print(f" - {full_data_parquet}")
    print(f" - {single_timepoint_parquet}")
    print(f" - {patient_timelines}")
    print(f" - {paired_exams_file}")
    print("Done.")


if __name__ == "__main__":
    #### Variables ####
    dataset_config_yaml = "configs/nlst_large.yaml"
    path_yaml = "configs/paths.yaml"
    split_group = "train"
    
    #### Load Configs ####
    args = load_config(dataset_config_yaml)
    base_paths = load_config(path_yaml)

    metadata_path = Path(base_paths.metadata_dir)  # / "nlst_large_metadata.csv"

    ## Load Dataset 
    nlst_raw = NLST_Survival_Dataset(args, split_group=split_group)
    nlst_data = nlst_raw.dataset

    ## create initial parquet with all data
    exam_to_nifti_file = metadata_path / "zack_exam_to_nifti.pkl"
    with open(exam_to_nifti_file, "rb") as f:
        exam_to_nifti = pickle.load(f)

    full_parquet_creation(
        nlst_data,
        exam_to_nifti,
        split_group,
        metadata_path=metadata_path,
        debug=True,
    )
    