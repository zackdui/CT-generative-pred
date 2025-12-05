import pickle
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd


def pickle_to_parquet(p_path: str, out_parquet: str):
    p_path = Path(p_path)     # your .p file
    out_parquet = Path(out_parquet)
    with p_path.open("rb") as f:
        raw = pickle.load(f)

        # If it's actually a dict {pid: nodules_dict}, convert:
        if isinstance(raw, dict):
            data = list(raw.items())
        else:
            data = raw

    rows: List[Dict[str, Any]] = []

    for pid, nodules in data:
        # nodules can be {} (no annotations)
        if not nodules:
            continue

        for nodule_group, exams_dict in nodules.items():
            # exams_dict: {exam_idx: lesion_info}
            for exam_idx, lesion in exams_dict.items():
                center = lesion.get("center")
                if center is not None:
                    center_i, center_j, center_k = center
                else:
                    center_i = center_j = center_k = None

                centers_past = lesion.get("centers_in_past_exam_ijk_space")
                if centers_past is not None:
                    cpi, cpj, cpk = centers_past
                else:
                    cpi = cpj = cpk = None

                coords = lesion.get("coords")
                coords_past = lesion.get("coords_in_past_exam_ijk_space")

                row = {
                    "pid": pid,
                    "nodule_group": int(nodule_group),
                    "exam_idx": int(exam_idx),
                    "exam": lesion.get("exam"),
                    "nodid_in_segmentation": lesion.get("nodid_in_segmentation"),
                    "volume": lesion.get("volume"),
                    "abnid_in_nlst": lesion.get("abnid_in_nlst"),
                    "coords": list(coords) if coords is not None else None,
                    "center_i": center_i,
                    "center_j": center_j,
                    "center_k": center_k,
                    "centers_past_i": cpi,
                    "centers_past_j": cpj,
                    "centers_past_k": cpk,
                    "coords_past": list(coords_past) if coords_past is not None else None,
                    "abn_prexist": lesion.get("abn_prexist"),
                    "screen_detected": lesion.get("screen_detected"),
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(out_parquet, index=False)
    print(df.head())
    print("Saved:", out_parquet)
