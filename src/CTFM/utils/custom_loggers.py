# registration_logger.py

from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Mapping, Any, Sequence, Union
import os
import pandas as pd
import sys
import atexit
from datetime import datetime


PathLike = Union[str, Path]

class RegistrationLogger:
    """
    Lightweight append-only logger that writes JSON Lines (JSONL) records
    for registration / NIfTI updates.

    Each record includes a special field `_logged_cols`, which is the list
    of *non-key* columns explicitly logged in that record. This lets the
    merge logic distinguish between:
      - "this column wasn't mentioned at all" (don't touch it)
      - "this column was explicitly set to pd.NA / null" (maybe clear it)
    """

    def __init__(self, log_path: str | Path, key_cols: Sequence[str]):
        self.log_path = Path(log_path)
        self.key_cols = list(key_cols)

        # Make sure parent directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _prepare_record(self, fields: Mapping[str, Any]) -> dict:
        # Validate presence of keys
        for key in self.key_cols:
            if key not in fields:
                raise ValueError(f"Missing key column '{key}' in log_record fields")

        # Copy to avoid mutating caller's dict
        rec = dict(fields)

        # Determine which NON-KEY columns are explicitly logged
        logged_cols = [c for c in rec.keys() if c not in self.key_cols]
        rec["_logged_cols"] = logged_cols

        return rec

    def log_record(self, **fields: Any) -> None:
        """
        Append a single update record to the JSONL log.

        All key columns (self.key_cols) must be present in `fields`.
        Extra fields are allowed and will be stored. `_logged_cols`
        is automatically added.
        """
        rec = self._prepare_record(fields)
        line = json.dumps(rec, separators=(",", ":"))
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_many(self, records: Iterable[Mapping[str, Any]]) -> None:
        """
        Append multiple records efficiently.
        """
        lines: list[str] = []
        for fields in records:
            rec = self._prepare_record(fields)
            lines.append(json.dumps(rec, separators=(",", ":")))

        if not lines:
            return

        with self.log_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def merge_log_into_parquet_sequential(
    base_parquet: PathLike,
    log_path: PathLike,
    key_cols: Sequence[str],
    output_parquet: PathLike | None = None,
) -> str:
    """
    Merge updates from a JSONL log into a base Parquet file, applying each log
    entry in order (later entries overwrite earlier ones for the same key).

    The log is expected to be written by RegistrationLogger and contain a
    `_logged_cols` field listing the non-key columns explicitly set in each
    record.

    For each row in the log:
      - Only columns listed in `_logged_cols` are considered.
      - If the value is non-NA, it overwrites the base value.


    Parameters
    ----------
    base_parquet : str | Path
        Path to the existing Parquet file.
    log_path : str | Path
        Path to the JSONL log created by RegistrationLogger.
    key_cols : Sequence[str]
        Column(s) that uniquely identify a row. They must be string types
    output_parquet : str | Path | None
        Where to write the merged Parquet. If None, overwrite base_parquet.

    Returns
    -------
    str
        Path to the merged Parquet file.
    """
    base_parquet = Path(base_parquet)
    log_path = Path(log_path)
    if output_parquet is None:
        output_parquet = base_parquet
    output_parquet = Path(output_parquet)
    
    if not log_path.exists():
        print(f"Log path {log_path} does not exist; skipping merge.")
        return str(output_parquet)

    df = pd.read_parquet(base_parquet)
    updates = pd.read_json(log_path, lines=True, dtype={k: "string" for k in key_cols})
    if updates.empty:
        print("Updates log is empty; nothing to merge.")
        return str(output_parquet)

    key_cols = list(key_cols)

    # Basic validation
    for k in key_cols:
        if k not in df.columns:
            raise ValueError(f"Key column '{k}' not in base DataFrame")
        if k not in updates.columns:
            raise ValueError(f"Key column '{k}' not in log file")

    if "_logged_cols" not in updates.columns:
        raise ValueError(
            "Log file is missing '_logged_cols'. "
            "Make sure you are using the updated RegistrationLogger."
        )

    # Set index on base df for O(1) lookup by key
    df = df.set_index(key_cols, drop=False)

    # Ensure all possible update columns exist in df
    # (we'll still use _logged_cols per-row to decide which ones to touch)
    non_key_cols_in_log = [
        c for c in updates.columns if c not in key_cols and c != "_logged_cols"
    ]
    for col in non_key_cols_in_log:
        if col not in df.columns:
            df[col] = pd.NA

    # Iterate log rows in order; later rows naturally overwrite earlier ones
    for row in updates.to_dict(orient="records"):
        # Build key from dict
        key_vals = tuple(row[k] for k in key_cols)
        key = key_vals[0] if len(key_vals) == 1 else key_vals

        if key not in df.index:
            print(f"[merge] key {key!r} NOT FOUND in base parquet, skipping")
            continue

        # Get logged_cols safely from dict
        logged_cols = row.get("_logged_cols") or []
        logged_cols = set(logged_cols)
        
        for col in logged_cols:
            if col in key_cols or col == "_logged_cols":
                continue

            # Always overwrite if the column is explicitly logged,
            # even if the value is None/NaN.
            val = row.get(col, pd.NA)  # if you're iterating with to_dict(...)
            df.at[key, col] = val
            
    # Reset index and write out atomically
    df = df.reset_index(drop=True)
    tmp_path = output_parquet.with_suffix(output_parquet.suffix + ".tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, output_parquet)

    return str(output_parquet)


class Logger:
    """
    A lightweight stdout replacement that mirrors all printed output to a file
    and prepends a timestamp to each new printed line.

    This allows you to `tail -f` a log file while still seeing output normally
    in the terminal.

    Usage:
        import sys
        sys.stdout = Logger("train.log")     # log entire script output

        # or only inside a block (if __enter__/__exit__ added)
        with Logger("debug.log"):
            print("This is timestamped and logged.")

    Notes:
        - Automatically flushes after each write.
        - File is closed automatically on program exit.
    """
    def __init__(self, filename, mode="a"):
        self.terminal = sys.stdout
        self.log = open(filename, mode, buffering=1, encoding="utf-8")
        self._at_line_start = True
        self._closed = False
        atexit.register(self.close)

    def _timestamp(self) -> str:
        return datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")

    def write(self, message: str):
        if not message:
            return

        for chunk in message.splitlines(keepends=True):
            if self._at_line_start and chunk != "\n":
                chunk_to_write = self._timestamp() + chunk
            else:
                chunk_to_write = chunk

            self.terminal.write(chunk_to_write)
            self.log.write(chunk_to_write)

            self._at_line_start = chunk.endswith("\n")

        self.flush()

    def flush(self):
        # These almost never fail, but we keep them safe.
        try:
            self.terminal.flush()
        except Exception:
            pass
        try:
            self.log.flush()
        except Exception:
            pass

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.log.flush()
            self.log.close()
        except Exception:
            pass

    # Optional: so you can use it as a context manager
    def __enter__(self):
        self._old_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.stdout = self._old_stdout
        self.close()


if __name__ == "__main__":
    # Example usage
    logger = RegistrationLogger(
        "/data/rbg/users/duitz/SybilX/metadata/registration_updates.jsonl",
        key_cols=["pid", "exam_a", "exam_b"],  # or ["exam_id"]
    )

    # logger.log_record(
    #     pid=pid,
    #     exam_a=exam_a,
    #     exam_b=exam_b,
    #     registration_exists=True,
    #     registration_file=forward_tx_path,
    #     generated_nifti=gen_nifti_path,
    # )
    merged_path = merge_log_into_parquet_sequential(
        base_parquet="exam_pairs.parquet",
        log_path="registration_updates.jsonl",
        key_cols=["pid", "exam_a", "exam_b"],
    )
    print("Merged parquet saved to:", merged_path)

