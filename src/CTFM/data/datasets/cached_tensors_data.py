from pathlib import Path
from typing import Callable, Optional, Union, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
import pandas as pd
from collections import OrderedDict
import numpy as np


class _ShardCache:
    """
    Tiny LRU cache for shard tensors.

    shard_tensor.shape for:
      - 2D enc: (N_slices, C_lat, H_lat, W_lat)
      - 3D enc: (N_vols,   C_lat, Z_lat, H_lat, W_lat)
    """
    def __init__(self, root: Path, map_location: str = "cpu", max_open: int = 1000):
        self.root = root
        self.map_location = map_location
        self.max_open = max_open
        self._open: OrderedDict[str, torch.Tensor] = OrderedDict()

    def get(self, shard_name: str) -> torch.Tensor:
        if shard_name not in self._open:
            shard_path = self.root / shard_name
            self._open[shard_name] = torch.load(shard_path, map_location=self.map_location)
            if len(self._open) > self.max_open:
                self._open.popitem(last=False)
            self._open.move_to_end(shard_name)
        return self._open[shard_name]
    
def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(None if seed is None else int(seed))

class Encoded2DSliceDataset(Dataset):
    """
    2D encoded latent dataset.

    Each item is a single 2D latent slice:
      - x: (C_lat, H_lat, W_lat)
      - meta: pandas.Series with original exam metadata + slice_idx

    Args:
      full_data_parquet: main metadata parquet with at least 'exam_id' and 'num_slices'.
      encoded_index_parquet: index.parquet from encode_and_cache (2D run). 
            - must have 'exam_id', 'slice_idx', 'shard', 'offset', 'split' columns.
      encoded_root: root of encoded cache (with 'tensors/' subdir).
      split: optional split filter (matches index['split']).
      max_cache_size: maximum number of open shard tensors to cache.
      max_length: optional maximum length of the dataset.
      predicate: row -> bool filter
        (optional) filter to apply to each row of the index-parquet.
        example: predicate=lambda r: int(r.cancer) == 1
      per_exam_k: keep K random slices per exam (mutually exclusive with global_n)
            - if both per_exam_k and global_n are None, keep all slices
      global_n: keep N random slices globally
      seed: randomness for the above
    """
    def __init__(
        self,
        full_data_parquet: Union[str, Path],
        encoded_index_parquet: Union[str, Path],
        encoded_root: Union[str, Path],
        split: Optional[str] = None,
        max_cache_size: int = 1000,
        max_length: Optional[int] = None,
        predicate: Optional[Callable[[pd.Series], bool]] = None,
        per_exam_k: Optional[int] = None,
        global_n: Optional[int] = None,
        seed: Optional[int] = 42,
    ):
        self.full_df = pd.read_parquet(full_data_parquet)
        self.index_df = pd.read_parquet(encoded_index_parquet)

        # Optional split filter
        if split is not None and "split" in self.index_df.columns:
            self.index_df = self.index_df[self.index_df["split"] == split].reset_index(drop=True)

        # Map exam_id -> row index in full_df for fast lookup
        self.exam_to_idx: Dict[str, int] = {
            row["exam_id"]: idx for idx, row in self.full_df.iterrows()
        }

        self.shards = _ShardCache(Path(encoded_root) / "tensors", max_open=max_cache_size)
        self.max_length = max_length

        if predicate is not None:
            self.index_df = self.index_df[self.index_df.apply(predicate, axis=1)]

        self.index_df = self._sample_index(self.index_df.reset_index(drop=True), per_exam_k, global_n, seed)

    def _sample_index(
        self,
        idx: pd.DataFrame,
        per_exam_k: Optional[int],
        global_n: Optional[int],
        seed: Optional[int],
    ) -> pd.DataFrame:
        if sum(x is not None for x in (per_exam_k, global_n)) > 1:
            raise ValueError("Choose only one of per_exam_k or global_n.")

        if global_n is not None:
            if global_n >= len(idx):
                return idx
            r = _rng(seed)
            take = np.sort(r.choice(len(idx), size=int(global_n), replace=False))
            return idx.iloc[take].reset_index(drop=True)

        if per_exam_k is None:
            return idx

        r = _rng(seed)
        rows: List[pd.DataFrame] = []
        for exam_id, df in idx.groupby("exam_id", sort=False):
            df = df.sort_values("slice_idx")
            n = len(df)
            k = min(int(per_exam_k), n)
            keep = df.iloc[np.sort(r.choice(n, size=k, replace=False))]
            rows.append(keep)
        return pd.concat(rows, axis=0).reset_index(drop=True)
    
    def __len__(self) -> int:
        return min(len(self.index_df), self.max_length) if self.max_length is not None else len(self.index_df)

    def __getitem__(self, idx: int):
        idx_row = self.index_df.iloc[idx]
        exam_id = idx_row["exam_id"]
        slice_idx = int(idx_row["slice_idx"])
        shard_name = idx_row["shard"]
        offset = int(idx_row["offset"])

        shard_tensor = self.shards.get(shard_name)     # (N_slices, C_lat, H_lat, W_lat)
        z_i = shard_tensor[offset]                     # (C_lat, H_lat, W_lat)

        # Original metadata row
        full_row = self.full_df.iloc[self.exam_to_idx[exam_id]].copy()
        full_row["slice_idx"] = slice_idx

        return z_i, full_row

class Encoded3DFrom2DDataset(Dataset):
    """
    Build 3D latent volumes from 2D-encoded slices.

    Each item:
      - x: (C_lat, Z, H_lat, W_lat)
      - meta: pandas.Series with original exam metadata (no slice_idx)

    Args:
      full_data_parquet: main metadata parquet with 'exam_id' and 'num_slices'.
      encoded_index_parquet: 2D index.parquet (exam_id, slice_idx, shard, offset, split, ...).
      encoded_root: root of encoded cache (with 'tensors/' subdir).
      split: optional split filter.
      max_cache_size: number of shards cached in memory.
      max_length: optional max number of exams (volumes) to keep.
      predicate: optional row->bool filter applied to index_df *before* grouping.
      seed: used only if you later add random exam subsampling.
    """
    def __init__(
        self,
        full_data_parquet: Union[str, Path],
        encoded_index_parquet: Union[str, Path],
        encoded_root: Union[str, Path],
        split: Optional[str] = None,
        max_cache_size: int = 1000,
        max_length: Optional[int] = None,
        predicate: Optional[Callable[[pd.Series], bool]] = None,
        seed: Optional[int] = 42,
    ):
        self.full_df = pd.read_parquet(full_data_parquet)
        self.index_df = pd.read_parquet(encoded_index_parquet)

        # Optional split filter
        if split is not None and "split" in self.index_df.columns:
            self.index_df = self.index_df[self.index_df["split"] == split].reset_index(drop=True)

        if predicate is not None:
            self.index_df = self.index_df[self.index_df.apply(predicate, axis=1)].reset_index(drop=True)

        # Map exam_id -> row index in full_df
        self.exam_to_idx: Dict[str, int] = {
            row["exam_id"]: idx for idx, row in self.full_df.iterrows()
        }

        # We group by exam_id, sort by slice_idx
        exam_groups = []
        exam_ids = []

        for exam_id, df_e in self.index_df.groupby("exam_id", sort=False):
            if exam_id not in self.exam_to_idx:
                continue  # skip weird unmatched rows
            df_e = df_e.sort_values("slice_idx").reset_index(drop=True)
            exam_groups.append(df_e)
            exam_ids.append(exam_id)

        self.exam_groups: List[pd.DataFrame] = exam_groups
        self.exam_ids: List[str] = exam_ids

        # Optional exam-level truncation
        if max_length is not None and max_length < len(self.exam_ids):
            r = _rng(seed)
            idxs = np.sort(r.choice(len(self.exam_ids), size=int(max_length), replace=False))
            self.exam_ids = [self.exam_ids[i] for i in idxs]
            self.exam_groups = [self.exam_groups[i] for i in idxs]

        self.max_length = max_length
        self.shards = _ShardCache(Path(encoded_root) / "tensors", max_open=max_cache_size)

    def __len__(self) -> int:
        return len(self.exam_ids)

    def __getitem__(self, idx: int):
        exam_id = self.exam_ids[idx]
        df_e = self.exam_groups[idx]   # rows for this exam, sorted by slice_idx

        slices = []
        for _, row in df_e.iterrows():
            shard_name = row["shard"]
            offset = int(row["offset"])
            shard_tensor = self.shards.get(shard_name)  # (N_slices, C_lat, H_lat, W_lat)
            slice_latent = shard_tensor[offset]         # (C_lat, H_lat, W_lat)
            slices.append(slice_latent)

        # Stack into (Z, C, H, W) then permute -> (C, Z, H, W)
        vol = torch.stack(slices, dim=0)   # (Z, C, H, W)
        vol = vol.permute(1, 0, 2, 3).contiguous()

        full_row = self.full_df.iloc[self.exam_to_idx[exam_id]].copy()
        # You can optionally attach depth if you like:
        full_row["num_slices_encoded"] = vol.shape[1]

        return vol, full_row

class Encoded3DDirectDataset(Dataset):
    """
    3D encoded latent dataset (direct 3D encoding).

    Each item:
      - x: (C_lat, Z_lat, H_lat, W_lat)
      - meta: pandas.Series with original exam metadata.

    Args:
      full_data_parquet: main metadata parquet with 'exam_id', etc.
      encoded_index_parquet: index.parquet from 3D encode_and_cache run
            - must have 'exam_id', 'shard', 'offset', 'split' columns.
      encoded_root: root of encoded cache (with 'tensors/' subdir).
      split: optional split filter.
      max_cache_size: max number of open shard tensors.
      predicate: optional row->bool filter applied to index rows.
      global_n: optional random subset of volumes globally.
      seed: RNG seed for global_n.
    """
    def __init__(
        self,
        full_data_parquet: Union[str, Path],
        encoded_index_parquet: Union[str, Path],
        encoded_root: Union[str, Path],
        split: Optional[str] = None,
        max_cache_size: int = 1000,
        predicate: Optional[Callable[[pd.Series], bool]] = None,
        global_n: Optional[int] = None,
        seed: Optional[int] = 42,
    ):
        self.full_df = pd.read_parquet(full_data_parquet)
        self.index_df = pd.read_parquet(encoded_index_parquet)

        # Optional split filter
        if split is not None and "split" in self.index_df.columns:
            self.index_df = self.index_df[self.index_df["split"] == split].reset_index(drop=True)

        if predicate is not None:
            self.index_df = self.index_df[self.index_df.apply(predicate, axis=1)].reset_index(drop=True)

        # Optional global subsample of volumes
        if global_n is not None and global_n < len(self.index_df):
            r = _rng(seed)
            sel = np.sort(r.choice(len(self.index_df), size=int(global_n), replace=False))
            self.index_df = self.index_df.iloc[sel].reset_index(drop=True)

        # Map exam_id -> row index in full_df
        self.exam_to_idx: Dict[str, int] = {
            row["exam_id"]: idx for idx, row in self.full_df.iterrows()
        }

        self.shards = _ShardCache(Path(encoded_root) / "tensors", max_open=max_cache_size)
    
    def __len__(self) -> int:
        n = len(self.index_df)
        return min(n, self.max_length) if self.max_length is not None else n

    def __getitem__(self, idx: int):
        idx_row = self.index_df.iloc[idx]
        exam_id = idx_row["exam_id"]
        shard_name = idx_row["shard"]
        offset = int(idx_row["offset"])

        shard_tensor = self.shards.get(shard_name)  # (N_vols, C_lat, Z_lat, H_lat, W_lat)
        z_i = shard_tensor[offset]                  # (C_lat, Z_lat, H_lat, W_lat)

        full_row = self.full_df.iloc[self.exam_to_idx[exam_id]].copy()
        return z_i, full_row
    
class Encoded2DSlicePairsDataset(Dataset):
    """
    2D encoded latent *pair* dataset.

    Each item corresponds to a pair of exams and a specific slice index:
      - x_pair: (2 * C_lat, H_lat, W_lat)  # channels concatenated: [z_a, z_b]
      - meta: pandas.Series from the pairs_parquet (includes exam_id_a, exam_id_b, etc.)

    Args:
      full_data_parquet: main metadata parquet with 'exam_id', ...
      encoded_index_parquet: 2D index.parquet (exam_id, slice_idx, shard, offset, split, ...)
      pairs_parquet: parquet with at least 'exam_id_a' and 'exam_id_b'
      encoded_root: root of encoded cache (with 'tensors/' subdir)
      split: optional split filter applied via the 2D index (e.g., "train")
      max_cache_size: LRU cache size for shard tensors
      max_length: optional max number of (pair, slice) items (truncate)
      predicate_pairs: optional filter on the *pairs* rows (row -> bool)
      seed: if you later want to do random subsampling
      per_exam_k: keep K random slices per exam pair
      global_n: keep N random (pair, slice) items globally
        Only one of per_exam_k or global_n can be set. If both are None, keep all slices.
    """
    def __init__(
        self,
        full_data_parquet: Union[str, Path],
        encoded_index_parquet: Union[str, Path],
        pairs_parquet: Union[str, Path],
        encoded_root: Union[str, Path],
        split: Optional[str] = None,
        max_cache_size: int = 1000,
        max_length: Optional[int] = None,
        predicate_pairs: Optional[Callable[[pd.Series], bool]] = None,
        seed: Optional[int] = 42,
        per_exam_k: Optional[int] = None,
        global_n: Optional[int] = None,
    ):
        self.full_df = pd.read_parquet(full_data_parquet)
        self.index_df = pd.read_parquet(encoded_index_parquet)
        self.pairs_df = pd.read_parquet(pairs_parquet)

        # Optional split filter on the encoded index
        if split is not None and "split" in self.index_df.columns:
            self.index_df = self.index_df[self.index_df["split"] == split].reset_index(drop=True)

        # Optional filter on pairs
        if predicate_pairs is not None:
            self.pairs_df = self.pairs_df[self.pairs_df.apply(predicate_pairs, axis=1)].reset_index(drop=True)

        # Map exam_id -> row index in full_df for later metadata lookup
        self.exam_to_idx: Dict[str, int] = {
            row["exam_id"]: idx for idx, row in self.full_df.iterrows()
        }

        # Only keep pairs where both exams have encoded slices AND exist in full_df
        encoded_exams = set(self.index_df["exam_id"].unique())
        valid_pairs_mask = (
            self.pairs_df["exam_id_a"].isin(encoded_exams)
            & self.pairs_df["exam_id_b"].isin(encoded_exams)
            & self.pairs_df["exam_id_a"].isin(self.exam_to_idx.keys())
            & self.pairs_df["exam_id_b"].isin(self.exam_to_idx.keys())
        )
        self.pairs_df = self.pairs_df[valid_pairs_mask].reset_index(drop=True)

        # Group 2D index by exam, and sort by slice_idx
        self.exam_to_slices: Dict[str, pd.DataFrame] = {}
        for exam_id, df_e in self.index_df.groupby("exam_id", sort=False):
            df_e = df_e.sort_values("slice_idx").reset_index(drop=True)
            self.exam_to_slices[exam_id] = df_e

        # Build a flat list of (pair_idx, rel_slice_idx) for the dataset
        assert not (per_exam_k is not None and global_n is not None), "Choose only one of per_exam_k or global_n."
        r = _rng(seed)
        items: List[Tuple[int, int]] = []
        for pair_idx, pair_row in self.pairs_df.iterrows():
            a = pair_row["exam_id_a"]
            b = pair_row["exam_id_b"]
            if a not in self.exam_to_slices or b not in self.exam_to_slices:
                continue
            df_a = self.exam_to_slices[a]
            df_b = self.exam_to_slices[b]
            assert len(df_a) == len(df_b), f"Exam pair {a}, {b} have different slice counts!"
            n = min(len(df_a), len(df_b))
            if per_exam_k is not None:       
                take = np.sort(r.choice(n, size=int(per_exam_k), replace=False))
                for s in take:
                    items.append((pair_idx, s))
            else:
                for s in range(n):
                    items.append((pair_idx, s))
        if global_n is not None and global_n < len(items):
            take = np.sort(r.choice(len(items), size=int(global_n), replace=False))
            items = [items[i] for i in take]
        # Optional truncation
        if max_length is not None and max_length < len(items):
            items = items[:max_length]

        self.items = items
        self.shards = _ShardCache(Path(encoded_root) / "tensors", max_open=max_cache_size)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        pair_idx, rel_slice_idx = self.items[idx]
        pair_row = self.pairs_df.iloc[pair_idx]

        exam_id_a = pair_row["exam_id_a"]
        exam_id_b = pair_row["exam_id_b"]

        df_a = self.exam_to_slices[exam_id_a]
        df_b = self.exam_to_slices[exam_id_b]

        row_a = df_a.iloc[rel_slice_idx]
        row_b = df_b.iloc[rel_slice_idx]

        shard_a = row_a["shard"]
        off_a = int(row_a["offset"])
        shard_b = row_b["shard"]
        off_b = int(row_b["offset"])

        tensor_a = self.shards.get(shard_a)[off_a]  # (C_lat, H_lat, W_lat)
        tensor_b = self.shards.get(shard_b)[off_b]  # (C_lat, H_lat, W_lat)

        # Concatenate along channel dimension: (2*C_lat, H, W)
        x_pair = torch.cat([tensor_a, tensor_b], dim=0)

        # Return pair metadata row (contains both exam_ids, and anything else in pairs_parquet)
        return x_pair, pair_row


class Encoded3DPairsFrom2DDataset(Dataset):
    """
    3D latent *pair* dataset built from 2D-encoded slices.

    Each item:
      - x_pair: (2 * C_lat, Z, H_lat, W_lat)
      - meta: pandas.Series from the pairs_parquet

    Args:
      full_data_parquet: main metadata with 'exam_id' etc.
      encoded_index_parquet: 2D index.parquet (exam_id, slice_idx, shard, offset, ...)
      pairs_parquet: parquet with 'exam_id_a', 'exam_id_b'
      encoded_root: root of encoded cache
      split: optional split filter based on index_df["split"]
      max_cache_size: for shard tensors
      max_length: optional max number of pairs
      predicate_pairs: filter on pairs rows
    """
    def __init__(
        self,
        full_data_parquet: Union[str, Path],
        encoded_index_parquet: Union[str, Path],
        pairs_parquet: Union[str, Path],
        encoded_root: Union[str, Path],
        split: Optional[str] = None,
        max_cache_size: int = 1000,
        max_length: Optional[int] = None,
        predicate_pairs: Optional[Callable[[pd.Series], bool]] = None,
    ):
        self.full_df = pd.read_parquet(full_data_parquet)
        self.index_df = pd.read_parquet(encoded_index_parquet)
        self.pairs_df = pd.read_parquet(pairs_parquet)

        if split is not None and "split" in self.index_df.columns:
            self.index_df = self.index_df[self.index_df["split"] == split].reset_index(drop=True)

        if predicate_pairs is not None:
            self.pairs_df = self.pairs_df[self.pairs_df.apply(predicate_pairs, axis=1)].reset_index(drop=True)

        self.exam_to_idx: Dict[str, int] = {
            row["exam_id"]: idx for idx, row in self.full_df.iterrows()
        }

        encoded_exams = set(self.index_df["exam_id"].unique())
        valid_pairs_mask = (
            self.pairs_df["exam_id_a"].isin(encoded_exams)
            & self.pairs_df["exam_id_b"].isin(encoded_exams)
            & self.pairs_df["exam_id_a"].isin(self.exam_to_idx.keys())
            & self.pairs_df["exam_id_b"].isin(self.exam_to_idx.keys())
        )
        self.pairs_df = self.pairs_df[valid_pairs_mask].reset_index(drop=True)

        # Group 2D index by exam, sorted by slice_idx
        self.exam_to_slices: Dict[str, pd.DataFrame] = {}
        for exam_id, df_e in self.index_df.groupby("exam_id", sort=False):
            df_e = df_e.sort_values("slice_idx").reset_index(drop=True)
            self.exam_to_slices[exam_id] = df_e

        # Optionally truncate the number of pairs
        if max_length is not None and max_length < len(self.pairs_df):
            self.pairs_df = self.pairs_df.iloc[:max_length].reset_index(drop=True)

        self.shards = _ShardCache(Path(encoded_root) / "tensors", max_open=max_cache_size)

    def __len__(self) -> int:
        return len(self.pairs_df)

    def _build_volume(self, exam_id: str) -> torch.Tensor:
        df_e = self.exam_to_slices[exam_id]
        slices = []
        for _, row in df_e.iterrows():
            shard_name = row["shard"]
            offset = int(row["offset"])
            shard_tensor = self.shards.get(shard_name)  # (N_slices, C_lat, H_lat, W_lat)
            slices.append(shard_tensor[offset])         # (C_lat, H_lat, W_lat)

        vol = torch.stack(slices, dim=0)  # (Z, C, H, W)
        vol = vol.permute(1, 0, 2, 3).contiguous()  # (C, Z, H, W)
        return vol

    def __getitem__(self, idx: int):
        pair_row = self.pairs_df.iloc[idx]
        exam_id_a = pair_row["exam_id_a"]
        exam_id_b = pair_row["exam_id_b"]

        vol_a = self._build_volume(exam_id_a)  # (C, Z, H, W)
        vol_b = self._build_volume(exam_id_b)  # (C, Z, H, W) (assuming matching depth after your pipeline)

        x_pair = torch.cat([vol_a, vol_b], dim=0)  # (2*C, Z, H, W)
        return x_pair, pair_row


class Encoded3DPairsDirectDataset(Dataset):
    """
    3D encoded latent *pair* dataset (direct 3D encoding).

    Each item:
      - x_pair: (2 * C_lat, Z_lat, H_lat, W_lat)
      - meta: pandas.Series from pairs_parquet

    Args:
      full_data_parquet: main metadata with 'exam_id', ...
      encoded_index_parquet: 3D index.parquet (exam_id, shard, offset, split, ...)
      pairs_parquet: parquet with 'exam_id_a', 'exam_id_b'
      encoded_root: root of encoded cache
      split: optional split filter on index["split"]
      max_cache_size: LRU size
      max_length: optional max number of pairs
      predicate_pairs: filter on pair rows
    """
    def __init__(
        self,
        full_data_parquet: Union[str, Path],
        encoded_index_parquet: Union[str, Path],
        pairs_parquet: Union[str, Path],
        encoded_root: Union[str, Path],
        split: Optional[str] = None,
        max_cache_size: int = 1000,
        max_length: Optional[int] = None,
        predicate_pairs: Optional[Callable[[pd.Series], bool]] = None,
    ):
        self.full_df = pd.read_parquet(full_data_parquet)
        self.index_df = pd.read_parquet(encoded_index_parquet)
        self.pairs_df = pd.read_parquet(pairs_parquet)

        if split is not None and "split" in self.index_df.columns:
            self.index_df = self.index_df[self.index_df["split"] == split].reset_index(drop=True)

        if predicate_pairs is not None:
            self.pairs_df = self.pairs_df[self.pairs_df.apply(predicate_pairs, axis=1)].reset_index(drop=True)

        # exam_id -> row in full_df
        self.exam_to_idx: Dict[str, int] = {
            row["exam_id"]: idx for idx, row in self.full_df.iterrows()
        }

        # exam_id -> row in index_df (one volume per exam)
        # If for some reason there are duplicates, we just keep the first.
        self.exam_to_index_row: Dict[str, pd.Series] = {}
        for _, row in self.index_df.iterrows():
            ex = row["exam_id"]
            if ex not in self.exam_to_index_row:
                self.exam_to_index_row[ex] = row

        encoded_exams = set(self.exam_to_index_row.keys())
        valid_pairs_mask = (
            self.pairs_df["exam_id_a"].isin(encoded_exams)
            & self.pairs_df["exam_id_b"].isin(encoded_exams)
            & self.pairs_df["exam_id_a"].isin(self.exam_to_idx.keys())
            & self.pairs_df["exam_id_b"].isin(self.exam_to_idx.keys())
        )
        self.pairs_df = self.pairs_df[valid_pairs_mask].reset_index(drop=True)

        if max_length is not None and max_length < len(self.pairs_df):
            self.pairs_df = self.pairs_df.iloc[:max_length].reset_index(drop=True)

        self.shards = _ShardCache(Path(encoded_root) / "tensors", max_open=max_cache_size)

    def __len__(self) -> int:
        return len(self.pairs_df)

    def _load_vol(self, exam_id: str) -> torch.Tensor:
        row = self.exam_to_index_row[exam_id]
        shard_name = row["shard"]
        offset = int(row["offset"])
        shard_tensor = self.shards.get(shard_name)  # (N_vols, C_lat, Z_lat, H_lat, W_lat)
        vol = shard_tensor[offset]                  # (C_lat, Z_lat, H_lat, W_lat)
        return vol

    def __getitem__(self, idx: int):
        pair_row = self.pairs_df.iloc[idx]
        exam_id_a = pair_row["exam_id_a"]
        exam_id_b = pair_row["exam_id_b"]

        vol_a = self._load_vol(exam_id_a)
        vol_b = self._load_vol(exam_id_b)

        x_pair = torch.cat([vol_a, vol_b], dim=0)  # (2*C_lat, Z_lat, H_lat, W_lat)
        return x_pair, pair_row
    

class CachedNoduleDataset(Dataset):
    """
    3D Nodule Cached Dataset.

    Each item:
      - x: (C_lat, Z_lat, H_lat, W_lat)
      - meta: Dictionary with exam and nodule metadata.

    Args:
      full_data_parquet: main metadata parquet with 'exam_id', etc.
      full_nodule_parquet: nodule metadata parquet with 'nodule_group', 'exam', 'exam_idx', etc.
      data_index_parquet: index.parquet with 'nodule_key', 'exam', 'shard', 'offset', 'split', etc.
      data_root: root of encoded cache (with 'tensors/' subdir).
      paired_nodule_parquet: optional paired nodule metadata parquet (for 'paired' mode).
      mode: "single" or "paired" (if paired, uses paired_nodule_parquet).
      split: optional split filter on data_index_parquet.
      max_cache_size: LRU cache size for shard tensors.
      max_length: optional max number of nodules.
      return_meta_data: if True, return metadata dictionary along with tensor.
    """
    def __init__(
        self,
        full_data_parquet: Union[str, Path],
        full_nodule_parquet: Union[str, Path],
        data_index_parquet: Union[str, Path],
        data_root: Union[str, Path],
        paired_nodule_parquet: Union[str, Path] = None,
        mode: str = "single",
        split: Optional[str] = None,
        max_cache_size: int = 100,
        max_length: Optional[int] = None,
        return_meta_data: bool = False,
    ):
        self.full_df = pd.read_parquet(full_data_parquet)
        self.index_df = pd.read_parquet(data_index_parquet)
        self.full_nodule_df = pd.read_parquet(full_nodule_parquet)

        # Optional split filter
        if split is not None and "split" in self.index_df.columns:
            self.index_df = self.index_df[self.index_df["split"] == split].reset_index(drop=True)

        # Map exam_id -> row index in full_df
        self.exam_to_idx: Dict[str, int] = {
            row["exam_id"]: idx for idx, row in self.full_df.iterrows()
        }
        self.nodule_key_to_idx = {f"{row['nodule_group']}_{row['exam']}_{row['exam_idx']}": idx for idx, row in self.full_nodule_df.iterrows()}

        self.mode = mode
        if self.mode == "paired":
            if paired_nodule_parquet is None:
                raise ValueError("paired_nodule_parquet must be provided in 'paired' mode.")
            self.paired_nodule_df = pd.read_parquet(paired_nodule_parquet)

            self.nodule_key_to_cached_index: Dict[str, pd.Series] = {}
            for _, row in self.index_df.iterrows():
                ex = row["nodule_key"]
                self.nodule_key_to_cached_index[ex] = row

            encoded_nodules = set(self.nodule_key_to_cached_index.keys())
            nodule_key_a = (
                self.paired_nodule_df["nodule_group"].astype(str)
                + "_"
                + self.paired_nodule_df["exam_a"].astype(str)
                + "_"
                + self.paired_nodule_df["exam_idx_a"].astype(str)
            )

            nodule_key_b = (
                self.paired_nodule_df["nodule_group"].astype(str)
                + "_"
                + self.paired_nodule_df["exam_b"].astype(str)
                + "_"
                + self.paired_nodule_df["exam_idx_b"].astype(str)
            )
            valid_pairs_mask = (
                nodule_key_a.isin(encoded_nodules)
                & nodule_key_b.isin(encoded_nodules)
            )
            self.paired_nodule_df = self.paired_nodule_df[valid_pairs_mask].reset_index(drop=True)


        self.shards = _ShardCache(Path(data_root) / "tensors", max_open=max_cache_size)
        self.max_length = max_length
        self.return_meta_data = return_meta_data

    def __len__(self) -> int:
        if self.mode == "single":
            n = len(self.index_df)
        else: # paired
            n = len(self.paired_nodule_df)
        return min(n, self.max_length) if self.max_length is not None else n

    def __getitem__(self, idx: int):
        if self.mode == "single":
            idx_row = self.index_df.iloc[idx]
            nodule_key = idx_row["nodule_key"]
            exam_id = idx_row["exam"]
            shard_name = idx_row["shard"]
            offset = int(idx_row["offset"])

            shard_tensor = self.shards.get(shard_name)  # (N_vols, C_lat, Z_lat, H_lat, W_lat)
            z_i = shard_tensor[offset]                  # (C_lat, Z_lat, H_lat, W_lat)

            full_row = self.full_df.iloc[self.exam_to_idx[exam_id]].copy()
            full_nodule_row = self.full_nodule_df.iloc[self.nodule_key_to_idx[nodule_key]].copy()
            meta = full_row.to_dict() | full_nodule_row.to_dict()
            if self.return_meta_data:
                return z_i, meta
            return z_i
        else:  # paired
            pair_row = self.paired_nodule_df.iloc[idx]
            nodule_key_a = f"{pair_row['nodule_group']}_{pair_row['exam_a']}_{pair_row['exam_idx_a']}"
            nodule_key_b = f"{pair_row['nodule_group']}_{pair_row['exam_b']}_{pair_row['exam_idx_b']}"

            row_a = self.nodule_key_to_cached_index[nodule_key_a]
            row_b = self.nodule_key_to_cached_index[nodule_key_b]

            shard_name_a = row_a["shard"]
            offset_a = int(row_a["offset"])
            shard_name_b = row_b["shard"]
            offset_b = int(row_b["offset"])

            shard_tensor_a = self.shards.get(shard_name_a)
            z_i_a = shard_tensor_a[offset_a]
            shard_tensor_b = self.shards.get(shard_name_b)
            z_i_b = shard_tensor_b[offset_b]

            # Concatenate along channel dimension
            z_i = torch.cat([z_i_a, z_i_b], dim=0)

            full_row_a = self.full_df.iloc[self.exam_to_idx[pair_row["exam_a"]]].copy()
            full_nodule_row_a = self.full_nodule_df.iloc[self.nodule_key_to_idx[nodule_key_a]].copy()
            full_row_b = self.full_df.iloc[self.exam_to_idx[pair_row["exam_b"]]].copy()
            full_nodule_row_b = self.full_nodule_df.iloc[self.nodule_key_to_idx[nodule_key_b]].copy()

            if self.return_meta_data:
                meta = {
                                **{f"{k}_a": v for k, v in full_row_a.items()},
                                **{f"{k}_b": v for k, v in full_row_b.items()},
                                **{f"{k}_a": v for k, v in full_nodule_row_a.items()},
                                **{f"{k}_b": v for k, v in full_nodule_row_b.items()},
                            }
                 
                # meta = (
                #     full_row_a.to_dict() | full_nodule_row_a.to_dict() |
                #     full_nodule_row_b.to_dict()
                # )
                return z_i, meta
            else:
                return z_i
