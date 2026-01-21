import os
from pathlib import Path
from typing import Dict, Optional, Union
import torch.distributed as dist
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from .utils import collate_image_meta
from .datasets.CT_orig_data import CTOrigDataset2D, CTOrigDataset3D, CTNoduleDataset3D
from .datasets.cached_tensors_data import CachedNoduleDataset

def compute_completed_2d_exams(
    dataset_parquet: Union[str, Path],
    meta_dir: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
) -> set:
    """
    Compute which exam_ids are fully encoded in 2D.

    Uses:
      - dataset_parquet: original metadata with exam_id + num_slices
      - meta/index_shard_*.parquet: per-slice encoded info

    Returns:
      A set of exam_ids that have encoded_slices >= num_slices.
    Optionally writes them to out_path as a parquet with a single 'exam_id' column.
    """
    dataset_parquet = Path(dataset_parquet)
    meta_dir = Path(meta_dir)
    shard_paths = sorted(meta_dir.glob("index_shard_*.parquet"))
    shards_dir = meta_dir / "shards"
    if shards_dir.exists():
        shard_paths += sorted(shards_dir.glob("index_shard_*.parquet"))

    if not shard_paths:
        return set()

    # Original exam metadata
    df_all = pd.read_parquet(dataset_parquet)[["exam_id", "num_slices"]]

    # All encoded slices from index shards
    dfs = [pd.read_parquet(p) for p in shard_paths]
    idx_df = pd.concat(dfs, ignore_index=True)

    # Only 2D shards will have 'slice_idx'
    if "slice_idx" not in idx_df.columns:
        return set()

    # Count unique slice_idx per exam
    enc_counts = (
        idx_df.groupby("exam_id")["slice_idx"]
        .nunique()
        .reset_index(name="encoded_slices")
    )

    merged = df_all.merge(enc_counts, on="exam_id", how="left")
    merged["encoded_slices"] = merged["encoded_slices"].fillna(0).astype(int)

    completed = merged[merged["encoded_slices"] >= merged["num_slices"]]
    completed_ids = set(completed["exam_id"].tolist())

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        completed[["exam_id"]].to_parquet(out_path, index=False)

    return completed_ids

def encode_and_cache(
    dataset_parquet: Union[str, Path],
    encoder: torch.nn.Module,
    saved_transforms: Optional[Dict[str, str]] = None,
    out_root: Union[str, Path] = "/data/rbg/scratch/nlst_final_encoded",
    split: str = "train",
    is_3d: bool = False,
    batch_size: int = 8,
    num_workers: int = 4,
    shard_size: int = 200,
    device: str = "cuda",
    rank: int = 0,        # GPU / process id
):
    """
    Encode a dataset with an autoencoder and cache latents to disk.

    Multi-GPU pattern:
      - Create K Subset datasets (or index splits)
      - For process k: pass rank=k and dataset_k
      - All processes share the same out_root

    Requirements on dataset:
      - __getitem__ returns: image, meta_dict
      - meta_dict["exam_id"] is present
      - if not is_3d: meta_dict["slice_idx"] is present

    encoder: It should be a torch.nn.Module with an .encode() method that
      takes in a batch of images and returns the latent representation:
      - 2D: (B, C_in, H, W) -> (B, C_lat, H_lat, W_lat)
      - 3D: (B, C_in, Z, Y, X) -> (B, C_lat, Z_lat, Y_lat, X_lat)
      If the encoder returns a tuple/list, the first element will be used.
    """
    out_root = Path(out_root)
    tensors_dir = out_root / "tensors"
    meta_dir = out_root / "meta"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    def _infer_start_local_shard_idx() -> int:
        """
        Look for existing shard files for this rank and continue counting
        from the largest existing index + 1.

        Filenames: shard_r{rank:02d}_{idx:04d}.pt
        """
        pattern = f"shard_r{rank:02d}_*.pt"
        existing = list(tensors_dir.glob(pattern))
        if not existing:
            return 0

        max_idx = -1
        for p in existing:
            # stem example: "shard_r00_0003"
            stem = p.stem
            parts = stem.split("_")
            if len(parts) == 3:
                try:
                    idx = int(parts[2])
                    if idx > max_idx:
                        max_idx = idx
                except ValueError:
                    continue
        return max_idx + 1 if max_idx >= 0 else 0

    # 1) Discover existing indices (for resume)
    def _load_existing_keys():
        index_files = sorted(meta_dir.glob("index_shard_*.parquet"))
        shards_dir = meta_dir / "shards"
        if shards_dir.exists():
            index_files += sorted(shards_dir.glob("index_shard_*.parquet"))
        
        if not index_files:
            return set()

        dfs = [pd.read_parquet(p) for p in index_files]
        df = pd.concat(dfs, ignore_index=True)

        if is_3d:
            # One row per volume; key is exam_id
            processed = set(zip(df["exam_id"]))
        else:
            # One row per slice; key is (exam_id, slice_idx)
            processed = set(zip(df["exam_id"], df["slice_idx"]))

        return processed

    processed_keys = _load_existing_keys()
    print(f"[encode_and_cache r{rank}] Resuming with {len(processed_keys)} already-encoded keys.")

    
    # 2) Dataset and DataLoader + encoder setup
    # Optional: split work across ranks when launched with torchrun
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_rank = int(os.environ.get("RANK", str(rank)))

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # rewrite the parquet removing already processed exams so we don't load them again from scratch
    df = pd.read_parquet(dataset_parquet)
    if is_3d:
        # Processed keys are either (exam_id,) for 3D or (exam_id, slice_idx) for 2D
        processed_keys_exams = {k[0] for k in processed_keys}
        df_updated = df[~df['exam_id'].isin(processed_keys_exams)].reset_index(drop=True)
    else:
        completed_path = meta_dir / "completed_2d_exams.parquet"
        if world_size > 1:
            if global_rank == 0:
                compute_completed_2d_exams(
                    dataset_parquet=dataset_parquet,
                    meta_dir=meta_dir,
                    out_path=completed_path,
                )
            dist.barrier()  # wait for rank 0 to finish

            if completed_path.exists():
                completed_df = pd.read_parquet(completed_path)
                completed_ids = set(completed_df["exam_id"].tolist())
            else:
                completed_ids = set()
        else:
            # single-process case
            completed_ids = compute_completed_2d_exams(
                dataset_parquet=dataset_parquet,
                meta_dir=meta_dir,
                out_path=completed_path,
            )

        df_updated = df[~df["exam_id"].isin(completed_ids)].reset_index(drop=True)

    # Each rank gets a disjoint slice of the updated dataframe
    df_rank = df_updated.iloc[global_rank::world_size].reset_index(drop=True)

    if len(df_rank) == 0:
        print(f"[encode_and_cache r{rank}] No rows assigned to this rank after filtering.")
        return

    tmp_parquet = meta_dir / f"dataset_r{rank:02d}.parquet"
    df_rank.to_parquet(tmp_parquet, index=False)
    if is_3d:
        dataset = CTOrigDataset3D(
            tmp_parquet,
            mode="single",
            apply_encoder=False,  # no encoding yet
            saved_transforms=saved_transforms,
        )
    else:
        dataset = CTOrigDataset2D(
            tmp_parquet,
            mode="single",
            slice_mode="all",
            apply_encoder=False,  # no encoding yet
            saved_transforms=saved_transforms,
        )
    os.remove(tmp_parquet)  # clean up

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_image_meta,
    )

    encoder = encoder.to(device)
    encoder.eval()

    current_latents = []
    current_rows = []
    local_shard_idx = _infer_start_local_shard_idx()  # each rank has its own local shard counter

    def shard_name(local_idx: int) -> str:
        # Shard filenames include rank so processes never collide
        return f"shard_r{rank:02d}_{local_idx:04d}.pt"

    # 3) Flush shard helper
    def flush_shard():
        nonlocal current_latents, current_rows, local_shard_idx
        if not current_latents:
            return

        shard_tensor = torch.stack(current_latents, dim=0)
        sname = shard_name(local_shard_idx)
        shard_file = tensors_dir / sname
        torch.save(shard_tensor, shard_file)

        for offset, row in enumerate(current_rows):
            row["shard"] = sname
            row["offset"] = int(offset)

        index_df = pd.DataFrame(current_rows)
        index_path = meta_dir / f"index_shard_r{rank:02d}_{local_shard_idx:04d}.parquet"
        index_df.to_parquet(index_path, index=False)

        if rank == 0:
            print(
                f"[encode_and_cache r{rank}] Wrote shard {sname} "
                f"with {len(current_latents)} items."
            )

        current_latents = []
        current_rows = []
        local_shard_idx += 1

    # 4) Main loop: batched encoding
    with torch.no_grad():
        for batch_idx, (images, metas) in enumerate(loader):

            B = images.shape[0]
            images = images.to(device, non_blocking=True)
            latents = encoder.encode(images)  # 2D or 3D
            if isinstance(latents, (tuple, list)):
                latents = latents[0]
            latents = latents.detach().cpu().float()

            if is_3d:
                assert latents.dim() == 5, f"Expected 5D latents for 3D, got {latents.shape}"
            else:
                assert latents.dim() == 4, f"Expected 4D latents for 2D, got {latents.shape}"

            for i in range(B):
                meta = metas[i]  
                exam_id = meta["exam_id"]
                cancer = meta["cancer"]

                if is_3d:
                    key = (exam_id,)
                else:
                    slice_idx = int(meta["slice_idx"])
                    key = (exam_id, slice_idx)

                if key in processed_keys:
                    continue  # already encoded in a previous run / shard

                processed_keys.add(key)

                z_i = latents[i]  # 2D: (C,H,W) | 3D: (C,Z,Y,X)
                row = {
                    "exam_id": exam_id,
                    "split": split,
                }

                if is_3d:
                    C_lat, Z_lat, Y_lat, X_lat = z_i.shape
                    row.update({
                        "pid": str(meta['pid']),
                        "cancer": int(cancer),
                        "channels": int(C_lat),
                        "depth": int(Z_lat),
                        "height": int(Y_lat),
                        "width": int(X_lat),
                    })
                else:
                    C_lat, H_lat, W_lat = z_i.shape
                    row.update({
                        "pid": str(meta['pid']),
                        "cancer": int(cancer),
                        "slice_idx": slice_idx,
                        "channels": int(C_lat),
                        "height": int(H_lat),
                        "width": int(W_lat),
                    })

                current_latents.append(z_i)
                current_rows.append(row)

                if len(current_latents) >= shard_size:
                    flush_shard()

            if (batch_idx + 1) % 50 == 0:
                print(f"[encode_and_cache r{rank}] Processed {batch_idx + 1} batches.")

    # Flush any leftover
    flush_shard()

    print(
        f"[encode_and_cache r{rank}] Done. Total unique keys seen: {len(processed_keys)}"
    )
    
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def consolidate_indices(out_root: Union[str, Path] = "/data/rbg/scratch/nlst_final_encoded"):
    """
    Merge all meta/index_shard_*.parquet into meta/index.parquet.

    - For 2D (has 'slice_idx'): drop duplicates on (exam_id, slice_idx, shard, offset)
    - For 3D (no 'slice_idx'):  drop duplicates on (exam_id, shard, offset)
    """
    out_root = Path(out_root)
    meta_dir = out_root / "meta"
    shard_paths = sorted(meta_dir.glob("index_shard_*.parquet"))

    if not shard_paths:
        print(f"[consolidate_indices] No index_shard_*.parquet found in {meta_dir}")
        return

    # --- load new shards ---
    new_dfs = [pd.read_parquet(p) for p in shard_paths]
    new_df = pd.concat(new_dfs, ignore_index=True)

    # --- load existing index if present ---
    index_path = meta_dir / "index.parquet"
    if index_path.exists():
        old_df = pd.read_parquet(index_path)
        df = pd.concat([old_df, new_df], ignore_index=True)
        print(f"[consolidate_indices] Loaded existing index with {len(old_df)} rows.")
    else:
        df = new_df

    if "slice_idx" in df.columns:
        subset_cols = ["exam_id", "slice_idx", "shard", "offset"]
    else:
        subset_cols = ["exam_id", "shard", "offset"]

    df = df.drop_duplicates(subset=subset_cols, keep="last").reset_index(drop=True)

    df.to_parquet(index_path, index=False)
    print(f"[consolidate_indices] Updated {index_path} with {len(df)} total rows.")

    # Optional move all shard files to a subdirectory
    shard_subdir = meta_dir / "shards"
    shard_subdir.mkdir(exist_ok=True)
    for sp in shard_paths:
        sp.rename(shard_subdir / sp.name)

    print(f"[consolidate_indices] Moved {len(shard_paths)} shard files to {shard_subdir}/")


def encode_and_cache_nodule(
    parquet_path_full,
    nodule_parquet_path,
    out_root: Union[str, Path] = "/data/rbg/scratch/nlst_nodule_raw_cache",
    force_patch_size: Optional[tuple] = (128, 128, 32),
    encoder: torch.nn.Module = None,
    do_encode: bool = False,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
    shard_size: int = 500,
    device: str = "cuda",
    rank: int = 0,        # GPU / process id
):
    out_root = Path(out_root)
    tensors_dir = out_root / "tensors"
    meta_dir = out_root / "meta"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    def _infer_start_local_shard_idx() -> int:
        """
        Look for existing shard files for this rank and continue counting
        from the largest existing index + 1.

        Filenames: shard_r{rank:02d}_{idx:04d}.pt
        """
        pattern = f"shard_r{rank:02d}_*.pt"
        existing = list(tensors_dir.glob(pattern))
        if not existing:
            return 0

        max_idx = -1
        for p in existing:
            # stem example: "shard_r00_0003"
            stem = p.stem
            parts = stem.split("_")
            if len(parts) == 3:
                try:
                    idx = int(parts[2])
                    if idx > max_idx:
                        max_idx = idx
                except ValueError:
                    continue
        return max_idx + 1 if max_idx >= 0 else 0

    # 1) Discover existing indices (for resume)
    def _load_existing_keys():
        index_files = sorted(meta_dir.glob("index_shard_*.parquet"))
        shards_dir = meta_dir / "shards"
        if shards_dir.exists():
            index_files += sorted(shards_dir.glob("index_shard_*.parquet"))
        
        if not index_files:
            return set()

        dfs = [pd.read_parquet(p) for p in index_files]
        df = pd.concat(dfs, ignore_index=True)

        processed = set(zip(df["nodule_group"], df["exam"], df["exam_idx"]))

        return processed

    processed_keys = _load_existing_keys()
    print(f"[encode_and_cache r{rank}] Resuming with {len(processed_keys)} already-encoded keys.")

    
    # 2) Dataset and DataLoader + encoder setup
    # Optional: split work across ranks when launched with torchrun
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_rank = int(os.environ.get("RANK", str(rank)))

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # rewrite the parquet removing already processed exams so we don't load them again from scratch
    nodule_df = pd.read_parquet(nodule_parquet_path)
    key_cols = ["nodule_group", "exam", "exam_idx"]
    mask = ~pd.MultiIndex.from_frame(nodule_df[key_cols]).isin(processed_keys)
    nodule_df_updated = nodule_df[mask].reset_index(drop=True)

    # Each rank gets a disjoint slice of the updated dataframe
    df_rank = nodule_df_updated.iloc[global_rank::world_size].reset_index(drop=True)

    if len(df_rank) == 0:
        print(f"[encode_and_cache r{global_rank}] No rows assigned to this rank after filtering.")
        return

    tmp_parquet = meta_dir / f"dataset_r{global_rank:02d}.parquet"
    df_rank.to_parquet(tmp_parquet, index=False)

    if not do_encode:
        dataset = CTNoduleDataset3D(parquet_path_full, tmp_parquet, force_patch_size=force_patch_size, max_nodule_cache_size=10, max_volume_cache_size=20)
    else:
        dataset = CachedNoduleDataset(parquet_path_full, 
                                      tmp_parquet, 
                                      "/data/rbg/scratch/nlst_nodule_raw_cache/meta/index.parquet", 
                                      "/data/rbg/scratch/nlst_nodule_raw_cache", 
                                      split=split,
                                      max_length=None,
                                      max_cache_size=20,
                                      return_meta_data=True)
    os.remove(tmp_parquet)  # clean up

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
        collate_fn=collate_image_meta,
    )

    if do_encode:
        assert encoder is not None, "Encoder must be provided if do_encode is True"
        encoder = encoder.to(device)
        encoder.eval()

    current_images = []
    current_rows = []
    local_shard_idx = _infer_start_local_shard_idx()  # each rank has its own local shard counter

    def shard_name(local_idx: int) -> str:
        # Shard filenames include rank so processes never collide
        return f"shard_r{global_rank:02d}_{local_idx:04d}.pt"

    # 3) Flush shard helper
    def flush_shard():
        nonlocal current_images, current_rows, local_shard_idx
        if not current_images:
            return

        shard_tensor = torch.stack(current_images, dim=0)
        sname = shard_name(local_shard_idx)
        shard_file = tensors_dir / sname
        torch.save(shard_tensor, shard_file)

        for offset, row in enumerate(current_rows):
            row["shard"] = sname
            row["offset"] = int(offset)

        index_df = pd.DataFrame(current_rows)
        index_path = meta_dir / f"index_shard_r{global_rank:02d}_{local_shard_idx:04d}.parquet"
        index_df.to_parquet(index_path, index=False)

        if global_rank == 0:
            print(
                f"[encode_and_cache r{rank}] Wrote shard {sname} "
                f"with {len(current_images)} items."
            )

        current_images = []
        current_rows = []
        local_shard_idx += 1

    print(f"[encode_and_cache r{global_rank}] Starting caching loop...")
    # 4) Main loop: batched encoding
    iterator = loader
    if global_rank == 0:
        iterator = tqdm(loader, total=len(loader), desc="Encoding batches")
    with torch.no_grad():
        for batch_idx, (images, metas) in enumerate(iterator):

            B = images.shape[0]

            if do_encode:
                images = images.to(device, non_blocking=True)
                output = encoder.encode(images)  # 2D or 3D

                if isinstance(output, (tuple, list)):
                    images = output[0]
                else:
                    images = output
                
            images = images.detach().cpu().float()

            assert images.dim() == 5, f"Expected 5D latents for 3D, got {images.shape}"
            
            for i in range(B):
                meta = metas[i]
                nodule_group = meta["nodule_group"]
                exam_id = meta["exam_id"]
                exam_idx = meta["exam_idx"]
                pid = meta["pid"]
                key = (nodule_group, exam_id, exam_idx)
                key_str = f"{nodule_group}_{exam_id}_{exam_idx}"

                if key in processed_keys:
                    continue  # already encoded in a previous run / shard

                processed_keys.add(key)

                # The contiguous().clone() ensures the tensor is not a view and is contiguous in memory
                z_i = images[i].contiguous().clone()  # 2D: (C,H,W) | 3D: (C,Z,Y,X)
                row = {
                    "pid": pid,
                    "nodule_group": nodule_group,
                    "exam": exam_id,
                    "nodule_key": key_str,
                    "exam_idx": exam_idx,
                    "split": split,
                    "encoded": do_encode,
                }

                current_images.append(z_i)
                current_rows.append(row)

                if len(current_images) >= shard_size:
                    flush_shard()

            if (batch_idx + 1) % 100 == 0:
                print(f"[encode_and_cache r{global_rank}] Processed {batch_idx + 1} batches.")

    # Flush any leftover
    flush_shard()

    print(
        f"[encode_and_cache r{global_rank}] Done. Total unique keys seen: {len(processed_keys)}"
    )
    
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def consolidate_indices_nodule(out_root: Union[str, Path] = "/data/rbg/scratch/nlst_nodule_raw_cache"):
    """
    Merge all meta/index_shard_*.parquet into meta/index.parquet.

    """
    out_root = Path(out_root)
    meta_dir = out_root / "meta"
    shard_paths = sorted(meta_dir.glob("index_shard_*.parquet"))

    if not shard_paths:
        print(f"[consolidate_indices] No index_shard_*.parquet found in {meta_dir}")
        return

    # --- load new shards ---
    new_dfs = [pd.read_parquet(p) for p in shard_paths]
    new_df = pd.concat(new_dfs, ignore_index=True)

    # --- load existing index if present ---
    index_path = meta_dir / "index.parquet"
    if index_path.exists():
        old_df = pd.read_parquet(index_path)
        df = pd.concat([old_df, new_df], ignore_index=True)
        print(f"[consolidate_indices] Loaded existing index with {len(old_df)} rows.")
    else:
        df = new_df

    subset_cols = ["nodule_group", "exam", "exam_idx"]
   
    df = df.drop_duplicates(subset=subset_cols, keep="last").reset_index(drop=True)

    df.to_parquet(index_path, index=False)
    print(f"[consolidate_indices] Updated {index_path} with {len(df)} total rows.")

    shard_subdir = meta_dir / "shards"
    shard_subdir.mkdir(exist_ok=True)
    for sp in shard_paths:
        sp.rename(shard_subdir / sp.name)

    print(f"[consolidate_indices] Moved {len(shard_paths)} shard files to {shard_subdir}/")
