
from torch.utils.data import Dataset
from typing import Callable, Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
# import pyarrow.parquet as pq
import pyarrow.dataset as pa_ds
from collections import OrderedDict
import ants

from CTFM.data.utils import apply_transforms, get_ants_image_from_row, ants_crop_or_pad_like_torchio, ants_to_normalized_tensor

########## Caches ##########
class _ImageCache:
    """Tiny LRU cache for transformed images."""
    def __init__(self, root: Path, map_location: str = "cpu", max_open: int = 1000):
        self.root = root
        self.map_location = map_location
        self.max_open = max_open
        self._open: OrderedDict[str, torch.Tensor] = OrderedDict()

    def get_image(self, exam_id: str) -> torch.Tensor:
        if exam_id in self._open:
            return self._open[exam_id]
        else:
            return None

    def add_image(self, exam_id: str, image: torch.Tensor):
        self._open[exam_id] = image
        if len(self._open) > self.max_open:
            self._open.popitem(last=False)

########### Datasets ##########
    
class CTOrigDataset2D(Dataset):
    def __init__(self, 
                 parquet_path: str, 
                 parquet_pairs_path: Optional[str] = None,
                 apply_encoder: bool = False,
                 encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 device: str = "cuda",
                 saved_transforms: Optional[Dict[str, str]] = None,
                 mode: str = 'single', # Also pair mode
                 slice_mode: str = 'random', # all
                 slices_per_scan: int = 5, # Only for random mode # Must be less then min number of slices in any scan
                 crop_pad: bool = True, 
                 image_size: Tuple[int, int] = (512, 512),
                 resample: bool = True,
                 resample_size: Tuple[int, int] = (0.703125 ,0.703125, 2.5),
                 clip_window: Tuple[int, int] = (-1000, 400),
                 max_cache_size: int = 1000,
                 max_length: Optional[int] = None):
        """
        parquet_path: Path to the parquet file with the CT scan metadata.
        parquet_pairs_path: Path to the parquet file with the CT scan pairs metadata (only for pair mode).
        apply_encoder: Whether to apply the encoder to the images before returning.
        encoder: Should be an nn.Module with an .encode() method that
                takes in a batch of images and returns the latent representation.
                If the encoder returns a tuple/list, the first element will be used.
        device: Device to run the encoder on.
        saved_transforms: Dict mapping exam_id to path of saved nifti transform files (optional).
        mode: 'single' for single scan mode, 'pair' for scan pair mode.
        slice_mode: 'random' to sample random slices, 'all' to iterate over all slices.
        slices_per_scan: Number of slices to sample per scan in random mode.
        crop_pad: Whether to crop/pad the images to the target size.
        image_size: Target (X, Y) size for the images.
        resample: Whether to resample the images before applying transforms.
        resample_size: Voxel size (X, Y) to resample to.
        clip_window: HU window (min, max) for normalization.
        max_cache_size: Maximum number of images to keep in the cache.
        max_length: Optional maximum length of the dataset (for debugging).

        Images will be returned as torch Tensors of shape (C, Y, X) where C=1 for single mode and C=2 for pair mode.
        Values will be normalized to the range [-1, 1].
        Only images with registrations will be included.
        In metadata it will include 'slice_idx' indicating which slice was returned.

        If encode is applied, the returned images will be the output of the encoder.
        """
        self.df = pd.read_parquet(parquet_path)
        # Make sure pd.Na will be treated as False where appropriate
        self.df["registration_exists"] = self.df["registration_exists"].fillna(False)
        # Filter to only those with registrations
        self.df = self.df[self.df["registration_exists"] == True].reset_index(drop=True)
        self.df["has_nifti"] = self.df["has_nifti"].fillna(False)
        if mode == 'pair':
            assert parquet_pairs_path is not None, "parquet_pairs_path must be provided in pair mode"
            self.id_to_index = {row['exam_id']: idx for idx, row in self.df.iterrows()}
        if parquet_pairs_path is not None:
            self.pairs_df = pd.read_parquet(parquet_pairs_path)
        self.apply_encoder = apply_encoder
        self.device = device
        self.encoder = encoder
        if encoder is not None:
            self.encoder = self.encoder.to(self.device)
        self.saved_transforms = saved_transforms if saved_transforms is not None else dict()
        assert mode in ['single', 'pair'], "mode must be 'single' or 'pair'"
        assert slice_mode in ['random', 'all'], "slice_mode must be 'random' or 'all'"
        self.mode = mode
        self.slice_mode = slice_mode
        self.slices_per_scan = slices_per_scan
        self.crop_pad = crop_pad
        self.image_size = image_size
        self.pad_value = clip_window[0]
        self.resample = resample
        self.resample_size = resample_size
        self.clip_window = clip_window
        self.max_cache_size = max_cache_size
        self.cache = _ImageCache(root=Path(parquet_path).parent, max_open=self.max_cache_size)
        self.max_length = max_length
        if self.mode == 'single':
            if self.slice_mode == 'all':
                total_slices = 0
                for idx in range(len(self.df)):
                    row = self.df.iloc[idx]
                    image_size = int(row["num_slices"])
                    total_slices += image_size
                self.total_slices = total_slices
            else:
                self.total_slices = len(self.df) * self.slices_per_scan

            if self.slice_mode == 'all':
                self.slice_counts = self.df["num_slices"].astype(int).tolist()
                self.prefix = np.zeros(len(self.slice_counts) + 1, dtype=np.int64)
                self.prefix[1:] = np.cumsum(self.slice_counts)
        else: # Mode is pairs
            if self.slice_mode == 'all':
                total_slices = 0
                for idx in range(len(self.pairs_df)):
                    pair_row = self.pairs_df.iloc[idx]
                    row_a = self.df.iloc[self.id_to_index[pair_row['exam_id_a']]]
                    # After applying registration the fixed scan determines the number of slices
                    image_size = int(row_a["fixed_shape"][2])
                    total_slices += image_size
                self.total_slices = total_slices
            else:
                self.total_slices = len(self.pairs_df) * self.slices_per_scan

            if self.slice_mode == 'all':
                self.slice_counts = []
                for idx in range(len(self.pairs_df)):
                    pair_row = self.pairs_df.iloc[idx]
                    row_a = self.df.iloc[self.id_to_index[pair_row['exam_id_a']]]
                    self.slice_counts.append(int(row_a["fixed_shape"][2]))
                self.prefix = np.zeros(len(self.slice_counts) + 1, dtype=np.int64)
                self.prefix[1:] = np.cumsum(self.slice_counts)


    def __len__(self):
        return min(self.total_slices, self.max_length) if self.max_length is not None else self.total_slices

    def get_scan_idx(self, index: int) -> Tuple[int, int]:
        if self.slice_mode == 'all':
            i = self.prefix.searchsorted(index, side="right") - 1
            slice_idx = index - self.prefix[i]
            return i, slice_idx
        else:
            scan_idx = index // self.slices_per_scan
            if self.mode == 'single':
                row = self.df.iloc[scan_idx]
                num_slices = int(row["num_slices"])
                slice_idx = np.random.randint(0, num_slices)
            else:
                row = self.pairs_df.iloc[scan_idx]
                row_a = self.df.iloc[self.id_to_index[row['exam_id_a']]]
                num_slices = int(row_a["fixed_shape"][2])
                slice_idx = np.random.randint(0, num_slices)
            
            return scan_idx, slice_idx
        
    def process_row_single(self, row, slice_idx) -> torch.Tensor:
        exam_id = row['exam_id']
        image_tensor = self.cache.get_image(exam_id)
        if image_tensor is not None:
            image_tensor_slice = image_tensor[:, slice_idx, :, :]
            # Shape (1, Y, X)
            return image_tensor_slice
        # If the transform was saved as a nifti file load that
        if exam_id in self.saved_transforms:
            nifti_path = self.saved_transforms[exam_id]
            ants_image = ants.image_read(nifti_path)
            ants_transformed_image = ants_crop_or_pad_like_torchio(ants_image,
                                                                  target_size=(self.image_size[0], self.image_size[1], ants_image.shape[2]),
                                                                  pad_value=self.pad_value,
                                                                  only_xy=True)
        else:
            ants_image = get_ants_image_from_row(row)
            cur_image_size = ants_image.shape
            target_size_3d = (self.image_size[0], self.image_size[1], cur_image_size[2])
            forward_transform = row["registration_file"]
            if pd.isna(forward_transform):
                forward_transform = None
            ants_transformed_image = apply_transforms(ants_image,
                                                      forward_transform=forward_transform,
                                                      row=row,
                                                      resampling=self.resample, 
                                                      resampling_params=self.resample_size, 
                                                      crop_pad=self.crop_pad, 
                                                      target_size=target_size_3d,
                                                      pad_hu=self.pad_value,
                                                      only_xy=True)
        # Image will now be range (-1, 1) and shape (1, Z, Y, X)
        image_tensor = ants_to_normalized_tensor(ants_transformed_image,
                                                 clip_window=self.clip_window)
        assert image_tensor.dim() == 4, \
            f"Expected (1, Z, Y, X) but got {image_tensor.shape} for exam_id={exam_id}"
        assert image_tensor.shape[2] == self.image_size[0] and image_tensor.shape[3] == self.image_size[1], \
            f"XY mismatch after crop_pad for exam_id={exam_id}: got {image_tensor.shape[2:]}"
        self.cache.add_image(exam_id, image_tensor)
        image_tensor_slice = image_tensor[:, slice_idx, :, :]
        # Shape (1, Y, X)
        return image_tensor_slice
    
    def __getitem__(self, index: int):
        exam_index, slice_idx = self.get_scan_idx(index)

        if self.mode == 'single':
            row = self.df.iloc[exam_index]
            image_tensor_slice = self.process_row_single(row, slice_idx)
            row['slice_idx'] = slice_idx
            return image_tensor_slice, row
        elif self.mode == 'pair':
            row = self.pairs_df.iloc[exam_index]
            exam_id_a = row['exam_id_a']
            exam_id_b = row['exam_id_b']

            index_a = self.id_to_index[exam_id_a]
            index_b = self.id_to_index[exam_id_b]
            row_a = self.df.iloc[index_a]
            row_b = self.df.iloc[index_b]

            image_tensor_slice_a = self.process_row_single(row_a, slice_idx)
            image_tensor_slice_b = self.process_row_single(row_b, slice_idx)
            assert image_tensor_slice_a.shape == image_tensor_slice_b.shape, \
                f"Shape mismatch between pair volumes A={image_tensor_slice_a.shape}, B={image_tensor_slice_b.shape}"
            if self.apply_encoder:
                assert self.encoder is not None, "Encoder must be provided if apply_encoder is True"

                image_tensor_slice_a = image_tensor_slice_a.unsqueeze(0)  # Add batch dimension
                image_tensor_slice_b = image_tensor_slice_b.unsqueeze(0)  # Add batch dimension
                images = torch.cat([image_tensor_slice_a, image_tensor_slice_b], dim=0)  # Shape (2, 1, Y, X)
                encoded_images = self.encoder.encode(images.to(self.device))  # Assume encoder can handle batch of images
                if isinstance(encoded_images, tuple) or isinstance(encoded_images, list):
                    encoded_images = encoded_images[0]
                image_tensor_slice_a = encoded_images[0]
                image_tensor_slice_b = encoded_images[1]
                
            # Concatenate the two image tensors along the channel dimension: shape (2, Y, X)
            image_tensor = torch.cat([image_tensor_slice_a, image_tensor_slice_b], dim=0)
            row['slice_idx'] = slice_idx
            return image_tensor, row
        

class CTOrigDataset3D(Dataset):
    def __init__(self, 
                 parquet_path: str, 
                 parquet_pairs_path: Optional[str] = None,
                 apply_encoder: bool = False,
                 encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 device: str = "cuda",
                 saved_transforms: Optional[Dict[str, str]] = None,
                 mode: str = 'single', # Also pair mode
                 crop_pad: bool = True, 
                 image_size: Tuple[int, int, int] = (512, 512, 208),
                 resample: bool = True,
                 resample_size: Tuple[int, int, int] = (0.703125 ,0.703125, 2.5),
                 clip_window: Tuple[int, int] = (-1000, 400),
                 max_cache_size: int = 1000,
                 max_length: Optional[int] = None,
                 return_meta_data: bool = True):
        """
        parquet_path: Path to the parquet file with the CT scan metadata.
        parquet_pairs_path: Path to the parquet file with the CT scan pairs metadata (only for pair mode).
        apply_encoder: Whether to apply the encoder to the images before returning.
        encoder: Should be an nn.Module with an .encode() method that
                takes in a batch of images and returns the latent representation.
                If the encoder returns a tuple/list, the first element will be used.
        device: Device to run the encoder on.
        saved_transforms: Dict mapping exam_id to path of saved nifti transform files (optional).
        mode: 'single' for single scan mode, 'pair' for scan pair mode.
        crop_pad: Whether to crop/pad the images to the target size.
        image_size: Target (X, Y, Z) size for the images.
        resample: Whether to resample the images before applying transforms.
        resample_size: Voxel size (X, Y, Z) to resample to.
        clip_window: HU window (min, max) for normalization.
        max_cache_size: Maximum number of images to keep in the cache.
        max_length: Optional maximum length of the dataset (for debugging).
        return_meta_data: Whether to return the metadata row along with the image.

        Images will be returned as torch Tensors of shape (C, Z, Y, X) where C=1 for single mode and C=2 for pair mode.
        Values will be normalized to the range [-1, 1].
        Only images with registrations will be included.

        If encode is applied, the returned images will be the output of the encoder.
        """
        self.df = pd.read_parquet(parquet_path)
        # Make sure pd.Na will be treated as False where appropriate
        self.df["registration_exists"] = self.df["registration_exists"].fillna(False)
        # Filter to only those with registrations
        self.df = self.df[self.df["registration_exists"] == True].reset_index(drop=True)
        self.df["has_nifti"] = self.df["has_nifti"].fillna(False)
        if mode == 'pair':
            assert parquet_pairs_path is not None, "parquet_pairs_path must be provided in pair mode"
            self.id_to_index = {row['exam_id']: idx for idx, row in self.df.iterrows()}
        if parquet_pairs_path is not None:
            self.pairs_df = pd.read_parquet(parquet_pairs_path)
        self.apply_encoder = apply_encoder
        self.device = device
        self.encoder = encoder
        if encoder is not None:
            self.encoder = self.encoder.to(self.device)
        self.saved_transforms = saved_transforms if saved_transforms is not None else dict()
        assert mode in ['single', 'pair'], "mode must be 'single' or 'pair'"
        self.mode = mode
        self.crop_pad = crop_pad
        self.image_size = image_size
        self.pad_value = clip_window[0]
        self.resample = resample
        self.resample_size = resample_size
        self.clip_window = clip_window
        self.max_cache_size = max_cache_size
        self.cache = _ImageCache(root=Path(parquet_path).parent, max_open=self.max_cache_size)
        self.max_length = max_length
        self.return_meta_data = return_meta_data

    def __len__(self):
        if self.mode == 'single':
            return min(len(self.df), self.max_length) if self.max_length is not None else len(self.df)
        else:
            return min(len(self.pairs_df), self.max_length) if self.max_length is not None else len(self.pairs_df)

    def process_row_single(self, row) -> torch.Tensor:
        exam_id = row['exam_id']
        image_tensor = self.cache.get_image(exam_id)
        if image_tensor is not None:
            # Shape (1, Z, Y, X)
            return image_tensor
        # If the transform was saved as a nifti file load that
        if exam_id in self.saved_transforms:
            nifti_path = self.saved_transforms[exam_id]
            ants_image = ants.image_read(nifti_path)
            ants_transformed_image = ants_crop_or_pad_like_torchio(ants_image,
                                                                  target_size=self.image_size,
                                                                  pad_value=self.pad_value,
                                                                  only_xy=False)
        else:
            ants_image = get_ants_image_from_row(row)
            target_size_3d = self.image_size
            forward_transform = row["registration_file"]
            if pd.isna(forward_transform):
                forward_transform = None
            ants_transformed_image = apply_transforms(ants_image,
                                                      forward_transform=forward_transform,
                                                      row=row,
                                                      resampling=self.resample, 
                                                      resampling_params=self.resample_size, 
                                                      crop_pad=self.crop_pad, 
                                                      target_size=target_size_3d,
                                                      pad_hu=self.pad_value,
                                                      only_xy=False)
        # Image will now be range (-1, 1) and shape (1, Z, Y, X)
        image_tensor = ants_to_normalized_tensor(ants_transformed_image,
                                                 clip_window=self.clip_window)
        assert image_tensor.dim() == 4, \
            f"Expected (1, Z, Y, X) but got {image_tensor.shape} for exam_id={exam_id}"
        assert image_tensor.shape[1] == self.image_size[2] and image_tensor.shape[2] == self.image_size[0] and image_tensor.shape[3] == self.image_size[1], \
            f"XYZ mismatch after crop_pad for exam_id={exam_id}: got {image_tensor.shape[1:]}"
        self.cache.add_image(exam_id, image_tensor)
        # Shape (1, Z, Y, X)
        return image_tensor
    
    def __getitem__(self, index: int):

        if self.mode == 'single':
            row = self.df.iloc[index]
            image_tensor_slice = self.process_row_single(row)
            return image_tensor_slice, row
        elif self.mode == 'pair':
            row = self.pairs_df.iloc[index]
            exam_id_a = row['exam_id_a']
            exam_id_b = row['exam_id_b']

            index_a = self.id_to_index[exam_id_a]
            index_b = self.id_to_index[exam_id_b]
            row_a = self.df.iloc[index_a]
            row_b = self.df.iloc[index_b]

            image_tensor_a = self.process_row_single(row_a)
            image_tensor_b = self.process_row_single(row_b)
            assert image_tensor_a.shape == image_tensor_b.shape, \
                f"Shape mismatch between pair volumes A={image_tensor_a.shape}, B={image_tensor_b.shape}"
            if self.apply_encoder:
                assert self.encoder is not None, "Encoder must be provided if apply_encoder is True"
                image_tensor_a = image_tensor_a.unsqueeze(0)  # Add batch dimension
                image_tensor_b = image_tensor_b.unsqueeze(0)  # Add batch dimension
                images = torch.cat([image_tensor_a, image_tensor_b], dim=0)  # Shape (2, 1, Z, Y, X)
                encoded_images = self.encoder.encode(images.to(self.device))  # Assume encoder can handle batch of images
                if isinstance(encoded_images, tuple) or isinstance(encoded_images, list):
                    encoded_images = encoded_images[0]
                image_tensor_a = encoded_images[0]
                image_tensor_b = encoded_images[1]

            # Concatenate the two image tensors along the channel dimension: shape (2, Z, Y, X)
            image_tensor = torch.cat([image_tensor_a, image_tensor_b], dim=0)
            if self.return_meta_data:
                return image_tensor, row
            return image_tensor

class RepeatedImageDataset(Dataset):
        def __init__(self, image_tensor, repeat_count):
            self.image = image_tensor
            self.repeat_count = repeat_count

        def __len__(self):
            return self.repeat_count

        def __getitem__(self, idx):
            return self.image