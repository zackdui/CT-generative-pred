# Import required libraries
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

import cc3d
from lungmask import LMInferer

from .segmentation_pipeline import nnUNet, nnUNetConfidence, min_max_normalize_batch


def patch_segmenter(patch_tensor: torch.Tensor, segmentation_model: nnUNet, 
                    lungmask_patch_tensor: torch.Tensor | None = None, 
                    confidence_model: Optional[nnUNetConfidence] | None = None) -> torch.Tensor:
    """
    Segments a 3D medical image using a patch-based approach.

    Parameters:
    - patch_tensor (torch.Tensor): The input 3D patch tensor of shape (B, C, D, H, W) = (B, C, Z, Y, X) = (B, 1, 32, 128, 128), TODO = work with batches later
    - segmentation_model (nnUNet): The pre-trained segmentation model.
    - lungmask_patch_tensor (torch.Tensor): The input 3D patch tensor for lung mask of shape (B, D, H, W).
    - confidence_model (nnUNetConfidence, optional): The pre-trained confidence model.

    Returns:
    - torch.Tensor: The segmented output tensor of shape (B, D, H, W).
    """

    # Set to eval mode
    # segmentation_model.eval()
    # if confidence_model is not None:
    #     confidence_model.eval()

    # Set device
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare input tensor for segmentation model
    # window_center, window_width = -600, 1600
    # patches = apply_windowing(patch_tensor.numpy().astype(np.float64), window_center, window_width)
    # # Convert to tensor and normalize
    # patches = torch.tensor(patches)
    # patches = torch.tensor(patches) // 256
    # patches = patches.unsqueeze(1)  # shape: [B, 1, D, H, W], add channel dimension

    # Move to device
    # patches = patches.to(device=device).float()
    # segmentation_model = segmentation_model.to(device=device)
    
    # import pdb; pdb.set_trace()
    # Run segmentation model
    with torch.no_grad():
        # print(f"Max {patch_tensor.max()}, min {patch_tensor.min()}")
        normalized_patches = patch_tensor
        # normalized_patches = min_max_normalize_batch(patch_tensor)
        # print(f"After normalization - Max {normalized_patches.max()}, min {normalized_patches.min()}")
        
        # removed .cpu() on the next line
        if lungmask_patch_tensor is None:
            squeezed_patch = patch_tensor.squeeze(1)
            lungmask_patch_tensor = torch.ones_like(squeezed_patch, device="cuda")
        # import pdb; pdb.set_trace()
        # segmentation_outputs = segmentation_model.predict(normalized_patches.float())
        # binary_segmentation_outputs = (
        #     1 * (F.softmax(segmentation_outputs, 1)[0, 1] > 0.5) * lungmask_patch_tensor
        # )
        binary_segmentation_outputs = segmentation_model(normalized_patches.float())["pred_masks_pos"] * lungmask_patch_tensor
    binary_segmentation = binary_segmentation_outputs
    
    # Get connected components for each patch
    instance_segmentations = []
    
    for i, patch in enumerate(binary_segmentation):
        
        patch = patch.squeeze(0)
        # import pdb; pdb.set_trace()
        patch, num_instances = cc3d.connected_components(
            patch.cpu().numpy(),
            return_N=True,
        )
        instance_segmentations.append(patch)
    instance_segmentation = torch.tensor(np.stack(instance_segmentations, axis=0))  # shape: [B, D, H, W]
    # Include for each patch only the component that contains the most pixels in the center 20 by 20 region of the middle slice
    B, D, H, W = binary_segmentation.shape
    center_z, center_y, center_x = D // 2, H // 2, W // 2
    for b in range(B):

        center_region = instance_segmentation[
            b,
            center_z,
            center_y - 5 : center_y + 5,
            center_x - 5 : center_x + 5,
        ]
        center_labels, counts = torch.unique(center_region, return_counts=True)
        zero_index = center_labels.tolist().index(0) if 0 in center_labels else -1
        if zero_index != -1:
            center_labels = torch.cat((center_labels[:zero_index], center_labels[zero_index+1:]))
            counts = torch.cat((counts[:zero_index], counts[zero_index+1:]))
        

        if len(center_labels) > 0:
            most_frequent_label = center_labels[torch.argmax(counts)]
   
            binary_segmentation[b] = torch.tensor(
                1 * (instance_segmentation[b] == most_frequent_label)
            )
    confidence_scores = None
    # Run confidence model if needed
    if confidence_model is not None:
        patch_tensor = patch_tensor.squeeze(1)
        patch_stack = torch.stack([patch_tensor, binary_segmentation], dim=1).float()  # shape: [B, 2, D, H, W]
        
        with torch.no_grad():
            confidence_outputs = confidence_model(patch_stack)
        logits = confidence_outputs["logit"]
        confidence_scores = F.softmax(logits, dim=1)
        # print(f"confidence scores: {confidence_scores}")
    
    return binary_segmentation, confidence_scores


def load_segmentation_models():
    """
    This returns the segmentation and confidence models.
    """
    # Load segmentation model checkpoint
    segmentation_model_checkpoint = torch.load(
        "/data/rbg/scratch/lung_ct/checkpoints/5678b14bb8a563a32f448d19a7d12e6b/last.ckpt",
        weights_only=False
    )

    print("Loaded segmentation model checkpoint.")

    new_segmentation_model_state_dict = {}
    for k, v in segmentation_model_checkpoint["state_dict"].items():
        if "classifier" not in k:
            new_k = k.replace("model.model", "model")  
            new_segmentation_model_state_dict[new_k] = v

    segmentation_model = nnUNet(
        segmentation_model_checkpoint["hyper_parameters"]["args"]
    )
    segmentation_model.load_state_dict(new_segmentation_model_state_dict)

    print("Segmentation model loaded successfully.")

    # Load confidence model checkpoint
    confidence_model_checkpoint = torch.load(
        "/data/rbg/scratch/lung_ct/checkpoints/4296b4b6cda063e96d52aabfb0694a04/4296b4b6cda063e96d52aabfb0694a04epoch=9.ckpt",
        weights_only=False
    )

    new_confidence_model_state_dict = {}
    for k, v in confidence_model_checkpoint["state_dict"].items():
        new_k = k.replace("model.model", "model")  
        if "model.classifier" in new_k:
            new_k = new_k.replace("model.classifier", "classifier")
        new_confidence_model_state_dict[new_k] = v

    confidence_model = nnUNetConfidence(
        confidence_model_checkpoint["hyper_parameters"]["args"]
    )
    confidence_model.load_state_dict(new_confidence_model_state_dict)

    print("Confidence model loaded successfully.")

    return segmentation_model, confidence_model

def load_lung_model(batch_size=20):

    lungmask_model = LMInferer(
        modelpath="/data/rbg/users/pgmikhael/current/lungmask/checkpoints/unet_r231-d5d2fc3d.pth",
        tqdm_disable=True,
        batch_size=batch_size, # was 100
        force_cpu=False,
    )

    return lungmask_model

if __name__ == "__main__":
    # Load segmentation model checkpoint
    segmentation_model_checkpoint = torch.load(
        "/data/rbg/scratch/lung_ct/checkpoints/5678b14bb8a563a32f448d19a7d12e6b/last.ckpt",
        weights_only=False
    )

    print("Loaded segmentation model checkpoint.")

    new_segmentation_model_state_dict = {}
    for k, v in segmentation_model_checkpoint["state_dict"].items():
        if "classifier" not in k:
            new_k = k.replace("model.model", "model")  
            new_segmentation_model_state_dict[new_k] = v

    segmentation_model = nnUNet(
        segmentation_model_checkpoint["hyper_parameters"]["args"]
    )
    segmentation_model.load_state_dict(new_segmentation_model_state_dict)

    print("Segmentation model loaded successfully.")

    # Load confidence model checkpoint
    confidence_model_checkpoint = torch.load(
        "/data/rbg/scratch/lung_ct/checkpoints/4296b4b6cda063e96d52aabfb0694a04/4296b4b6cda063e96d52aabfb0694a04epoch=9.ckpt",
        weights_only=False
    )

    new_confidence_model_state_dict = {}
    for k, v in confidence_model_checkpoint["state_dict"].items():
        new_k = k.replace("model.model", "model")  
        if "model.classifier" in new_k:
            new_k = new_k.replace("model.classifier", "classifier")
        new_confidence_model_state_dict[new_k] = v

    confidence_model = nnUNetConfidence(
        confidence_model_checkpoint["hyper_parameters"]["args"]
    )
    confidence_model.load_state_dict(new_confidence_model_state_dict)

    print("Confidence model loaded successfully.")


    print("Models loaded successfully.")

    # load example patch tensor batch
    patch_tensor = torch.load("nodule_patches_sample_10000402215824639.pt").cpu()  # shape: [N, D, H, W], insert correct path here for tensor of batch of patches of original images
    lungmask_patch_tensor = torch.load("lungmask_patches_sample_10000402215824639.pt").cpu()  # shape: [N, D, H, W], insert correct path here for tensor of batch of lungmasks of original images
    print(f"Patch tensor shape: {patch_tensor.shape}")
    
    # Load normalization stats from full scan
    # normalization_stats = torch.load("normalization_stats_sample_10000402215824639.pt")
    # print(f"Loaded normalization stats: min={normalization_stats['min_vals'].squeeze().item()}, max={normalization_stats['max_vals'].squeeze().item()}")
    
    # Run patch segmenter
    binary_segmentation = patch_segmenter(
        patch_tensor,
        segmentation_model,
        lungmask_patch_tensor, 
        confidence_model,
        # normalization_stats=normalization_stats,  # Pass the normalization stats
    )  # shape: [B, D, H, W]
    # visualize segmentation
    visualize_segmentation(patch_tensor, binary_segmentation, output_path="segmentation_outputs")
    print(f"Binary segmentation shape: {binary_segmentation.shape}")
    # Calculate volumes
    pixel_spacing = [0.8007810115814209/2, 0.8007810115814209/2, 2.5]  # example pixel spacing (z, y, x) in mm
    pixel_volume = pixel_spacing[0] * pixel_spacing[1] * pixel_spacing[2]  # in mm^3
    volumes = get_volumes(binary_segmentation, pixel_spacing)  # shape: [B,]
    print(f"Nodule volumes (mm^3): {volumes}")
    print(f"total nodule volume (mm^3): {volumes.sum()}")
    
