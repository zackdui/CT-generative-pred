import math
import torch
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import os
import numpy as np
from CTFM.utils import reverse_normalize, window_ct_hu_to_png

class ImageEvaluatorPrep:
    def __init__(self, decode_model, inference_model, n_steps=50, reverse_input=False, interpolate=True):
        self.decode_model = decode_model
        self.inference_model = inference_model
        self.reverse_input = reverse_input
        self.interpolate = interpolate
        self.n_steps = n_steps

    def prepare_images_for_eval(self, images, sample=True):
        if sample:
            images = self.sample_euler(images)
        if self.decode_model is not None:
            images = self.decode_model.decode(images)
        clipped = images.clip(-1, 1)
        hu_images = reverse_normalize(clipped)
        final_images = window_ct_hu_to_png(hu_images)

        if self.interpolate:
            final_images = F.interpolate(
                final_images.float(),
                size=(final_images.shape[2], final_images.shape[3] * 2, final_images.shape[4] * 2),
                mode="trilinear",
                align_corners=False,
            )

        return final_images
    
    @torch.no_grad()
    def sample_euler(self, input: Tensor):
        """
        v_fn: callable (x_t, t_vec) -> v_t
            expects t_vec shape (B,)
        reverse: if reverse is True the v_fn will expect (t_vec, x_t)
        input: Tensor (B, C, D, H, W)
        """
        x_t = input
        B = x_t.shape[0]
        t_vec = torch.linspace(0.0, 1.0, self.n_steps + 1, device=x_t.device, dtype=x_t.dtype)

        for t0, t1 in zip(t_vec[:-1], t_vec[1:]):
            dt = t1 - t0
            t = t0.expand(B)
            if self.reverse_input:
                v = self.inference_model(t, x_t)  # your UNet expects (t, x)
            else:
                v = self.inference_model(x_t, t)  # your UNet expects (x, t)
            x_t = x_t + v * dt
        return x_t

@torch.no_grad()
def sample_euler(v_fn, input: Tensor, n_steps: int = 50, reverse=False):
    """
    v_fn: callable (x_t, t_vec) -> v_t
          expects t_vec shape (B,)
    reverse: if reverse is True the v_fn will expect (t_vec, x_t)
    input: Tensor (B, C, D, H, W)
    """
    x_t = input
    B = x_t.shape[0]
    t_vec = torch.linspace(0.0, 1.0, n_steps + 1, device=x_t.device, dtype=x_t.dtype)

    for t0, t1 in zip(t_vec[:-1], t_vec[1:]):
        dt = t1 - t0
        t = t0.expand(B)
        if reverse:
            v = v_fn(t, x_t)  # your UNet expects (t, x)
        else:
            v = v_fn(x_t, t)  # your UNet expects (x, t)
        x_t = x_t + v * dt
    return x_t



@torch.no_grad()
def generate_images_rk2(v_fn, x_input: Tensor, n_steps: int = 50, reverse=False) -> Tensor:
    """
    RK2 (Heun) integrator for dx/dt = v(x,t).
    Integrates t from 0 -> 1 with n_steps steps.

    v_fn: callable (x_t, t) -> v_t
    reverse: if true then v_fn will expect (t, x_t)
    n_steps: is the steps to compute
    """
    # Start state
    x_t = x_input  # shape (B, C, D, H, W)
    B = x_t.shape[0]

    t_vec = torch.linspace(0.0, 1.0, n_steps + 1, device=x_t.device, dtype=x_t.dtype)

    for t0, t1 in zip(t_vec[:-1], t_vec[1:]):
        dt = t1 - t0
        t0b = t0.expand(B)
        t1b = t1.expand(B)

        # k1 = v(x, t0)
        if reverse:
            k1 = v_fn(t0b, x_t)
        else:
            k1 = v_fn(x_t, t0b)

        # predictor: x_euler = x + dt*k1
        x_euler = x_t + dt * k1

        # k2 = v(x_euler, t1)
        if reverse:
            k2 = v_fn(t1b, x_euler)
        else:
            k2 = v_fn(x_euler, t1b)

        # Heun update: x_{t+dt} = x + dt * (k1 + k2)/2
        x_t = x_t + dt * 0.5 * (k1 + k2)

    return x_t

@torch.no_grad()
def generate_images_rk4(v_fn, x_input: Tensor, n_steps: int = 50, reverse=False) -> Tensor:
    """
    Classic RK4 integrator for dx/dt = v(x,t).
    Integrates t from 0 -> 1 with n_steps steps.

    v_fn: callable (x_t, t) -> v_t
    reverse: if true then v_fn will expect (t, x_t)
    n_steps: is the steps to compute
    """
    x_t = x_input
    B = x_t.shape[0]

    t_vec = torch.linspace(0.0, 1.0, n_steps + 1, device=x_t.device, dtype=x_t.dtype)

    for t0, t1 in zip(t_vec[:-1], t_vec[1:]):
        dt = t1 - t0
        th = t0 + 0.5 * dt  # half-step time

        t0b = t0.expand(B)
        thb = th.expand(B)
        t1b = t1.expand(B)

        if reverse:
            k1 = v_fn(t0b, x_t)
            k2 = v_fn(thb, x_t + 0.5 * dt * k1)
            k3 = v_fn(thb, x_t + 0.5 * dt * k2)
            k4 = v_fn(t1b, x_t + dt * k3)
        else:
            k1 = v_fn(x_t, t0b)
            k2 = v_fn(x_t + 0.5 * dt * k1, thb)
            k3 = v_fn(x_t + 0.5 * dt * k2, thb)
            k4 = v_fn(x_t + dt * k3, t1b)

        x_t = x_t + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return x_t

def binary_classifier_area_growth(initial_areas: torch.Tensor, target_areas: torch.Tensor, threshold, output_dir, show_hist=False) -> torch.Tensor:
    """
    Create binary classifier on if tumor growed based on area comparison. Take difference
    in areas between initial and target images. Any delta area greater than a given threshold will be classified as tumor growth. All input images did have tumor growth

    Provide some data on the distribution of area differences like standard deviation and mean. visualize with histogram.
     
    Args:
        initial_areas (torch.Tensor): Tensor of shape (N,) representing areas from initial images.
        target_areas (torch.Tensor): Tensor of shape (N,) representing areas from target images.
    """

    # Calculate area differences
    area_differences = target_areas - initial_areas

    # Compute mean and standard deviation of area differences
    mean_diff = area_differences.mean()
    std_diff = area_differences.std()
    print(f"Mean area difference: {mean_diff.item()}, Standard Deviation: {std_diff.item()}")   

    # Visualize distribution of area differences
    plt.hist(area_differences.numpy(), bins=30, alpha=0.7, color='blue')
    plt.axvline(mean_diff.item(), color='red', linestyle='dashed', label='Mean')
    plt.axvline((mean_diff - std_diff).item(), color='green', linestyle='dashed', label='Mean - 1 Std Dev')
    plt.title('Distribution of Area Differences')
    plt.xlabel('Area Difference')
    plt.ylabel('Frequency')
    plt.legend()
    if show_hist:
        plt.savefig(f'{output_dir}/area_differences_histogram.png')
        plt.show()
        plt.close()

    # # Create binary classifier based on threshold (mean - 1 std dev) possible alternative
    # threshold = mean_diff - std_diff
    binary_classification = (area_differences > threshold).float()

    # Print classification results
    num_growth = binary_classification.sum().item()
    num_no_growth = (binary_classification == 0).sum().item()
    print(f"\nClassification Results (threshold: {threshold}):")
    print(f"Classified as growth: {num_growth}/{len(binary_classification)}")
    print(f"Classified as no growth: {num_no_growth}/{len(binary_classification)}")

    return binary_classification


def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    Compute the Dice Score between predicted and target tensors (each are masks of 1's and 0's).

    Args:
        pred (torch.Tensor): Predicted tensor of shape (N, C, H, W) or (N, H, W).
            where N is batch size, C is number of classes, H and W are height and width.
        target (torch.Tensor): Ground truth tensor of the same shape as pred.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: Dice Score.
    """

    # Flatten the tensors
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    # Calculate number of true positives
    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

    return dice.item()

def f1_score(binaryPreds: torch.Tensor, binaryTargets: torch.Tensor) -> float:
    """
    Compute the F1 Score between binary predictions and binary targets.

    Args:
        binaryPreds (torch.Tensor): Binary predictions tensor of shape (N) for N samples.
        binaryTargets (torch.Tensor): Binary ground truth tensor of the same shape as binaryPreds.

    Returns:
        float: F1 Score.
    """

    # Calculate true positives, false positives, and false negatives
    true_positives = (binaryPreds * binaryTargets).sum().item()
    false_positives = (binaryPreds * (1 - binaryTargets)).sum().item()
    false_negatives = ((1 - binaryPreds) * binaryTargets).sum().item()

    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)

    # Calculate F1 Score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return f1

def get_volumes(binary_segmentation: torch.Tensor, pixel_spacing: list[float]) -> torch.Tensor:
    """
    Gets volume of segmented nodule patches for batch of binary segmentations.

    Parameters:
    - binary_segmentation (torch.Tensor): The binary segmentation tensor of shape (B, D, H, W).
    - pixel_spacing (list[float]): The pixel spacing in mm for each dimension (z, y, x).

    Returns:
    - volumes (torch.Tensor): The volumes of the segmented nodules for each patch in the batch, shape (B,).
    """
    pixel_volume = pixel_spacing[0] * pixel_spacing[1] * pixel_spacing[2]  # in mm^3 ??
    patch_pixel_counts = binary_segmentation.sum(dim=(1, 2, 3))  # shape: (B,)
    # print(f"patch pixel counts: {patch_pixel_counts}")
    volumes = patch_pixel_counts * pixel_volume  # shape: (B,)
    return volumes

def visualize_segmentation(patch_tensor: torch.Tensor, binary_segmentation: torch.Tensor, output_path: str, bbox=None, code="8"):
    """
    visualizes the binary segmentation of each patch in the batch with 
    the slice having the largest nodule area highlighted.

    Parameters:
    - patch_tensor (torch.Tensor): The input patch tensor of shape (B, 1, D, H, W).
    - binary_segmentation (torch.Tensor): The binary segmentation tensor of shape (B, D, H, W).
    - output_path (str): The path to save the visualization output.
    - if bbox is provided it will be plotted on the slices
    - code (str): The code to make a patch unique.
    """
    batch_size = patch_tensor.shape[0]
    for i in range(batch_size):
        patch = patch_tensor[i][0]  # shape: (D, H, W)
        segmentation = binary_segmentation[i]  # shape: (D, H, W)

        # Find the slice with the largest nodule area
        slice_areas = segmentation.sum(dim=(1, 2))  # shape: (D,)
        # max_slice_idx = torch.argmax(slice_areas).item()
        topk = min(5, slice_areas.numel())
        top_slice_idxs = torch.topk(slice_areas, k=topk).indices.tolist()

        # Plot the slice and its segmentation
        for rank, max_slice_idx in enumerate(top_slice_idxs):
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(patch[max_slice_idx].cpu(), cmap='gray')
            ax[0].set_title('Original Patch Slice')
            ax[1].imshow(segmentation[max_slice_idx].cpu(), cmap='gray')
            ax[1].set_title('Binary Segmentation Slice')

            # ---- draw bbox if provided ----
            if bbox is not None:
                i0, i1, j0, j1, k0, k1 = bbox[i]

                # Only draw if this slice intersects the bbox in Z
                if k0 <= max_slice_idx <= k1:
                    height = i1 - i0 + 1
                    width  = j1 - j0 + 1

                    rect = patches.Rectangle(
                        (j0, i0),            # (x, y)
                        width,
                        height,
                        linewidth=2,
                        edgecolor="red",
                        facecolor="none",
                    )
                    ax[0].add_patch(rect)

            # plt.suptitle(f'Patch {i} - Slice {max_slice_idx} with Largest Nodule Area')
            plt.suptitle(f'Patch {i} - Top-{rank+1} Slice z={max_slice_idx}')
            # plt.savefig(f"{output_path}/patch_{code}_{i}_segmentation_scaled.png")
            plt.savefig(f"{output_path}/patch_{code}_{i}_top{rank}_z{max_slice_idx}_segmentation_scaled.png")
            plt.close()

def save_montage_with_bbox3d(
    volume,                 # shape (Z, H, W) or (1, Z, H, W)
    out_path,
    slice_indices=list(range(0, 32)),
    ncols=6,
    cmap="gray",
    vmin=None,
    vmax=None,
    title=None,
    show_slice_labels=True,
    save_fig=True,
    return_fig=False,
    bbox_yxz=None,          # (y0, y1, x0, x1, z0, z1) in voxel indices
    bbox_color="red",
    bbox_lw=2.0,
):
    """
    Draws a 3D bounding box by overlaying a 2D rectangle on each slice where z is in [z0, z1).

    bbox_yxz:
        Tuple[int,int,int,int,int,int] = (y0, y1, x0, x1, z0, z1)
        Convention: end-inclusive (z1, y1, x1 included). Both endpoints are drawn

    If you return the fig make sure to do plt.close(fig) after returning it.
    """
    if len(volume.shape) == 4:
        volume = volume.squeeze(0)

    if torch.is_tensor(volume) and volume.is_cuda:
        volume = volume.detach().to(dtype=torch.float32).cpu()

    Z, H, W = volume.shape[0], volume.shape[1], volume.shape[2]

    # sanitize slice indices
    slice_indices = [int(k) for k in slice_indices if 0 <= int(k) < Z]
    if len(slice_indices) == 0:
        raise ValueError("No valid slice indices to montage.")

    # sanitize bbox (optional)
    bbox = None
    if bbox_yxz is not None:
        if len(bbox_yxz) != 6:
            raise ValueError("bbox_yxz must be a 6-tuple: (y0, y1, x0, x1, z0, z1)")
        y0, y1, x0, x1, z0, z1 = [int(v) for v in bbox_yxz]

        # clamp to valid ranges (end-exclusive)
        z0 = max(0, min(z0, Z - 1))
        z1 = max(0, min(z1, Z - 1))
        y0 = max(0, min(y0, H - 1))
        y1 = max(0, min(y1, H - 1))
        x0 = max(0, min(x0, W - 1))
        x1 = max(0, min(x1, W - 1))

        if (z1 > z0) and (y1 > y0) and (x1 > x0):
            bbox = (z0, z1, y0, y1, x0, x1)
        # else: invalid bbox -> ignore silently

    n = len(slice_indices)
    nrows = math.ceil(n / ncols)

    # Make figure size scale with grid (same as yours)
    fig_w = 2.2 * ncols
    fig_h = 2.2 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))

    axes = np.atleast_2d(axes)

    for idx, k in enumerate(slice_indices):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        slice_ = volume[k]

        if torch.is_tensor(slice_):
            slice_ = slice_.numpy()

        ax.imshow(slice_, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis("off")

        if show_slice_labels:
            ax.set_title(f"z={k}", fontsize=10)

        # draw bbox rectangle on this slice if within z-range
        if bbox is not None:
            z0, z1, y0, y1, x0, x1 = bbox
            if z0 <= k <= z1:
                rect = Rectangle(
                    (x0, y0),                 # (x, y) top-left in image coords
                    (x1 - x0 + 1),                # width
                    (y1 - y0 + 1),                # height
                    fill=False,
                    edgecolor=bbox_color,
                    linewidth=bbox_lw,
                )
                ax.add_patch(rect)

    # Turn off any unused panels
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout()
    if title:
        fig.subplots_adjust(top=0.92)

    if save_fig:
        out_dir = os.path.dirname(str(out_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=200)

    if return_fig:
        return fig

    plt.close(fig)


def scale_bbox_after_interpolate(bbox, sy=2.0, sx=2.0, sz=1.0):
    i0, i1, j0, j1, k0, k1 = bbox

    def scale_inclusive(lo, hi, s):
        lo2 = int(math.floor(lo * s))
        hi2 = int(math.floor((hi + 1) * s) - 1)
        return lo2, hi2

    i0, i1 = scale_inclusive(i0, i1, sy)  # Y
    j0, j1 = scale_inclusive(j0, j1, sx)  # X
    k0, k1 = scale_inclusive(k0, k1, sz)  # Z (unchanged here)
    return (i0, i1, j0, j1, k0, k1)
