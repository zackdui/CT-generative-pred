import sys
import os
from monai.networks.nets import DiffusionModelUNet
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json

# For local use of vae3d2d module
# sys.path.insert(0, "/data/rbg/users/duitz/VAE3d/src")
from vae3d2d import CustomVAE, AttnParams, setup_logger, eval_model_3D

from CTFM.utils import load_config, OPTIMIZERS
from CTFM.data import CachedNoduleDataset, collate_image_meta, recover_small_bbox, save_montage, save_two_figs_side_by_side
from CTFM.utils import (
                        load_config, 
                        OPTIMIZERS,
                        window_ct_hu_to_png,
                        reverse_normalize)
from CTFM.models import UnetLightning3D
from CTFM.eval import (load_segmentation_models, 
                       ImageEvaluatorPrep, 
                       patch_segmenter, 
                       get_volumes, 
                       visualize_segmentation,
                       scale_bbox_after_interpolate,
                       save_montage_with_bbox3d)



if __name__=="__main__":
    print("Finished Loading imports")
    ## Variables
    # fm_model_checkpoint = "experiments/fm_3d_pretrain/full_latent/lightning_logs/version_1/checkpoints/val-epoch=429-val_loss=0.32.ckpt"
    # vae_checkpoint = "/data/rbg/users/duitz/CT-generative-pred/final_saved_models/vae_fixed_std_no_reg.pt"
    batch_size = 2
    max_saved_examples = 1
    # interpolate=True
    # if interpolate:
    interpolated_pixel_spacing = [0.703125 / 2, 0.703125 / 2, 2.5]
    # else:
    non_interpolated_pixel_spacing = [0.703125, 0.703125, 2.5]

    # max_examples = 10

    path_yaml = "configs/paths.yaml"

    ## Load Dataset
    base_paths = load_config(path_yaml)
    full_train_parquet = base_paths.full_data_train_parquet
    full_nodule_parquet = base_paths.bounding_boxes_train_parquet
    # raw_nodule_index = base_paths.encoded_nodule_test_index
    paired_nodule_parquet = base_paths.paired_nodules_train_parquet
    # data_root = "/data/rbg/scratch/test_nlst_nodule_encoded_cache"

    # test_dataset = CachedNoduleDataset(full_test_parquet, 
    #                                    full_nodule_parquet, 
    #                                    raw_nodule_index, 
    #                                    data_root, 
    #                                    paired_nodule_parquet=paired_nodule_parquet,
    #                                    mode="paired",
    #                                    split="test",
    #                                    max_length=None,
    #                                    max_cache_size=50,
    #                                    return_meta_data=True)
    
    # Alternative dataset with raw cached nodule dir
    test_dataset = CachedNoduleDataset(full_train_parquet, 
                                       full_nodule_parquet, 
                                       base_paths.raw_nodule_index, 
                                       base_paths.raw_cached_nodule_dir, 
                                       paired_nodule_parquet=paired_nodule_parquet,
                                       mode="paired",
                                       split="train",
                                       max_length=None,
                                       max_cache_size=50,
                                       return_meta_data=True)
    
    ## Load Models
    # vae_model = CustomVAE.load_from_checkpoint(vae_checkpoint, map_location="cuda")
    # lit = UnetLightning3D.load_from_checkpoint(
    #                                             fm_model_checkpoint,
    #                                             unet_cls=DiffusionModelUNet,
    #                                             strict=False,   
    #                                         )
    seg_model, confidence_model = load_segmentation_models()

    ## Set all models to cuda and eval
    # vae_model.eval().to("cuda")
    # lit.eval().to("cuda")
    seg_model.eval().to("cuda")
    confidence_model.eval().to("cuda")

    ## Run Evaluation
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 2, drop_last=False, collate_fn=collate_image_meta)
    # evaluator_prep = ImageEvaluatorPrep(vae_model, lit, n_steps=50, reverse_input=False, interpolate=interpolate)

    example_one = True

    def prep_images(image_to_prep, interpolate=False):
        hu_images = reverse_normalize(image_to_prep)
        final_images = window_ct_hu_to_png(hu_images)

        if interpolate:
            final_images = F.interpolate(
                final_images.float(),
                size=(final_images.shape[2], final_images.shape[3] * 2, final_images.shape[4] * 2),
                mode="trilinear",
                align_corners=False,
            )
        return final_images
    
    output_data = {}
    saved_examples = 0

    print("Starting Going through loader")
    total = 0
    for batch in test_loader:
        images, meta = batch

        input_images, output_images = torch.chunk(images, 2, dim=1)

        input_images, output_images = input_images.to("cuda"), output_images.to("cuda")
        
        preped_input_images = prep_images(input_images, interpolate=False)
        preped_input_images_inter = prep_images(input_images, interpolate=True)

        preped_output_images = prep_images(output_images, interpolate=False)
        preped_output_images_inter = prep_images(output_images, interpolate=True)

        binary_segmentation_input, confidence_scores_input = patch_segmenter(preped_input_images_inter, seg_model, lungmask_patch_tensor=None, confidence_model=confidence_model)
        binary_segmentation_output, confidence_scores_output = patch_segmenter(preped_output_images_inter, seg_model, lungmask_patch_tensor=None, confidence_model=confidence_model)

        
        volumes_in = get_volumes(binary_segmentation_input, pixel_spacing=interpolated_pixel_spacing)
        volumes_out = get_volumes(binary_segmentation_output, pixel_spacing=interpolated_pixel_spacing)

        
        input_bboxes = []
        output_bboxes = []
        for i in range(len(meta)):
            ## For the first image in the pair get the box
            bbox_a = meta[i]['coords_fixed_a']
            spacing_a = meta[i]['fixed_spacing_a']
            shape_a = meta[i]['fixed_shape_a']
            assert meta[i]['nodule_group_a'] == meta[i]['nodule_group_b']
            nodule_id_a = f"{meta[i]['nodule_group_a']}_{meta[i]['exam_a']}_{meta[i]['exam_idx_a']}"
            
                
            shape_XYZ_a = tuple(
                int(round(n * s_old / s_new))
                for n, s_old, s_new in zip(shape_a, spacing_a, non_interpolated_pixel_spacing)
            )

            small_bbox_input = recover_small_bbox(bbox_a, shape_XYZ_a)

            small_bbox_input_interp = scale_bbox_after_interpolate(small_bbox_input)

            input_bboxes.append(small_bbox_input)

            output_data[nodule_id_a] = {
                "bbox": [int(x) for x in small_bbox_input],
                "bbox_interp": [int(x) for x in small_bbox_input_interp],
                "nodule_volume": volumes_in[i].item()
            }


            ## For the second image in the pair get the bounding box
            bbox_b = meta[i]['coords_fixed_b']
            spacing_b = meta[i]['fixed_spacing_b']
            shape_b = meta[i]['fixed_shape_b']
            nodule_id_b = f"{meta[i]['nodule_group_b']}_{meta[i]['exam_b']}_{meta[i]['exam_idx_b']}"

            shape_XYZ_b = tuple(
                int(round(n * s_old / s_new))
                for n, s_old, s_new in zip(shape_b, spacing_b, non_interpolated_pixel_spacing)
            )

            small_bbox_output = recover_small_bbox(bbox_b, shape_XYZ_b)

            small_bbox_output_interp = scale_bbox_after_interpolate(small_bbox_output)

            output_data[nodule_id_b] = {
                "bbox": [int(x) for x in small_bbox_output],
                "bbox_interp": [int(x) for x in small_bbox_output_interp],
                "nodule_volume": volumes_out[i].item()
            }

            if saved_examples < max_saved_examples:
                if saved_examples == 0:
                    "Print saving first example!!"
                save_montage_with_bbox3d(preped_input_images[i], out_path=f"./test_eval_segmentations/{saved_examples}_montage_in.png", bbox_yxz=small_bbox_input, title="Input Image with Nodule BBox No Interpolation")
                input_fig = save_montage_with_bbox3d(preped_input_images_inter[i], out_path=f"./test_eval_segmentations/{saved_examples}_montage_in_inter.png", bbox_yxz=small_bbox_input_interp, title="Input Image with Interpolation", return_fig=True)
                binary_seg_fig = save_montage(binary_segmentation_input.float()[i], out_path=f"./test_eval_segmentations/{saved_examples}_montage_in_seg.png", title="Input Segmentation", return_fig=True)
                save_two_figs_side_by_side(input_fig, binary_seg_fig, out_path=f"./test_eval_segmentations/{saved_examples}_montage_in_combined_vol_{volumes_in[i].item()}.png")
                # import pdb; pdb.set_trace()
                plt.close(input_fig)
                plt.close(binary_seg_fig)

                save_montage_with_bbox3d(preped_output_images[i], out_path=f"./test_eval_segmentations/{saved_examples}_montage_out.png", bbox_yxz=small_bbox_output, title="Output Image with Nodule BBox No Interpolation")
                output_fig = save_montage_with_bbox3d(preped_output_images_inter[i], out_path=f"./test_eval_segmentations/{saved_examples}_montage_out_inter.png", bbox_yxz=small_bbox_output_interp, title="Output Image with Interpolation", return_fig=True)
                binary_seg_fig_out = save_montage(binary_segmentation_output.float()[i], out_path=f"./test_eval_segmentations/{saved_examples}_montage_out_seg.png", title="Output Segmentation", return_fig=True)
                save_two_figs_side_by_side(output_fig, binary_seg_fig_out, out_path=f"./test_eval_segmentations/{saved_examples}_montage_out_combined_vol_{volumes_out[i].item()}.png")

                plt.close(output_fig)
                plt.close(binary_seg_fig_out)

                saved_examples += 1
            total += 1
            if total % 200 == 0:
                print(f"Processed {total} nodules...")

    with open("train_raw_data_nodule_original_boxes.json", "w") as f:
        json.dump(output_data, f)
