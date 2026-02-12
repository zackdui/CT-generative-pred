import json
import os
from monai.networks.nets import DiffusionModelUNet
from torch.utils.data import DataLoader
import torch
import wandb
import matplotlib.pyplot as plt

# For local use of vae3d2d module
# import sys
# sys.path.insert(0, "/data/rbg/users/duitz/VAE3d/src")
from vae3d2d import CustomVAE

from CTFM.utils import load_config
from CTFM.data import (CachedNoduleDataset, 
                       collate_image_meta, 
                       save_montage, 
                       save_two_figs_side_by_side)
from CTFM.models import UnetLightning3D
from CTFM.eval import (load_segmentation_models, 
                       ImageEvaluatorPrep, 
                       patch_segmenter, 
                       get_volumes, 
                       save_montage_with_bbox3d,
                       dice_score)



if __name__=="__main__":
    ## Make sure to check path to index and directory when deciding on which data should be used
    ## Variables
    # fm_model_checkpoint = "/data/rbg/users/duitz/CT-generative-pred/final_saved_models/val-epoch=32-val_loss=0.02.ckpt"
    # fm_model_checkpoint = "experiments/fm_3d_pretrain/full_latent/lightning_logs/version_1/checkpoints/val-epoch=429-val_loss=0.32.ckpt"
    fm_model_checkpoint = "/data/rbg/users/duitz/CT-generative-pred/experiments/fm_3d_paired/test_no_pretrain_single_pair/lightning_logs/version_4/checkpoints/val-epoch=452-val_loss=0.13.ckpt"
    vae_checkpoint = "/data/rbg/users/duitz/CT-generative-pred/final_saved_models/vae_fixed_std_no_reg.pt"
    batch_size = 2
    pixel_spacing_non_interpolate = [0.703125, 0.703125, 2.5]
    pixel_spacing_interpolate = [0.703125 / 2, 0.703125 / 2, 2.5]
    max_dataset_size=None
    max_examples = 5
    location = "/data/rbg/users/duitz/CT-generative-pred/experiments/fm_3d_paired/results/one"

    os.makedirs(location, exist_ok=True)

    ## Set up wandb
    wandb.init(
        project="lung-ct-eval",
        name="flow_matching_eval_one",
        job_type="eval",
        config={
            "fm_model_checkpoint": fm_model_checkpoint,
            "vae_checkpoint": vae_checkpoint,
            "batch_size": batch_size,
            "pixel_spacing_non_interpolate": pixel_spacing_non_interpolate,
            "pixel_spacing_interpolate": pixel_spacing_interpolate,
            "max_examples": max_examples
        }
    )

    path_yaml = "configs/paths.yaml"

    ## Load Dataset
    base_paths = load_config(path_yaml)
    full_test_parquet = base_paths.full_data_test_parquet
    full_nodule_parquet = base_paths.bounding_boxes_test_parquet
    raw_nodule_index = base_paths.encoded_nodule_test_index
    paired_nodule_parquet = base_paths.paired_nodules_test_parquet
    data_root = base_paths.encoded_nodule_test_dir
    small_bboxes_path = base_paths.test_small_bboxes_encoded

    test_dataset = CachedNoduleDataset(full_test_parquet, 
                                       full_nodule_parquet, 
                                       raw_nodule_index, 
                                       data_root, 
                                       paired_nodule_parquet=paired_nodule_parquet,
                                       mode="paired",
                                       split="test",
                                       max_length=max_dataset_size,
                                       max_cache_size=50,
                                       return_meta_data=True)
    
    # test_dataset = CachedNoduleDataset(base_paths.full_data_val_parquet, 
    #                                    base_paths.bounding_boxes_val_parquet, 
    #                                    "/data/rbg/scratch/nlst_nodule_encoded_cache/meta/index.parquet", 
    #                                    "/data/rbg/scratch/nlst_nodule_encoded_cache", 
    #                                    paired_nodule_parquet=base_paths.paired_nodules_val_parquet,
    #                                    mode="paired",
    #                                    split="val",
    #                                    max_length=max_dataset_size,
    #                                    max_cache_size=50,
    #                                    return_meta_data=True)
    
    ## Load Models
    vae_model = CustomVAE.load_from_checkpoint(vae_checkpoint, map_location="cuda")
    # import torch

    # ckpt = torch.load(fm_model_checkpoint, map_location="cpu")

    # hparams = ckpt.get("hyper_parameters", {})  # most common
    # # Sometimes it's nested / different:
    # # hparams = ckpt.get("hparams", ckpt.get("hyper_parameters", {}))
    # print(hparams)

    # # This is what your __init__ wants:
    # model_hparams = hparams["model_hyperparameters"]

    # # Recreate the module with the right required arg(s)
    # lit = UnetLightning3D(
    #     model_hyperparameters=model_hparams,
    #     unet_cls=DiffusionModelUNet,
    # )

    # # Lightning checkpoints usually store model weights here:
    # state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # model_sd = lit.state_dict()

    # shape_mismatch = []
    # for k, v in state.items():
    #     if k in model_sd and tuple(v.shape) != tuple(model_sd[k].shape):
    #         shape_mismatch.append((k, tuple(v.shape), tuple(model_sd[k].shape)))

    # print("=== SHAPE MISMATCH ===")
    # for k, s_ckpt, s_model in shape_mismatch:
    #     print(k, s_ckpt, "->", s_model)

    # # Compare + load
    # missing, unexpected = lit.load_state_dict(state, strict=False)

    # print("=== MISSING (model expected, not in ckpt) ===")
    # for k in missing:
    #     print(k)

    # print("\n=== UNEXPECTED (in ckpt, not used by model) ===")
    # for k in unexpected:
    #     print(k)

    # import pdb; pdb.set_trace()
    lit = UnetLightning3D.load_from_checkpoint(
                                                fm_model_checkpoint,
                                                unet_cls=DiffusionModelUNet,
                                                strict=False,   
                                            )
    seg_model, confidence_model = load_segmentation_models()

    ## Set all models to cuda and eval
    vae_model.eval().to("cuda")
    lit.eval().to("cuda")
    seg_model.eval().to("cuda")
    confidence_model.eval().to("cuda")

    ## Run Evaluation
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 2, drop_last=False, collate_fn=collate_image_meta)
    evaluator_prep_interp = ImageEvaluatorPrep(vae_model, lit, n_steps=100, reverse_input=False, interpolate=True)
    evaluator_prep_no_interp = ImageEvaluatorPrep(vae_model, lit, n_steps=100, reverse_input=False, interpolate=False)

    ## Load Bboxes
    with open(small_bboxes_path, "r") as f:
        bboxes = json.load(f)

    examples_saved = 0
    metrics_table = wandb.Table(columns=["index", "nodule_a_id", "nodule_b_id", "Input Volume", "Output Volume", "Generated Volume", "Dice Score"])
    total = 0

    print(len(test_loader))

    for batch in test_loader:
        images, meta = batch
        
        input_images, output_images = torch.chunk(images, 2, dim=1)
        
        input_images, output_images = input_images.to("cuda"), output_images.to("cuda")

        ## Prepare all the images. There is some repeated work but I will leave it for now
        sampled_input_images_interp = evaluator_prep_interp.prepare_images_for_eval(input_images, sample=True)
        preped_input_images_interp = evaluator_prep_interp.prepare_images_for_eval(input_images, sample=False)
        preped_output_images_interp = evaluator_prep_interp.prepare_images_for_eval(output_images, sample=False)
        sampled_input_images_no_interp = evaluator_prep_no_interp.prepare_images_for_eval(input_images, sample=True)
        preped_input_images_no_interp = evaluator_prep_no_interp.prepare_images_for_eval(input_images, sample=False)
        preped_output_images_no_interp = evaluator_prep_no_interp.prepare_images_for_eval(output_images, sample=False)

        
        binary_segmentation_generated, confidence_scores_generated = patch_segmenter(sampled_input_images_interp, seg_model, lungmask_patch_tensor=None, confidence_model=confidence_model) 
        # binary_segmentation_input, confidence_scores_input = patch_segmenter(preped_input_images_interp, seg_model, lungmask_patch_tensor=None, confidence_model=confidence_model)
        binary_segmentation_output, confidence_scores_output = patch_segmenter(preped_output_images_interp, seg_model, lungmask_patch_tensor=None, confidence_model=confidence_model)

        volumes_generated = get_volumes(binary_segmentation_generated, pixel_spacing=pixel_spacing_interpolate)
        
        # import pdb; pdb.set_trace()
        table = wandb.Table(columns=["id", "Input", "Output", "Generated"])

        input_bboxes = []
        output_bboxes = []
        generated_bbox = []
        input_volumes = []
        output_volumes = []
        for i in range(len(meta)):
            total += 1
            nodule_id_a = f"{meta[i]['nodule_group_a']}_{meta[i]['exam_a']}_{meta[i]['exam_idx_a']}"
            nodule_id_b = f"{meta[i]['nodule_group_b']}_{meta[i]['exam_b']}_{meta[i]['exam_idx_b']}"

            saved_input_box_info = bboxes[nodule_id_a]
            saved_output_box_info = bboxes[nodule_id_b]

            input_bbox_interp = saved_input_box_info["bbox_interp"]
            input_bbox_no_interp = saved_input_box_info["bbox"]
            input_volume_saved = saved_input_box_info["nodule_volume"]

            input_bboxes.append(input_bbox_interp)
            input_volumes.append(input_volume_saved)

            output_bbox_interp = saved_output_box_info["bbox_interp"]
            output_bbox_no_interp = saved_output_box_info["bbox"]
            output_volume_saved = saved_output_box_info["nodule_volume"]

            output_bboxes.append(output_bbox_interp)
            output_volumes.append(output_volume_saved)

            if binary_segmentation_generated[i].sum() == 0:
                gen_box = None
            else:
                z, y, x = torch.where(binary_segmentation_generated[i])
                gen_box = (x.min().item(), x.max().item(), y.min().item(), y.max().item(), z.min().item(), z.max().item())

            generated_bbox.append(gen_box)

            dice_scores = dice_score(binary_segmentation_generated[i], binary_segmentation_output[i])

            metrics_table.add_data(total, nodule_id_a, nodule_id_a, input_volumes[i], output_volumes[i], volumes_generated[i], dice_scores)

            if examples_saved < max_examples:

                fig_input = save_montage_with_bbox3d(preped_input_images_interp[i], out_path="", save_fig=False, return_fig=True) #, bbox_yxz=input_bboxes[i])
                fig_output = save_montage_with_bbox3d(preped_output_images_interp[i], out_path="", save_fig=False, return_fig=True) #, bbox_yxz=output_bboxes[i])
                fig_generated = save_montage_with_bbox3d(sampled_input_images_interp[i], out_path="", save_fig=False, return_fig=True, bbox_yxz=generated_bbox[i])

                fig_input_montage = save_montage(preped_input_images_interp[i], out_path="", title=f"Input Fig {total}", save_fig=False, return_fig=True)
                fig_output_montage = save_montage(preped_output_images_interp[i], out_path="", title=f"Output Fig {total}", save_fig=False, return_fig=True)
                fig_generated_montage = save_montage(sampled_input_images_no_interp[i], out_path="", title=f"Generated Fig {total}", save_fig=False, return_fig=True)

                save_two_figs_side_by_side(fig_input, fig_output, out_path=f"{location}/{total}_input_output.png", title="Input vs Real Output", subtitle_1="Input", subtitle_2="Output")
                save_two_figs_side_by_side(fig_output, fig_generated, out_path=f"{location}/{total}_output_generated.png", title="Real Output vs Generated Output", subtitle_1="Given Output", subtitle_2="Generated Output")
                save_two_figs_side_by_side(fig_input, fig_generated, out_path=f"{location}/{total}_input_generated.png", title="Input vs Generated Output", subtitle_1="Input", subtitle_2="Generated Output")

                wandb.log({
                    "input_output": wandb.Image(f"{location}/{total}_input_output.png"),
                    "output_generated": wandb.Image(f"{location}/{total}_output_generated.png"),
                    "input_generated": wandb.Image(f"{location}/{total}_input_generated.png"),
                    "input": wandb.Image(fig_input),
                    "output": wandb.Image(fig_output),
                    "generated": wandb.Image(fig_generated),
                    "input_no_box": wandb.Image(fig_input_montage),
                    "output_no_box": wandb.Image(fig_output_montage),
                    "generated_no_box": wandb.Image(fig_generated_montage),
                })

                table.add_data(examples_saved, wandb.Image(fig_input), wandb.Image(fig_output), wandb.Image(fig_generated))

                plt.close(fig_input)
                plt.close(fig_output)
                plt.close(fig_generated)

                examples_saved += 1
        wandb.log({"eval_samples": table})

    wandb.log({"metrics": metrics_table})
    artifact = wandb.Artifact("volumes_table", type="metrics")
    artifact.add(metrics_table, "volumes_table")
    wandb.log_artifact(artifact)
