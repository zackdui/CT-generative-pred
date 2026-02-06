import sys
import torch
from torchview import draw_graph
from monai.networks.nets import DiffusionModelUNet

# For local use of vae3d2d module
sys.path.insert(0, "/data/rbg/users/duitz/VAE3d/src")
from vae3d2d import CustomVAE, AttnParams

from CTFM.utils import load_config

def save_encoder_image(encoder_3d_configs):
    act = tuple(encoder_3d_configs.act)
    attn_params = AttnParams(**encoder_3d_configs.attn_params)

    model_name = encoder_3d_configs.model_name

    encoder_model = CustomVAE(
        model_name=model_name[:-3],
        fixed_std=encoder_3d_configs.fixed_std,
        num_groups=encoder_3d_configs.num_groups,
        in_channels=encoder_3d_configs.in_channels,
        dropout_prob=encoder_3d_configs.dropout_prob,
        spatial_dims=3,
        vae_latent_channels=encoder_3d_configs.vae_latent_channels,
        smallest_filters=encoder_3d_configs.smallest_filters,
        debug_mode=encoder_3d_configs.debug_mode,
        act=act,
        upsample_mode=encoder_3d_configs.upsample_mode,
        init_filters=encoder_3d_configs.init_filters,
        res_block_weight=encoder_3d_configs.res_block_weight,
        beta=encoder_3d_configs.beta,
        vae_use_log_var=encoder_3d_configs.vae_use_log_var,
        blocks_down=tuple(encoder_3d_configs.blocks_down),
        blocks_up=tuple(encoder_3d_configs.blocks_up),
        use_attn=encoder_3d_configs.use_attn,
        attn_params=attn_params,
        use_checkpoint=encoder_3d_configs.use_checkpoint,
        custom_losses=['mse'],
        downsample_strides=encoder_3d_configs.downsample_strides,
        vae_down_stride=encoder_3d_configs.vae_down_stride,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_model.to(device).eval()
    x = torch.randn(1, 1, 32, 128, 128).to(device)
    # t = torch.randint(0, 1000, (1,))

    g = draw_graph(
        encoder_model,
        input_data=(x),
        expand_nested=True,
        depth=2,       # bump up for more detail
        roll=True,      # makes wide UNets readable
        show_shapes=True,    # 🔑 removes input/output sizes
        graph_name="Vae 3D Architecture"
    )
    g.visual_graph.graph_attr.update(rankdir="LR")
    g.visual_graph.graph_attr.update(nodesep="0.15", ranksep="0.25")
    g.visual_graph.render("vae_unet_torchviewshape", format="svg", cleanup=True)  # SVG is great for thesis
   
    # torch.onnx.export(
    #     encoder_model,
    #     (x,),                      # or (x, t, cond, ...)
    #     "model.onnx",
    #     input_names=["x"],
    #     output_names=["y"],
    #     opset_version=17,
    #     do_constant_folding=True,
    #     dynamic_axes={"x": {0: "B"}, "y": {0: "B"}},
    # )

def save_unet_model_image(unet_3d_cfm_configs):
    unet_kwargs = dict(
            spatial_dims=unet_3d_cfm_configs.spatial_dims,
            in_channels=unet_3d_cfm_configs.in_channels,
            out_channels=unet_3d_cfm_configs.out_channels,
            num_res_blocks=unet_3d_cfm_configs.num_res_blocks,
            channels=unet_3d_cfm_configs.channels,
            resblock_updown=unet_3d_cfm_configs.resblock_updown,
            attention_levels=unet_3d_cfm_configs.attention_levels,
            num_head_channels=unet_3d_cfm_configs.num_head_channels,
            norm_num_groups=unet_3d_cfm_configs.norm_num_groups,
            use_flash_attention=True,
        )
    mod = DiffusionModelUNet(**unet_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod.to(device).eval()
    
    x = torch.randn(1, 16, 16, 16, 16).to(device)
    # t = torch.randn(1).to(device)
    t = torch.randint(low=0, high=1000, size=(x.shape[0],), device=device, dtype=torch.long)

    g2 = draw_graph(
        mod,
        input_data=(x, t),
        expand_nested=True,
        depth=1,       # bump up for more detail
        roll=True,     # makes wide UNets readable
        show_shapes=True,
        hide_inner_tensors=True
    )
    g2.visual_graph.graph_attr.update(rankdir="LR")
    g2.visual_graph.render("cfm_unet_torchviewshape", format="svg", cleanup=True)  # SVG is great for thesis

if __name__ == "__main__":
    vae_3d_model_yaml = "configs/vae_3d_model.yaml"
    encoder_3d_configs = load_config(vae_3d_model_yaml)
    save_encoder_image(encoder_3d_configs)

    unet_3d_cfm = "configs/unet_3d_cfm.yaml"
    unet_3d_cfm_configs = load_config(unet_3d_cfm)
    save_unet_model_image(unet_3d_cfm_configs)



    