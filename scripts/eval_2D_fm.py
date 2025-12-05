# model = UNetModel(
    #     image_size=512,
    #     in_channels=1,           # grayscale images
    #     model_channels=64,       # base number of channels
    #     out_channels=1,          # same as in_channels, for image generation
    #     channel_mult=(1, 2, 4, 8),
    #     use_fp16=False,
    #     use_checkpoint=True,
    #     num_res_blocks=2,
    #     attention_resolutions=[32, 16],  
    # )
    # test_model = UnetLightning.load_from_checkpoint(best_path, model=model)
    # trainer.test(model=test_model, dataloaders=test_loader) #, ckpt_path="best")
##################################

