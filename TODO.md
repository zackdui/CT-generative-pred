# TODO


## Currently Working On

Add aditional eval on the finished model for VAE
finish the 3D flow matching training script for unpaired and paired training
Need a wandb eval for 3D flow matching training script

Set up EMA weights for inference

Remove 2D from process.md doc if never run it

## Full Files to write
- [ ] 3D FM pretraining script
- [ ] 3D FM paired training script
- [ ] 2D FM paired script
- [ ] 2D evaluation script
- [ ] 3D evaluation script
- [ ] 2D VAE finetuning script
- [ ] 3D autencoder eval which also should be worked on in VAE to update to the training script

## Data

- [ ] Check same Z value when loading a pair after processing full volume
- [ ] Check reconstruction quality of an actual nodule from autoencoders in 2D
- [ ] Check 2D autoencoding in HU space and then bringing it back after for visualization

---

## Model
- [ ] If now it is needed because of HU values finetune the 2D VAE on HU value images
- [ ] For the 2D model put the metrics with the outputs directory
- [ ] Save more checkpoints for the 2D model during training
- [ ] Try making the loss more specifically around the nodule regions
- [ ] Make charts as I go along the 2D training experiments to track the results

---

## 3D VAE Improvements and Tests

---

## Evaluation

---

## Experiments
- [ ] 2D
 - [ ] Pretraining
    - [ ] Try with regular images
    - [ ] Full run on encoded images
 - [ ] For some try with and without time component and compare results
  - [ ] Time component concatenatet
  - [ ] Time component added
 - [ ] Full paired test no pretraining encoded images
 - [ ] Full paired test no pretraining regular images
 - [ ] Full paired test pretraining encoded images
 - [ ] Full paired test pretraining regular images
 - [ ] Full paired test pretraining encoded images: Scans with nodules only
 - [ ] Full paired test pretraining regular images: Scans with nodules only
 - [ ] Start from gaussian conditioned on the image
 - [ ] Start from the image directly
 - [ ] Run model on bounding boxes around the nodules 
    - [ ] starting from pretrained on full lungs and pretrained on just segments
 - [ ] Try smaller architectures. The paper had a smaller architecture I can use
- [ ] 3D
- [ ] Try different clipping ranges for HU values and update padding
- Try with and without autoencoding for 3D small patches around nodules

---

## Docs

---

## Backup Ideas
- [ ] If regular 3D train doesn’t work try video type generation where you condition on the previous few slices but still only generate one 2D slice at a time
- [ ] Biomed parse is 2D parsing

## Questions

## Evaluation Metrics
- [ ] Compare ground truth to generation
- [ ] Compare generation to previous timepoint


## Backlog Ideas
- [ ] Better ODE solver
- [ ] Try a siren model
- [ ] detach tensors after every step
- [ ] 16 bit mixed precision requires custom implementation
- [ ] Replace flow matching library
- [ ] Replace lightning
- [ ] Use .github files from sybil in my autencoder repo to update package
- [ ] setup.cfg and setup.py
- [ ] Add in lpips loss to VAE's but might not help because not using visual loss



