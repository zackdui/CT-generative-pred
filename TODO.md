# TODO

## Data
- [x] Create initial metadata
- [ ] Check same Z value when loading a pair after processing full volume
- [ ] Test apply transform in processing
- [ ] Have registration add on to a file of bad niftis and add the ones we already passed
- [ ] Check 2D autoencoding in HU space and then bringing it back after for visualization
  - [ ] Check a scan with cancer and see if it generates the cancer
- [ ] Write code to get and visualize bounding boxes around tumors so I can check them
  - [ ] Add a dataloader for padded regions around tumors
- [ ] Crop Loading, load crops around tumors and then pad them
- [ ] Use the given segmentations for testing. Try to load them and visualize them
  - [ ] Use the volume
  - [ ] Use the coordinates. See if my mapping matches up

---

## Model
- [ ] Fix config file for model hyperparams
- [ ] Fix config file for model training 3d

---

## 3D VAE Tests
- [ ] Simpler first upsample
- [ ] Deconv as first upsample layer
- [ ] Group norm back
- [ ] Fix loss functions
- [ ] Stride 1 for VAE down

---

## Evaluation
- [ ] Ask peter to show me a nodule to test on
- [ ] Ask about confidence scores and how that works
---

## Experiments
- [ ] 2D
 - [ ] Pretraining
    - [ ] Try with regular images
    - [ ] Full run on encoded images
 - [ ] For some try with and without time component and compare results
 - [ ] Full paired test no pretraining encoded images
 - [ ] Full paired test no pretraining regular images
 - [ ] Full paired test pretraining encoded images
 - [ ] Full paired test pretraining regular images
 - [ ] Full paired test pretraining encoded images: Scans with nodules only
 - [ ] Full paired test pretraining regular images: Scans with nodules only
- [ ] 3D


---

## Infrastructure
- [ ] Fix the __init__.py files
- [ ] Allow for local access to 3D_Vae and downloaded package

---

## Docs
- [ ] Write `README.md` with setup + quickstart
- [ ] Document data preprocessing steps
- [ ] Describe experiment naming convention
- [ ] Describe full repo layout
- [ ] Describe all parquet files
- [ ] Add mamba env create -f environment.yml to read me
- [ ] Add citation of the original unet files
- [ ] Document that sorted paths is a json
- [ ] List of all top level packages I actually installed

---

## Backup Ideas
- [ ] Video style generation where you generate slice by slice but condition on previous few slices
- [ ] Biomed parse is 2D parsing

## Questions
- [ ] Should segmentation take 20 mins on a single image or something is wrong
- [ ] Need to understand nodule mapping from segmentation to original image
 - [ ] Did the exams have pixel resampling done? registration?
 - [ ] How should I compare nodules in this case from my generated to the original?
 - [ ] What are the units of volume


For benny: 'volume': 106.787109375, 'exam': '20808922492012730', Patient ID: 208089
Only tracked nodule in this exam



