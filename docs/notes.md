# Notes

> Internal notes on data, models, and anything else.
> Not guaranteed to be up to date or exhaustive.

---

## 1. Data Notes

### 1.1 Data structure
- Original HU clipping is -1350, + 150
- My range -2000, 500 because contains 98% of the points. If I used -2000, 160 it would cover 90%

---

## 2. Other Notes
- You don’t need grad scalar if I am in fp32 only need in fp16 and not in bf16
- Final case noise added in flow matching should be 0 because I have exact paths
- Confidence model should be taking in a patch from the image (bounding box around the nodule) and outputting first score is yes and second score is no
- Bounding boxes are in the Y, X, Z order as provided.
- I think ants is (X, Y, Z) so I transpose it to (2, 1, 0) to get (Z, Y, X) for display
- I can try using the regular unet in flow matching if I want