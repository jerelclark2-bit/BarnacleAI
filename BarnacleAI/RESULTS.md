# BarnacleAI empirical findings — October 12, 2026

**Status: research prototype, not a validated scientific barnacle counter.**

## What the models actually do

1. A small PyTorch convolutional network is trained on image patches from the first photograph and annotation-derived center heatmaps. At inference, it produces a heatmap; local peaks above a threshold become proposed barnacle positions. It **does not** predict a statistical 95% count interval on its own.
2. Ridge regression is trained on engineered 128 x 128 image-feature vectors (brightness/density, intensity centroid, symmetry, left-right balance, edges, texture), predicting the number of blue-annotation-derived centers in a tile.
3. Both approaches are evaluated on photograph `img2`, held out from training. The image's incomplete blue outlines are **not** verified true population labels.
4. The interactive app displays predictions, highlights model disagreements, and accepts manual corrections. Feedback is stored for possible later retraining but **does not automatically retrain**.

## Experiments: held-out `img2`

| Measure | CNN 12 epochs, 110 train tiles | CNN 32 epochs, 64 train tiles |
|---|---:|---:|
| CNN proposed centers (entire cropped square) | 1,819 | **1,104** |
| Extracted blue annotation centers (incomplete) | 350 | 350 |
| CNN mean absolute tile difference vs annotation proxy | 24.39 | **12.57** |
| Ridge mean absolute tile difference vs annotation proxy | 14.85 | 14.85 |
| Fraction of annotation centers with a CNN point within 10 pixels | 55.1% | **62.3%** |
| Tiles flagged for review (disagreement heuristic) | 13/49 | 1/49 |
| CNN detection threshold selected from training photo | 0.55 | 0.75 |

**Important:** 62.3% is a spatial proximity statistic, **not** a one-to-one detection recall or biological accuracy. The initial CNN heatmap was nearly flat, with values near 0.55 across large areas. The 32-epoch model has a wider score distribution and better agreement with annotation proxies. We do not know how many unlabeled barnacles it detects or how many detections are false positives without hand-counted examples.

**Do not treat a peak score of 0.75 as a 75% chance that an object is a barnacle.** Scores are not calibrated probabilities.

The 49-tile evaluation grid excludes narrow right and bottom edge strips, while the CNN whole-crop count includes them. Use the same partition when comparing counts.

## 95% confidence interval: what is actually supported?

We cannot currently give a valid 95% interval for the **true number of barnacles** from model output alone. We built and tested an approximate **randomized sampling interval** method, which requires a person to count randomly selected image tiles without relying on blue annotations.

The image is partitioned into 64 nearly equal-area tiles that cover every pixel. Twelve tiles are sampled uniformly without replacement using random seed `20261012`. Let `N=64`, `n=12`, `y_i` be the manually verified number of barnacle centers in sampled tile `i`, and `m_i` be the CNN's predicted centers in tile `i` (available for all 64 tiles).

- **Direct estimator:** `T_hat = N * mean(y_sample)`.
- **CNN-assisted estimator:** `T_hat = sum(m_all) + N * mean(y_sample - m_sample)`. This is a design-based difference estimator and does not require the CNN to be unbiased.
- **Approximate standard error:** `SE = N * sqrt((1-n/N) * s_residual²/n)` (for direct sampling, the residuals are the human counts themselves).
- **Approximate 95% interval:** `[max(0,T_hat - t_(0.975,n-1)*SE), max(0,T_hat + t_(0.975,n-1)*SE)]`.

Under proper random sampling, accurate counts, and an adequate t-approximation, this estimates the **total count of the fixed pictured region**, not the generalization error of the CNN on future images. Strong clustering or sampling skew can affect small-sample interval coverage. The count convention is that each barnacle is assigned to the tile containing its **center**.

### Tested math: incomplete annotation proxies only (NOT true barnacles)

Replacing manual `y_i` with the provided blue annotation counts in 12 randomly selected tiles gives:

| Demonstration only | Estimated annotated-center count | Approximate 95% interval |
|---|---:|---:|
| Direct sampling | 341.33 | 238.45–444.22 |
| CNN-assisted sampling using **retrained CNN** | 272.00 | 94.49–449.51 |
| Whole-crop blue annotation count (for comparison) | 350 | — |

Both example intervals include the known annotation-proxy count of 350. The CNN-assisted interval is wider here, showing that the model currently does **not** improve this proxy counting estimator. **Neither example estimates the true biological count.**

### To obtain the real interval

1. Run `python scripts/make_review_sheet.py --photo img2`, which creates a blinded 12-tile review collage `results/random_manual_review_img2.png`.
2. Open `results/manual_counts_img2.csv` (generate it with `python scripts/manual_count_interval.py --photo img2`). Count each photographed tile's barnacle centers and fill in all 12 `manual_count` cells. Do not use the blue masks to count.
3. Run `python scripts/manual_count_interval.py --photo img2` again. It writes `results/count_interval_img2.json` with the two approximate 95% intervals based on the verified counts.
4. Alternatively, upload `img2.png` in the Streamlit app, choose the `DALI img2 central square` region, and use the randomized verification section. It will not display a biological interval until all 12 tiles are manually counted.
5. Report both how many regions were verified and how much time manual counting took. Do not substitute algorithmically generated numbers for independent human counts.

## Next scientific work

Collect more whole photographs with substantially more accurate center labels; evaluate one-to-one object matching (precision/recall) on independent human-labeled tiles; use larger and more diverse sample sizes to validate interval coverage; explore a pretrained image detection model as time permits. API and 3D display features do not substitute for scientific validation.
