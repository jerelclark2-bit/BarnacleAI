# BarnacleAI 🐚

**DALI Data Challenge 26F** | Prototype for counting barnacles *inside the inner square of the teal frame*.

## Research question
Can a lightweight deep-learning detector and a hand-engineered regression model provide useful, auditable barnacle-count estimates, while identifying regions that merit human review?

## What this submission does

- Extracts approximate barnacle-center annotations from supplied blue contour masks. Annotations are imperfect/incomplete and **must not be equated with exact population ground truth**.
- Restricts analysis of `img1`/`img2` to manually inspected central-square crops. Coordinates are documented in `barnacleai/data.py`. Outer side squares and yellow ruler are excluded.
- Trains a **small PyTorch convolutional neural network from scratch** to output barnacle-center heatmaps. Unlike transfer learning, this does *not* require downloading pretrained weights. It is a pilot and may perform poorly on different lighting/species.
- Trains a separate **Ridge regression model** based on interpretable features: mean brightness (density), horizontal/vertical intensity center, symmetry, left-right and top-bottom balance, edge density, brightness variation, and local texture.
- Evaluates on a *different whole photograph*, to avoid tile leakage. Reports absolute per-tile errors versus annotation-derived proxy counts, not biological accuracy.
- Flags tiles where the neural and regression estimates disagree significantly. Such disagreement is a *heuristic human-review flag*, **not a statistically calibrated confidence interval**.
- Builds an interactive Streamlit dashboard showing overlaid detections, flagged regions and a 3D diagnostic (X/Y position, Z model disagreement). A scientist can select a tile, manually correct its count, and save that correction locally.
- Provides a working local FastAPI endpoint for predictions and feedback. Feedback is saved locally and does not automatically modify model weights.

## Data and provenance

Supplied DALI assets in `data/`: `img1.png`, `img2.png`, `mask1.png`, `mask2.png`, `unseen_img1.png`, and `unseen_img2.png`.

Masks are *training/evaluation aids only*. The scientists would not have them for new photos. `unseen_img2.png` is a close-up with no green frame and should not be interpreted as a whole-square count. `masked_img1/2.png` can be used for human interpretation but were not required for training because their coordinate systems differ from originals.

The sample contains only two labeled photographs, which drastically limits generalization claims. Our approximate extracted centers may merge adjacent annotations or miss unlabeled barnacles. All counts must be described as estimates.

## Quick start

Use Python 3.11+ in a virtual environment, from this project directory:

```bash
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m barnacleai.train --epochs 3 --max-train-tiles 75
```

Training outputs go to `results/`: `cnn.pt`, `ridge.joblib`, `metrics.json`, `heldout_overlay.png`, and `model_disagreement_3d.html`.

Run the interactive interface:

```bash
streamlit run app.py
```

Run the REST API in another terminal:

```bash
uvicorn api:app --reload
```

- `GET /health` for readiness
- `POST /predict` accepts `file` (image) and `roi` (`none`, `img1`, `img2`, `unseen_img1`)
- `POST /feedback` stores `image_id`, `x`, `y`, `corrected_count` to a local JSONL file
- API documentation at `http://127.0.0.1:8000/docs`

For `unseen_img1`, select its central-square crop if using the DALI framed image. For other image sources, crop manually first or specify `none`. Do **not** score unlabeled images as if they had known targets.



Let `w_ab` be grayscale brightness normalized to [0,1]. We calculate:

- **Density:** `D = mean(w)`.
- **Horizontal intensity center:** `mu_x = sum(x*w) / sum(w)` (scaled to [0,1]).
- **Vertical intensity center:** `mu_y = sum(y*w) / sum(w)`.
- **Symmetry:** `S = 1 - sum(abs(w - flip_left_right(w))) / (2 * sum(w))`.
- **Balance:** `B = (L-R) / (L+R)`.

The brightest regions do **not** necessarily correspond to barnacles, so these are descriptive features for the separate regression model, not individual-object labels. We add edge/texture features and standardize before fitting Ridge. For an actual regression count estimate, ridge generalizes the OLS solution by regularizing coefficients. More training images are necessary to assess this model reliably.

See [RESULTS.md](RESULTS.md) for the measured output of the included pilot training run.

## Scientific validation and failure analysis

- **Split by photograph, never random tiles.** Tiles from one photograph share lighting, rock texture, camera properties and organisms; mixing them into train/test would inflate accuracy.
- **No test-set tuning.** The CNN's peak threshold is selected using the training photograph only.
- **Proxy labels.** Connected blue annotation outlines are counted as imperfect centers; two touching contours may merge.
- **CNN heatmap peaks.** A probability peak is a proposed detection, not proof of a barnacle. Thresholds are not calibrated probabilities.
- **Regression target.** This model estimates annotations per 128 × 128 patch. It only reports sums over *complete evaluated tiles*, while the CNN predicts centers over the entire central crop.
- **Uncertainty.** Disagreement is a practical review flag. A genuine 95% count prediction interval would require additional, diverse *independent labeled photos* and appropriate calibration.
- **Manual scientist corrections.** Feedback is stored to permit future analysis, not instantly incorporated into an online model.

## Next experiments

Create independent hand-checked validation areas, capture more field photos, compare a pretrained CNN if weights are available, improve per-object labeling, and estimate scientist time saved. The UI and APIs are optional extensions and do not replace measured model quality.

## 95% intervals from randomized human verification (new)

We now have a **real statistical procedure** for estimating uncertainty in the fixed photo's total count, but it **needs manual counts**. The CNN's raw heatmap score and the difference between the CNN and regression predictions are **not** calibrated confidence intervals.

The central square is divided into 64 nearly equal-area tiles covering all pixels. Twelve are selected uniformly at random, using a published seed (`20261012`), and a person records how many barnacles have their centers in each selected tile. A finite-population, t-approximate interval estimates the square's total. A second *model-assisted* interval uses the CNN's full-grid predictions plus average manually observed CNN counting errors on sampled tiles. Both methods assume randomized selection and accurate manual counts. Details and equations are in [RESULTS.md](RESULTS.md).

**Get started without the web interface:**

```bash
python scripts/make_review_sheet.py --photo img2
python scripts/manual_count_interval.py --photo img2
# Inspect results/random_manual_review_img2.png; enter counts in
# results/manual_counts_img2.csv, leaving tile IDs untouched.
python scripts/manual_count_interval.py --photo img2
```

The CSV remains intentionally blank until a real person verifies the tiles. No biological 95% confidence interval can be claimed before that step. The file `results/sampling_demo_img2.json` demonstrates and tests the calculations using the **incomplete blue annotations only**; do not interpret its numerical interval as a barnacle-population confidence interval.

**Updated training result:** A second experiment with 32 epochs and 64 sampled training tiles produced 1,104 proposed held-out centers and a mean absolute per-128px-tile difference of 12.57 from incomplete blue annotations (earlier 12-epoch version: 1,819 and 24.39). This is an improvement against the proxy benchmark, **not validated biological performance**. The actual pretrained weights used in these experiments: none; the CNN is trained from scratch. View [RESULTS.md](RESULTS.md) for the comparison.
