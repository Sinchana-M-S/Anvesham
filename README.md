# DataSprint 2.0 | CV Track — Road Damage Detection with YOLO26m

## 1. Problem Understanding & Objective
Road agencies inspect thousands of kilometres of road manually. This project builds an automated road damage detector capable of running on ordinary smartphone or dash-camera photos to prioritise repairs before small cracks become dangerous potholes.

* **Task:** Detect and classify road damage in street photos.
* **Classes:**
  * `0`: `D00` — Longitudinal crack
  * `1`: `D10` — Transverse crack
  * `2`: `D20` — Alligator crack
  * `3`: `D40` — Pothole
* **Evaluation Criteria:** Accuracy (mAP@50, mAP@50:95, precision, recall), inference speed (target < 25 ms per image on a T4 GPU), and robustness on degraded photos.

## 2. Repository Structure
```text
.
├── road_damage_final.ipynb     # End-to-end audit, evaluation, robustness, and submission notebook
├── inference.py                # Model inference script
├── benchmark_speed.py          # Latency and throughput benchmarking script
├── score.py                    # Official competition evaluation scorer
├── make_kaggle_submission.py   # Formats test predictions into Kaggle submission CSV
├── check_submission.py         # Validates submission CSV format and image IDs
├── weights/
│   └── best.pt                 # Trained YOLO26m weights (SHA-256 verified)
├── submission.csv              # Final test set predictions
├── requirements.txt            # Python dependencies
├── writeup.pdf                 # Two-page project write-up
└── README.md                   # Project documentation
```

## 3. Environment & Setup
Install the required dependencies (exact Ultralytics version pinned to match the notebook):

```bash
pip install -r requirements.txt
```

Verify the trained weights integrity before running evaluation or inference:
* **Expected SHA-256 (`weights/best.pt`):** `9f9fd2b5d51818759337daff442acdb8101ec2018be89b1deb31331d3cfde9a4`

## 4. Dataset Overview & Data Audit
The solution uses the official RDD2020 competition splits:
* **Train:** 14,691 images
* **Validation:** 2,150 images
* **Test:** 4,200 images (containing both normal and degraded photos)

The notebook performs a full data audit across all splits:
* **Image integrity:** Verifies readability and dimensions of every image with `PIL`.
* **Duplicate detection:** Computes MD5 hashes to check for exact duplicates within splits and data leakage across `train-val` and `train-test`.
* **Class & box distribution:** Evaluates class imbalance, boxes per damaged image, and relative bounding box sizes (`tiny < 1%`, `small 1–5%`, `medium 5–20%`, `large > 20%`).

## 5. Data Cleaning
Before training, bounding box annotations are validated and cleaned using the following rules:
* **Unknown class IDs:** Filtered to keep only valid classes (`0–3`).
* **Out-of-bounds coordinates:** Clipped to `[0, width]` and `[0, height]` so normalised YOLO coordinates remain within `[0, 1]`.
* **Zero-area boxes:** Dropped (`width < 1 px` or `height < 1 px`).
* **Exact duplicate boxes:** Deduplicated on `(image_id, class_id, x_min, y_min, x_max, y_max)`.
* **Background images (no damage):** Explicitly retained to teach the detector not to fire on clean roads.

## 6. Preprocessing & Augmentation
Training hyperparameters are logged inside the `weights/best.pt` checkpoint (`train_args`). Key preprocessing and augmentation stages include:
* **Letterbox resize:** Preserves aspect ratio with padding to avoid stretching thin cracks.
* **Normalisation:** Scales pixels to `0–1` in RGB order to match COCO pretraining.
* **Mosaic & Close Mosaic:** Stitches 4 images to improve small-object recall, then disables mosaic in the final epochs so the model converges on natural-looking images.
* **Geometric & Colour Augmentations:** Random scale, translation, horizontal flip (`fliplr`), and HSV colour jitter (`hsv_h`, `hsv_s`, `hsv_v`) to build invariance to camera distance, orientation, and lighting conditions.

## 7. Model Selection & Architecture
* **Chosen Model:** Single **Ultralytics YOLO26m** detector.
* **Rationale:**
  * Two-stage detectors (e.g., Faster R-CNN) exceed the 25 ms T4 latency target.
  * Smaller one-stage models (YOLO n/s) struggle with thin, low-contrast cracks.
  * Large models, test-time augmentation (TTA), and ensembles hurt the speed component (10% of the final score).
  * YOLO26m provides the best accuracy-to-latency trade-off within the T4 speed budget.

## 8. Training & Pretrained Weights
* **Pretrained Weights Used:** `yolo26m.pt, COCO` (generic COCO weights only; no external road, pavement, or crack datasets used).
* **Training Data:** Trained exclusively on the competition dataset.
* **Checkpoint Metadata:** Exact training arguments (`imgsz`, `epochs`, `batch`, `optimizer`, `lr0`, `cos_lr`, etc.) and best epoch are inspected directly from `weights/best.pt` in Section 4 of the notebook.

## 9. Validation & Official Scoring
Validation predictions are generated and scored using the official competition scripts:

```bash
python benchmark_speed.py --repo . --images <path/to/val_images> --out-csv val_pred.csv --out-json val_speed.json --warmup 20 --runs 2
python score.py --pred val_pred.csv --gt <path/to/val_labels.csv> --image-ids <path/to/val_image_ids.csv>
```

* **Validation Results:**
  * mAP@50: `[Insert from notebook output]`
  * mAP@50:95: `[Insert from notebook output]`
  * Precision / Recall: `[Insert from notebook output]`
  * Official Accuracy Score: `[Insert from notebook output]`

## 10. Speed Benchmark
Inference speed is measured on an NVIDIA T4 GPU using `benchmark_speed.py` (with warmup runs enabled):
* **Median Latency:** `[Insert median_ms]` ms/image (Target: < 25 ms)
* **P95 Latency:** `[Insert p95_ms]` ms/image
* **Median FPS:** `[Insert fps_median]`
* **Speed Score:** `[Insert speed_score]`
* **Peak GPU Memory:** `[Insert peak_gpu_mem_mb]` MB

## 11. Robustness Evaluation (Degraded Photos)
Because the test set includes degraded imagery, the notebook evaluates the model on a synthetically degraded copy of the validation set (seeded with `random.Random(0)`), randomly applying one of five corruptions per image:
1. **Gaussian Blur** (`sigma` in `1.0–2.5`)
2. **Gaussian Noise** (`std` in `8–20`)
3. **Heavy JPEG Compression** (quality `10–25`)
4. **Low Light / Darkening** (brightness factor `0.35–0.60`)
5. **Low Resolution** (downscale `0.3–0.5` and upsample back)

Clean vs. degraded metrics and retention percentages (`kept %`) are computed with `score.py` in Section 7 of the notebook.

## 12. Test Inference, Submission & Limitations
To reproduce the final test submission:

```bash
python benchmark_speed.py --repo . --images <path/to/test_images> --out-csv test_pred.csv --out-json test_speed.json --warmup 10 --runs 1
python make_kaggle_submission.py --pred test_pred.csv --image-ids <path/to/test_image_ids.csv> --out submission.csv
python check_submission.py --pred submission.csv --image-ids <path/to/test_image_ids.csv>
```

**Limitations & Future Scope:**
* **Thin Cracks:** Longitudinal (`D00`) and transverse (`D10`) cracks are thin and low-contrast, scoring lower than potholes and alligator cracks.
* **Localisation Strictness:** `mAP50:95` is lower than `mAP50` due to ambiguous crack endpoints.
* **Degraded Imagery:** Performance drops under heavy blur, darkness, and JPEG compression.
* **Next Steps:** Higher input resolution or image tiling for thin cracks, degradation-aware training augmentations, INT8/TensorRT export for edge deployment, and GPS tagging for road-health mapping.