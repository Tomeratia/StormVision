# StormVision - Stage 2 (YOLO Dataset Build + Baseline vs Fine-tuning)

StormVision is a Computer Vision project for **small-object detection in maritime/storm scenes**, focusing on detecting:
- **Persons** (includes swimmers)


Stage 2 focuses on:
- Converting COCO annotations to a **YOLO-ready dataset**
- Building a **paired ORIG vs SYNTH split** with leakage prevention (split by `pair_id`)
- Training a **fine-tuned YOLO model** (Model B) and comparing it against a **pretrained baseline** (Model A)
- Reporting detection metrics (**Precision, Recall, F1, mAP50**) and qualitative visual comparisons

---

## What’s in Stage 2

### Dataset (current snapshot)
- **Paired dataset** of images (ORIG + SYNTH) stored in a single folder:
  - `pair_<id>_ORIG_<orig_name>.jpg`
  - `pair_<id>_SYNTH_<synth_name>.jpg`
- Annotations:
  - Original COCO: `instances_train.json`, `instances_val.json`
  - Synthetic COCO: `instances_synth.json`

### Label space (2 classes)
The project maps multiple COCO categories into two training classes:
- `0: person` (includes `person`, `swimmer`)
- `1: boat` (includes `boat`, `jetski`, `buoy`, `life_saving_appliances`)

### Split strategy (leakage-safe)
To avoid data leakage, the split is done by **pair_id**, meaning ORIG and SYNTH images from the same pair always belong to the same split:
- Train: 70%
- Val: 15%
- Test: 15%

During dataset build, YOLO label files were created for:
- Train labels: **311**
- Val labels: **68**
- Test labels: **71**

> Note: Some images are background-only (no labels) and are still included in the dataset folders.

---

## Models

### Model A (baseline)
- **YOLOv8s pretrained** (`yolov8s.pt`)
- Used as a strong, off-the-shelf detector without domain adaptation

### Model B (fine-tuned)
- **YOLOv8s fine-tuned** on the StormVision 2-class dataset
- Training configuration (current run):
  - Epochs: 25
  - Image size: 1024 (chosen to help small-object detection)
  - Batch: 8
  - Early stopping patience: 10

Outputs are saved under:
- Runs: `output_runs_stormvision/Model_B_Final/`
- Best weights: `output_runs_stormvision/Model_B_Final/weights/best.pt`

---

## Evaluation setup

We evaluate on the **test split** in two scenarios:

1. **Real Data (ORIG-only)**  
   Test images filtered by filename containing `ORIG`

2. **Synthetic Data (SYNTH-only)**  
   Test images filtered by filename containing `SYNTH`

### Metrics
For each scenario, we compare Model A vs Model B using:
- **Precision**
- **Recall**
- **F1** (computed from Precision and Recall)
- **mAP50**

Evaluation is focused on:
- `classes=[0]` (Person-only), since the main research question is small-person detection

---

## Visual outputs (Stage 2)
Stage 2 includes two types of analysis:
1. **Quantitative comparison plots**  
   Bar charts comparing Model A vs Model B across ORIG vs SYNTH for mAP50, Precision, Recall, F1.

2. **Qualitative improvement examples**  
   Side-by-side detections for cases where Model B shows a significant confidence gain over Model A on the test set.

---

## Repository files (Stage 2)

### Notebook
- `notebooks/StormVision_Stage2_YOLO_Baseline_vs_Finetune.ipynb`  
  End-to-end workflow:
  - Build YOLO dataset from paired images + COCO annotations
  - Train Model B (fine-tuning)
  - Evaluate Model A vs Model B on ORIG-only vs SYNTH-only test subsets
  - Produce metric plots and qualitative improvement visualizations

### Inputs (expected paths)
- `storm_synth_out_onefolder/Combination/`  
  Flat paired images folder (ORIG + SYNTH mixed)
- `data/annotations/instances_train.json`  
  Original COCO train annotations
- `data/annotations/instances_val.json`  
  Original COCO val annotations
- `storm_synth_out_onefolder/instances_synth.json`  
  Synthetic COCO annotations

### Outputs (generated)
- `output_dataset_pairs/`  
  YOLO-formatted dataset:
  - `images/{train,val,test}/`
  - `labels/{train,val,test}/`
  - `dataset.yaml`
- `output_runs_stormvision/`  
  Training runs and weights (Ultralytics format)

---

## How to run (Colab)
1. Mount Google Drive.
2. Set `BASE_DIR` to your project root in Drive.
3. Run the cells in order:
   - Install + imports
   - Dataset build (`process_dataset`)
   - Training (M
