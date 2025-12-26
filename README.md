# StormVision - Stage 2 (EDA + YOLO Baseline vs Fine-tuning)

Stage 2 covers:
- EDA on the original SeaDroneSee dataset
- Synthetic image generation + metadata export
- Building a paired ORIG vs SYNTH flat dataset
- YOLOv8 baseline vs fine-tuning evaluation on Real vs Synthetic test subsets

---

## Files

- `eda_original_data.ipynb`  
  EDA and sanity checks on the original SeaDroneSee dataset (image-level summaries such as `has_person`, `persons_count`, and bbox-related stats).

- `Synthetic Data Generator.ipynb`  
  Generates synthetic images and exports:
  - synthetic images folder
  - `instances_synth.json` (COCO annotations for synthetic images)
  - `meta_synth.csv` (mapping between original and synthetic filenames)

- `make_flat_pairs.py`  
  Creates a flat paired dataset folder by copying ORIG + SYNTH images


- `baseline vs finetune.ipynb`  
  End-to-end pipeline:
  - prepares a YOLO-ready dataset from the paired images + COCO annotations
  - trains a fine-tuned YOLOv8 model
  - compares against a pretrained YOLOv8 baseline
  - evaluates on ORIG-only vs SYNTH-only test subsets and reports Precision/Recall/F1/mAP50 with plots
