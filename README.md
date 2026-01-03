# StormVision (Stage 2)

GenAI-based person-in-water detection in rough seas.

## What this stage covers
- EDA on the original SeaDroneSee dataset
- Synthetic image generation + metadata export
- Building a paired ORIG vs SYNTH flat dataset
- YOLOv8 baseline vs fine-tuning evaluation on real vs synthetic test subsets

## Files
- `eda_original_data.ipynb` - EDA and sanity checks on the original dataset.
- `Synthetic Data Generator.ipynb` - Generates synthetic images and exports annotations + metadata.
- `make_flat_pairs.py` - Creates a flat paired dataset folder by copying ORIG + SYNTH images.
- `baseline vs finetune.ipynb` - End-to-end YOLO pipeline: baseline vs fine-tune, evaluation and plots.

## Folders
- `data/` - data and intermediate artifacts
- `notebooks/` - notebooks for EDA, generation, and evaluation
- `src/` - helper scripts
- `results/` - exported metrics and figures
- `presentation/` - slides
