# StormVision (Stage 2)

## Short Background
StormVision develops GenAI-assisted person-in-water detection for rough-sea scenarios.  
Stage 2 focuses on exploratory analysis of the original SeaDroneSee data, synthetic image generation, paired dataset construction (original vs synthetic), and baseline vs fine-tuned YOLOv8 experiments.

## Files
- `eda_original_data.ipynb` - Exploratory data analysis and sanity checks on the original SeaDroneSee dataset (data structure, annotations, distributions, and QC).
- `Synthetic Data Generator.ipynb` - Generates synthetic images and exports corresponding annotations and metadata.
- `make_flat_pairs.py` - Builds a flat paired dataset by organizing original and synthetic images into a unified structure.
- `baseline vs finetune.ipynb` - End-to-end YOLOv8 pipeline including baseline training, fine-tuning, evaluation on real vs synthetic test subsets, and result visualization.

## Folders
- `data/` - Data and intermediate artifacts. Includes a pointer (URL) to the original SeaDroneSee dataset.
- `notebooks/` - Notebooks for EDA, generation, and evaluation.
- `src/` - Helper scripts.
- `results/` - Sample outputs (example prediction images).
- `presentation/` - Slides.
