# StormVision Dataset Structure

This directory contains the training and validation data for the StormVision project.
The dataset is utilized for training YOLOv8 models to detect maritime objects in stormy sea conditions.

## Folder Structure

```text
data/
├── train/                  # Real training images (SeaDronesSee)
│   └── images/
├── val/                    # Real validation images (SeaDronesSee)
│   └── images/
│
├── synthetic/              # Generated stormy images
│   ├── train/
│   │   ├── images/         # Synthetic versions of training set
│   │   └── train_manifest.csv
│   └── test/
│       ├── images/         # Synthetic versions of validation set (for testing)
│       └── test_manifest.csv
│
├── processed/              # YOLO training files
│   ├── labels/             # All label files (.txt)
│   │   ├── train/          # Labels for real training images
│   │   ├── val/            # Labels for real validation images
│   │   ├── synthetic_train/# Labels for synthetic training images
│   │   └── synthetic_test/ # Labels for synthetic test images
│   │
│   ├── splits/             # Text files defining image lists for training
│   │   ├── train_mixed.txt
│   │   ├── test_mixed.txt
│   │   └── ...
│   │
│   └── eval_*.yaml         # YOLO configuration files
│
└── annotations/            # Original COCO format annotations (JSON)
```

## Details

*   **Real Images**: Sourced from the SeaDronesSee dataset.
*   **Synthetic Images**: Generated using SDXL + ControlNet based on the real images.
*   **Storage**: Due to size, the labels for training are stored centrally in `processed/labels` and mapped via the `.yaml` and `.txt` split files during training.

## Classes

*   `0`: swimmer
*   `1`: boat
*   `2`: jetski
*   `3`: life_saving_appliances
*   `4`: buoy

## How to Use

To train a model (e.g., the Mixed model), point the YOLO training command to the appropriate YAML file in the `processed/` directory (e.g., `processed/data_mixed.yaml` if available, or generate it using the notebooks).
