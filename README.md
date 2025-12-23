# 🌊 StormVision: Synthetic Augmentation for Maritime SAR
> **Interim Phase (Stage 2)**: Dataset Transformation, EDA, and Baseline Implementation.

## 📌 Project Overview
The goal of this project is to improve object detection (persons and boats) in maritime Search and Rescue (SAR) scenarios under adverse weather conditions. This phase focuses on using Generative AI to bridge the "Domain Gap" found in standard maritime datasets.

## 📂 Source Dataset: SeaDronesSee
The foundation of this project is the **SeaDronesSee** dataset:
* **Origin**: Large-scale visual object detection dataset captured from UAVs (drones).
* **Initial State**: Primarily "Fair Weather" conditions (clear skies, calm sea).
* **Challenge**: Standard models trained on this data fail when faced with storms or low visibility.

## 🛠️ Data Generation Pipeline (The Hybrid Approach)
We developed a pipeline to transform the **SeaDronesSee** images into realistic stormy scenarios:
* **Targeted Inpainting**: Uses `StableDiffusionInpaintPipeline` with automated masks to protect existing objects while regenerating the environment.
* **Structural Preservation**: Integrated **ControlNet (Canny)** to lock the scene's geometry, ensuring horizons and object scales remain intact.


## 📊 Dataset Composition & EDA
Analysis performed on the original SeaDronesSee data:
* **Tiny Objects**: Targets (swimmers) occupy **<1%** of the image pixels, making detection difficult.
* **Background Dominance**: Images are **95%+ water**, requiring the model to filter significant environmental noise.


## 🤖 Baseline Implementation
* **Model**: **YOLOv8s** (Small) - optimized for real-time drone inference.
* **Weights**: Pre-trained on **COCO** (Zero-shot performance on storm data).
* **Role**: Acts as a control group to quantify performance drops in adverse weather.

## 📈 Evaluation Setup
* **Metrics**: **mAP@0.5** and **Recall** (Crucial for SAR to minimize missed victims).
* **Output**: Bounding Box + Confidence score, evaluated as **Person vs. No-Person** presence.

