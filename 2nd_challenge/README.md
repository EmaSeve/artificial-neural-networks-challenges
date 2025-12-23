# AN2DL - Histopathological Molecular Subtype Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

**Team:** Emanuele Severino, Vincenzo Del Grosso, Mattia Rusconi, Riccardo Scarabelli  
**Context:** Artificial Neural Networks and Deep Learning - Second Challenge 

## 📄 Project Overview

This repository contains the source code and documentation for the **Artificial Neural Networks and Deep Learning (AN2DL)** Second Challenge. The project focuses on the analysis of histological microscopic images of diseased human tissue—associated with binary masks identifying diseased regions—to predict four molecular subtypes: **Luminal A, Luminal B, HER2+, and Triple Negative**. To achieve this, we explored Convolutional Neural Networks (CNNs) with transfer learning and a One-vs-Rest (OvR) XGBoost approach.

## 📂 Repository Structure

* `efficientnet.py`: Contains the baseline experiments using the EfficientNet architecture.
* `resnet18_efficencenet.py`: The primary training pipeline. It includes data loading, the multiscale centroid-based tiling strategy, the ResNet-18 model definition (frozen and fine-tuned), Test-Time Augmentation (TTA), and Grad-CAM visualization.
* `ovr_xgboost.py`: Implementation of the One-vs-Rest strategy combined with XGBoost for slide-level classification.
* `report.pdf`: The detailed project report describing the methodology, experiments, and results.

## 🔬 Methodology

### 1. Preprocessing and Data Cleaning
* **Mask Refinement:** Morphological operations were applied to the masks to improve region contiguity.
* **Tumor-Centered Tiling:** To extract informative regions, we computed the centroid of connected mask regions and cropped fixed-size patches centered at these locations.
* **Multiscale Strategy:** Tiles were extracted at different zoom levels around each centroid to provide complementary local and contextual views.

### 2. Model Architectures
We evaluated and compared the following architectures:
* **ResNet-18 (Best Performing):** A ResNet-18 architecture pretrained on ImageNet was adopted as the backbone. The original fully connected layer was replaced with a custom classifier head composed of two linear layers separated by a ReLU activation and Dropout.
* **EfficientNet:** Used as a baseline model due to its simplicity compared to other architectures.
* **One-vs-Rest XGBoost:** Four independent binary classifiers were trained to distinguish one molecular subtype against all others. Their outputs were aggregated at the slide level to train an XGBoost model.

### 3. Training Strategy
* **Class Imbalance:** A weighted cross-entropy loss was adopted, where weights were computed as the inverse frequency of training tiles per class.
* **Fine-Tuning:** The backbone was initially frozen to train only the classifier head, and subsequently unfrozen after a fixed number of epochs to fine-tune.
* **Augmentation:** We applied extensive augmentation including random 90-degree rotations, flips, color jittering, Gaussian blur, and random erasing.

## 📊 Results

All models were evaluated using the F1-score under identical experimental conditions.

| Model | F1-Score | Notes |
| :--- | :--- | :--- |
| **ResNet-18** | **0.4403** | **Best Performance**  |
| EfficientNet | 0.3602 | Baseline  |
| One-vs-Rest (XGBoost) | 0.3406 |  |
| Random Classifier | 0.2500 |  |

## 🧠 Interpretability

Grad-CAM was used to inspect model attention at different network depths on both training and validation data.

## 📚 References

1.  M. I. Jaber *et al.*, "A deep learning image-based intrinsic molecular subtype classifier of breast tumors...", *Breast Cancer Research*, 2020.
2.  N. Shvetsov *et al.*, "Deep learning-based classification of breast cancer molecular subtypes...", *Journal of Pathology Informatics*, 2025.