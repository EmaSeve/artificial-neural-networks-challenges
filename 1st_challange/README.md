# AN2DL - Timeseries Classification 

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Framework](https://img.shields.io/badge/PyTorch-Deep_Learning-red)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

 
**Team:** Emanuele Severino, Vincenzo Del Grosso, Mattia Rusconi, Riccardo Scarabelli  
**Context:** Artificial Neural Networks and Deep Learning - First Challenge 

## 📄 Project Overview
This project addresses a **multiclass time-series classification** task aimed at predicting pain levels in subjects ("pirates").
The model utilizes multivariate temporal data, including joint movements, survey responses, and physical characteristics, to classify the subject's status into one of three categories: 
**No Pain, Low Pain, or High Pain**.

The solution implements a deep Recurrent Neural Network (RNN) pipeline that handles data imbalance and temporal dependencies, achieving significant improvements over the baseline.

## 📂 Repository Structure

* `an2dl_notebook_challenge.py`: The main Python script containing the full pipeline: data loading, preprocessing, model training, cross-validation, and inference.
* `report.pdf`: A detailed technical report describing the problem analysis, methodology, and experimental results.

## 📊 Dataset & Problem Analysis

The dataset presents several challenges addressed in this solution:
* **Input Data:** Time-series of 30 joint features (`joint_00` to `joint_29`), subjective pain surveys, and binary injury indicators.
* **Imbalance:** The data is heavily skewed toward the "No Pain" class, requiring specific handling strategies like class weighting.
* **Correlations:** High correlation exists between specific joint groups, while other features are sparse or constant.

## 🧠 Methodology

### 1. Preprocessing
* **Cleaning:** Removal of constant features (e.g., `joint_30`) and redundant anatomical descriptors.
* **Normalization:** Min-Max scaling is applied to continuous features to stabilize distributions.
* **Windowing:** Data is segmented into fixed-length sliding windows (`WINDOW_SIZE=20`) with a stride (`STRIDE=10`) to create sequence inputs for the RNN.

### 2. Model Architecture
The core model is a **RecurrentClassifier**.
* **Backbone:** Uses **GRU** (Gated Recurrent Unit) or LSTM layers to process temporal sequences.
* **Configuration:** * Hidden Size: 128
    * Dropout: 0.4
    * Bidirectional: True.
* **Classification Head:** The final hidden state is passed through an MLP (Linear, ReLU, Dropout) to produce class logits.

### 3. Training Strategy
* **Loss Function:** A weighted Cross-Entropy Loss is used to penalize errors on minority classes (Low/High Pain) more heavily.
* **Optimization:** The model is trained using the **AdamW** optimizer with mixed-precision support.
* **Cross-Validation:** A K-Shuffle-Split strategy partitions users into training and validation sets to ensure robust performance estimation.

### 4. Inference (Thresholding)
To further address class imbalance during inference, a custom thresholding scheme is used instead of a standard `argmax`. 
The model prioritizes minority classes if their predicted probability exceeds specific tuned thresholds ($t_c$), ensuring higher sensitivity to pain detection.

## 📈 Results

The final model configuration achieved the following performance on the validation set:

| Metric | Score |
| :--- | :--- |
| **Weighted F1**  |**~0.94**|
| **No Pain F1**   | 0.97 |
| **Low Pain F1**  | 0.91 |
| **High Pain F1** | 0.70 |

The approach significantly outperforms the majority-class baseline (F1 0.67).

## 🚀 Usage

### Prerequisites
The code requires Python and the following libraries:
```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn ydata-profiling kaggle