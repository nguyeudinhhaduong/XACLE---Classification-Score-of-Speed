# XACLE-AudioMOS: Audio Quality Prediction using Deep Learning

This repository implements a neural network model to predict **Mean
Opinion Score (MOS)** for audio clips using the **XACLE dataset**.\
The model is trained on audio `.wav` files and textual descriptions,
with MOS labels provided in CSV annotations.

------------------------------------------------------------------------

## 📂 Dataset Structure

Project follows the official XACLE dataset structure:

    xacle-dataset/
    │
    ├── XACLE_dataset_train_val/
    │   └── XACLE_dataset/
    │       ├── meta_data/
    │       │   ├── train.csv
    │       │   ├── train_average.csv
    │       │   ├── validation.csv
    │       │   └── validation_average.csv
    │       │
    │       └── wav/
    │           ├── train/
    │           └── validation/
    │
    └── XACLE_test_data/
        └── XACLE_test_data/

### 📄 Example CSV format (`train_average.csv`)

  ---------------------------------------------------------------------------------
  wav_file_name   text                                              average_score
  --------------- ------------------------------------------------- ---------------
  00000.wav       A water vehicle travels through the water with    8.0
                  wind noise...                                     

  00001.wav       A motorcycle drives by                            7.5

  00002.wav       Some liquid flows followed by something sink      7.5

  00003.wav       Rain falling with distant thunder roaring         9.25

  00004.wav       The propellers of a helicopter scream as someone  2.0
                  yells                                             

  00005.wav       Silence then suddenly a loud honk occurs...       7.75

  00006.wav       Some snapping and music, traffic passes           2.75
  ---------------------------------------------------------------------------------

------------------------------------------------------------------------

## 🧠 Model Overview

The model predicts **audio quality (MOS)** using:

-   Acoustic features extracted from raw audio (`torchaudio`)
-   Text embeddings for the audio captions (optional depending on
    config)
-   Regression head predicting a single MOS value

Loss functions include:

-   MSE Loss\
-   Ranking loss (SRCC optimization)
-   Correlation-based loss

Training includes:

-   Gradient accumulation\
-   Early stopping\
-   Best checkpoint saving

------------------------------------------------------------------------

## 🚀 Training Results

Below is the training log (30 epochs with early stopping):

    📊 EPOCH 1: Loss=0.4255 | SRCC=0.5702 | LCC=0.5897 | MSE=6.1402
    ✅ BEST SAVED
    ----------------------------------------
    📊 EPOCH 2: Loss=0.3638 | SRCC=0.6122 | LCC=0.6220 | MSE=7.0771
    ✅ BEST SAVED
    ----------------------------------------
    📊 EPOCH 3: Loss=0.3278 | SRCC=0.6085 | LCC=0.6259 | MSE=5.4378
    ⚠️ Patience: 1/6
    ----------------------------------------
    📊 EPOCH 4: Loss=0.2988 | SRCC=0.6123 | LCC=0.6363 | MSE=4.8074
    ✅ BEST SAVED
    ----------------------------------------
    📊 EPOCH 5: Loss=0.2800 | SRCC=0.6017 | LCC=06258 | MSE=4.0529
    ⚠️ Patience: 1/6
    ----------------------------------------
    📊 EPOCH 6: Loss=0.2576 | SRCC=0.6247 | LCC=0.6449 | MSE=3.9568
    ✅ BEST SAVED
    ----------------------------------------
    📊 EPOCH 7–12: No improvement → early stopping triggered
    ⏹️ EARLY STOPPING

### 🏁 **Best Performance:**

-   **SRCC:** 0.6247\
-   **LCC:** 0.6449\
-   **MSE:** 3.95

------------------------------------------------------------------------


------------------------------------------------------------------------

## 📊 Evaluation Metrics

  Metric     Description
  ---------- --------------------------------------------------------
  **SRCC**   Spearman Rank Correlation --- measures ranking quality
  **LCC**    Linear Correlation --- strength of linear relationship
  **MSE**    Regression error

------------------------------------------------------------------------

## 🏗️ Requirements

    torch
    torchaudio
    transformers
    pandas
    numpy
    tqdm
    scikit-learn

Install:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 📜 License

This project is for research purposes only.

------------------------------------------------------------------------

## ✨ Acknowledgement

Dataset: **XACLE: Cross-modal Audio Quality and Caption Evaluation**
