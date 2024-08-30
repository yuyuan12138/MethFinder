# MethFinder

This repository contains a deep learning-based framework called for predicting DNA methylation sites using a custom neural network architecture. The model utilizes a combination of location-specific 1D convolutions and transformers to improve the prediction accuracy of DNA methylation sites from sequence data.

## Table of Contents

- [MethFinder](#methfinder)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Model Architecture](#model-architecture)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Training the Model](#training-the-model)

## Introduction

DNA methylation is an important epigenetic modification that affects gene expression and plays a critical role in numerous biological processes. Predicting DNA methylation sites computationally can help in understanding these biological processes and their underlying mechanisms. This project implements a deep learning model designed to predict DNA methylation sites from genomic sequences.

## Model Architecture

The model utilizes a custom architecture that combines:
- **Location-Specific 1D Convolutions**: A novel `Conv1d_location_specific` module that performs convolutions in a location-specific manner, allowing for more targeted feature extraction.
- **Transformers**: Transformers are utilized to capture long-range dependencies in the sequence data.
- **Custom Neural Network (`Net`)**: The main model architecture incorporates these modules to predict methylation sites from DNA sequence data.

The training process involves optimizing several evaluation metrics such as Accuracy (ACC), Sensitivity (SN), Specificity (SP), Area Under the Curve (AUC), Matthews Correlation Coefficient (MCC), Precision, and F1-Score.

## Installation

To set up the environment, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/DNA-Methylation-Prediction.git
    cd DNA-Methylation-Prediction
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

To train the model, run the `main.py` script. This script initializes the model, loads the data, sets the loss function, and begins the training process:

```bash
python main.py
```

<style>
p {
    font-family: "Times New Roman", serif;
}
</style>
