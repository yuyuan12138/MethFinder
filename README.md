# MethFinder

- [MethFinder](#methfinder)
  - [Introduction](#introduction)
  - [Directory Overview](#directory-overview)
    - [Pre-existing Directories and Files](#pre-existing-directories-and-files)
    - [Generated Files and Directories](#generated-files-and-directories)
  - [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Training the Model](#training-the-model)
      - [Command-Line Arguments](#command-line-arguments)
    - [Viewing Results](#viewing-results)
  - [Future Work](#future-work)

## Introduction

This repository contains the implementation code for **"MethFinder: A Novel Approach to DNA Methylation Prediction Using Adversarial-Specificity Convolutional and Specificity Transformer Techniques."** MethFinder is designed to predict DNA methylation sites with a focus on adversarial specificity and advanced machine learning models.

## Directory Overview

### Pre-existing Directories and Files

1. **`combined_data`**: Contains datasets used for training the pretrained models.
2. **`data`**: The main dataset used in the paper.
3. **`pretrain-model`**: Directory for pretrained models for different types of DNA methylation (4mC, 5hmC, 6mA).
4. **`model`**: Contains the model architecture code.
5. **`utils`**: Includes utility functions to facilitate various operations.
6. **`config.py`**: Configurations for hyperparameters.
7. **`data.py`**: Handles data preprocessing.
8. **`train.py`**: Script for training the models.

### Generated Files and Directories

1. `umap`: Contains visualizations of the dataset using Uniform Manifold Approximation and Projection (UMAP).
2. `acc_loss_plot`: Contains plots showing accuracy and loss over the training period.
3. `models`: Directory where the trained models are saved.
4. `results.csv`: A CSV file that stores the evaluation results, including performance metrics like ACC, SN, SP, AUC, MCC, and F1-Score.

## Getting Started

Follow these steps to reproduce the results from our paper.

### Installation

To set up the environment, follow these steps:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yuyuan12138/MethFinder.git
    cd MethFinder
    ```

2. **Create a virtual environment** (optional but recommended):

    ```bash
    conda create -n MethFinder python=3.9.12
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:

   This project uses PyTorch 1.12.0+cu116. You may choose a compatible version if necessary. Refer to the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) for more details.

    ```bash
    pip install -r requirements.txt
    ```

### Training the Model

To train the model, run the `train.py` script. This script initializes the model, loads the data, sets the loss function, and begins the training process:

```bash
python train.py --data 4mC_C.equisetifolia --epochs 50 --batch_size 512 --learning_rate 1e-4 --is_drawing_plot --is_umap
```

#### Command-Line Arguments

1. `--data/-d`: Choose the dataset to train on. Available options are: `['4mC_C.equisetifolia', '4mC_F.vesca', '4mC_S.cerevisiae', '4mC_Tolypocladium', '5hmC_H.sapiens', '5hmC_M.musculus', '6mA_A.thaliana', '6mA_C.elegans', '6mA_C.equisetifolia', '6mA_D.melanogaster', '6mA_F.vesca', '6mA_H.sapiens', '6mA_R.chinensis', '6mA_S.cerevisiae', '6mA_T.thermophile', '6mA_Tolypocladium', '6mA_Xoc_BLS256']`
2. `--epochs/-ep`: Set the number of epochs, e.g., `--epochs 50`.
3. `--batch_size/-bs`: Define the batch size, e.g., `--batch_size 512`.
4. `--learning_rate/-lr`: Set the learning rate, e.g., `--learning_rate 1e-4`.
5. `--is_drawing_plot/-dp`: Option to enable drawing of accuracy and loss plots during training.
6. `--is_umap/-iu`: Option to generate UMAP visualizations for the training data.

### Viewing Results

After training, the following results and files will be generated:

1. `results.csv`: Contains the best performance metrics, including ACC, SN, SP, AUC, MCC, and F1-Score. These results are also printed in the console.
2. `models` Directory: The directory where the best-trained models are saved for future use or evaluation.
3. `acc_loss_plot` Directory: Stores plots illustrating the accuracy and loss trends over the training period, helping in visualizing the model's performance.
4. `umap` Directory: Contains UMAP visualizations that represent high-dimensional data in a 2D space for better interpretation and analysis.

## Future Work

There are several improvements planned for this repository:

- [ ] Add a link to the article and web resources.
- [x] Implement and commit drawing functions for data visualization.
- [x] Upgrade README.md to include more details about `config.py`.
- [x] Upgrade `requirements.txt` to ensure compatibility with new dependencies.
