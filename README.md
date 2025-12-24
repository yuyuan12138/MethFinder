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

This repository contains the implementation code for **"MethFinder: a DNA-sequence decoder pinpointing cancer-type-specific driver methylation."** MethFinder is a deep learning framework designed to directly predict cancer-type-specific driver DNA methylation from local DNA sequence patterns. It consists of two core modules: the first captures the intrinsic methylation potential of a sequence, while the second evaluates its driver methylation potential in the context of a specific cancer type.

## Directory Overview

### Pre-existing Directories and Files

1. **`Datasets`**: It includes a cancer methylation dataset folder (`cancer_meth`) used for training the model, along with folders of driver methylation datasets for 16 cancer types, each named with the cancer abbreviation (e.g., `BLCA`). Every folder contains both a `train.tsv` and a `test.tsv` file..
2. **`models`**: The folder stores the pre-trained model parameters provided in the study. Each file is named according to its corresponding dataset, such as `cancer_meth.pth`, `BLCA.pth`, and so on. User-trained models will also be automatically saved to this folder in the future.
3. **`Conv1d_weight`**: One of the key components of the MethFinder model.
4. **`model`**: Contains the MethFinder architecture code.
5. **`Transformer_weight`**: One of the key components of the MethFinder model.
6. **`config.py`**: Configurations for hyperparameters.
7. **`data.py`**: Handles data preprocessing.
8. **`train.py`**: Script for training the models.
9. **`predict.py`**: A predictor capable of loading and executing pre-trained models.
10. **`reproduce_reslts.py`**: A script designed for one-click reproduction of all results reported in the paper.
11. **`environment.txt`**: It includes the required environment dependencies for running MethFinder.

### Generated Files and Directories

1. **`seeds`**: MethFinder uses a random seed for each retraining session, and the corresponding training logs are stored in this folder..
2. **`preds`**: All prediction results will be saved in this folder.
3. **`reproduce_results`**: All reproduced results will be saved in this folder.

***`Note`***: These directories are auto-generated upon running the code.

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
