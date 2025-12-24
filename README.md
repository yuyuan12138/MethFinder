# MethFinder

- [MethFinder](#methfinder)
  - [Introduction](#introduction)
  - [Directory Overview](#directory-overview)
    - [Pre-existing Directories and Files](#pre-existing-directories-and-files)
    - [Generated Files and Directories](#generated-files-and-directories)
  - [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Eproducing the results from the MethFinder paper](#Eproducing the results from the MethFinder paper)
    - [Using the MethFinder tool for methylation prediction](#Using the MethFinder tool for methylation prediction)
    - [Applying the MethFinder framework for custom training and prediction tasks](#Applying the MethFinder framework for custom training and prediction tasks)
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

This section details the usage and operation of the MethFinder framework.

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

### Eproducing the results from the MethFinder paper

To reproduce the results reported in the paper, make sure the following files exist in the main directory:

`Transformer_weight.py`, `model.py`, `config.py`, `Conv1d_weight.py`, `data.py`, and `reproduce_results.py`.

Then simply run:

```bash
python reproduce_results.py
```

This will automatically create a folder named `reproduce_results/`, which contains two `.csv` files:

* `human_reproduce.csv`: Reproduction of the first-stage results (methylation prediction).
* `cancer_reproduce.csv`: Reproduction of the second-stage results (cancer-specific methylation prediction).


### Using the MethFinder tool for methylation prediction

To use MethFinder for methylation prediction, ensure the main directory contains:

`models/`, `Transformer_weight.py`, `model.py`, `config.py`, `Conv1d_weight.py`, `data.py`, and `predict.py`.

**Standard usage**:

```bash
python predict.py -step 1 -test xxx.tsv -output xxxx
```

Parameter descriptions；

* `-step`: specifies which MethFinder module to use:
  
  `-step 1` for genome-wide methylation prediction under a cancer background,
  
  `-step 2` for identifying cancer-type-specific driver methylation events.

* `-test`: specifies the path to your test file.
  
  The test file must be in `.tsv` format with a single column named `text`.
  
  You can refer to `test_example.tsv` in the main directory for reference.

* `-output`: specifies the name of the output file.

After execution, the prediction results will be saved in the `./preds/` directory.
  
### Applying the MethFinder framework for custom training and prediction tasks

To train and predict on your own datasets using the MethFinder framework, make sure the main directory contains:

`models/`, `Datasets/`, `Transformer_weight.py`, `model.py`, `config.py`, `Conv1d_weight.py`, `data.py`, `predict.py`, and `train.py`.

**Step 1 :Prepare your dataset**

Inside the `Datasets/` directory, create a new folder (e.g., `Mydataset`).

Split your data into training and test sets, and save them as `train.tsv` and `test.tsv` respectively.

You can refer to the existing datasets under `Datasets/` as examples.

**Step 2: Train the model**

Run the following command:

```bash
python train.py -data Mydataset -th_metric MCC
```

**Parameters:**

* `-data`: specifies the dataset folder name (the one you created).
* `-th_metric`: specifies which metric to use for binary classification threshold selection.

After training, a model file A.pth will be saved in `./models/`,

and the random seed used during training will be saved in `./seeds/`.

**Step 3: Predict using your trained model**

```bash
python predict.py -model Mydataset -test xxx.tsv -output xxxx
```

**Parameters:**

* `-model`: specifies which trained model to use for prediction.

The prediction results will be saved in the `./preds/` directory.

  ***Note:*** The `-model` and `-step` parameters in `predict.py` cannot be used simultaneously.

  For more details on parameter settings, please refer to `config.py`.

## Future Work

There are several improvements planned for this repository:

- [ ] Add a link to the article and web resources.
- [x] Upgrade README.md to include more details about `config.py`.
- [x] Upgrade `requirements.txt` to ensure compatibility with new dependencies.
