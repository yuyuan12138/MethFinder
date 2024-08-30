# MethFinder

- [MethFinder](#methfinder)
  - [Introduction](#introduction)
  - [Basic Dictionary](#basic-dictionary)
  - [Get Started](#get-started)
    - [Installation](#installation)
    - [Training the Model](#training-the-model)
    - [View Result](#view-result)
  - [TODO](#todo)

## Introduction

This repository contains code for “**MethFinder: A Novel Approach to DNA Methylation Prediction Using Adversarial-Specificity Convolutional and Specificity Transformer Techniques.**”

## Basic Dictionary

1. **combined_data**: The folder is used for storing data which trains our pretrained model.
2. **data**: Dataset in paper.
3. **pretrain-model**: Pretrained model, 4mC, 5hmC, 6mA, respectively.
4. **model**: Model Architecture.
5. **utils**: Some useful python functions.
6. **config.py**: Hyper-Parameters.
7. **data.py**: data preprocessing file.
8. **train.py**: data training file.

## Get Started

It is easy to reproduce our result just step by step.

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

   We use Pytorch==1.12.0+cu116, you can choose your version if necessary.

    ```bash
    pip install -r requirements.txt
    ```

### Training the Model

To train the model, run the `train.py` script. This script initializes the model, loads the data, sets the loss function, and begins the training process:

```bash
python train.py --data 4mC_C.equisetifolia --epochs 50 --batch_size 512 --learning_rate 1e-4
```

1. `--data/-d`: You can choose which dataset you want to train. `--data/-d ['4mC_C.equisetifolia', '4mC_F.vesca', '4mC_S.cerevisiae', '4mC_Tolypocladium', '5hmC_H.sapiens', '5hmC_M.musculus', '6mA_A.thaliana', '6mA_C.elegans', '6mA_C.equisetifolia', '6mA_D.melanogaster', '6mA_F.vesca', '6mA_H.sapiens', '6mA_R.chinensis', '6mA_S.cerevisiae', '6mA_T.thermophile', '6mA_Tolypocladium', '6mA_Xoc_BLS256']`
2. `--epochs/-ep`: You can choose how many epochs. `--epochs/-ep Num_of_Epochs`
3. `--batch_size/-bs`: Set batch size you want. `--batch_size/-bs Num_of_Batch_size`
4. `--learning_rate\-lr`: Set Learning Rate. `--learning_rate\-lr 1e-4`

### View Result

After training, there will generate a file `results.csv` which you can see the best performance in ACC, SN, SP, AUC, MCC and F1-Score that also printed on your console. Meanwhile, you can see the best trained-model in a folder **models**.

## TODO

Here is something should be upgraded.

- [ ] Add article link and weblink.
- [ ] Commit drawing functions.
