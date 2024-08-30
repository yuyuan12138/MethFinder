# MethFinder

## Table of Contents

- [MethFinder](#methfinder)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Basic Dictionary](#basic-dictionary)
  - [Get Started](#get-started)
    - [Installation](#installation)
  - [Usage](#usage)
    - [Training the Model](#training-the-model)

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

## Usage

### Training the Model

To train the model, run the `main.py` script. This script initializes the model, loads the data, sets the loss function, and begins the training process:

```bash
python main.py
```
