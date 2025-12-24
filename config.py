"""
Configuration module for MethFinder.

This file defines the Config class, which stores all hyperparameters and paths
needed for training and evaluation. The configuration is initialized via
command-line arguments, with default values provided for each parameter.

Note:
- Do NOT change parameter values here if you want to ensure compatibility with
  previously trained models.
"""

import torch
import argparse
import os

class Config():
    """
    Configuration class for model training and evaluation.

    Attributes:
        device (torch.device): Device used for training (default: 'cuda:0').
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        batch_size (int): Mini-batch size.
        data_name (str): Dataset name (e.g., BRCA).
        data_path (str): Path to the dataset folder.
        loop (int): Number of training loops or repetitions.
        model (str): Model name or path.
        threshold (float): Decision threshold for classification.
        seed (int): The seed for training model.
    """

    # Fixed device assignment
    # device = torch.device('cuda:0')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        parser = argparse.ArgumentParser()

        # Command-line arguments with default values
        parser.add_argument('-data', default='human', help="Dataset name")
        parser.add_argument('-epochs', default=200, help="Number of training epochs")
        parser.add_argument('-bsize', default=1024, help="Batch size")
        parser.add_argument('-lr', default=1e-4, help="Learning rate")
        parser.add_argument('-loop', default=0, help="Number of training loops")
        parser.add_argument('-combined', default=False)
        parser.add_argument('-model', default='None', help="Model name or path")
        parser.add_argument('-threshold', type=float, default=0.5,
                            help="Decision threshold for classification")
        parser.add_argument("-test", default='None', help="Path to test dataset (.tsv)")
        parser.add_argument("-output", default="prediction.csv", help="Output CSV path")
        parser.add_argument('-seed', type=int, default=None,
                            help="The seed for training model")
        parser.add_argument("-th_metric", default="MCC",
                            choices=["MCC", "AP", "AUC", "F1", "Precision", "Recall"],
                            help="Metric used to select best threshold during validation")
        parser.add_argument('-step', type=int, default=1,
                            help="1: predict methylation, 2: predict cancer types")
        args = parser.parse_args()

        # Store parameters
        self.epochs = args.epochs
        self.lr = args.lr
        self.batch_size = args.bsize
        self.data_name = args.data
        self.data_path = 'Datasets/'
        self.loop = args.loop
        self.combined = args.combined
        self.model = './models/' + args.model + '.pth'
        self.test = args.test
        self.threshold = args.threshold
        self.seed = args.seed
        self.th_metric = args.th_metric
        self.step = args.step
        self.output = os.path.join("./preds", f"{args.output}.csv")


# Global config instance
config = Config()


