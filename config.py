import torch
import argparse

class Config():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-data', default='4mC_C.equisetifolia')
        parser.add_argument('-epochs', default=200)
        parser.add_argument('-bsize', default=512)
        parser.add_argument('-lr', default=1e-4)
        parser.add_argument('-loop', default=114514)
        parser.add_argument('-combined', default=False)
        parser.add_argument('-model', default='4mC_dzy.pth')
        args = parser.parse_args()
        self.epochs = args.epochs if type(args.epochs) != str else eval(args.epochs)
        self.lr = args.lr
        self.batch_size = args.bsize
        self.data_name = args.data
        self.data_path = './combined_data/'
        self.loop = args.loop
        self.combined = args.combined
        self.model = args.model

config = Config()