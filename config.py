import torch
import argparse

class Config():
    def __init__(self):
        # Initialize argument parser
        parser = argparse.ArgumentParser(description="The configuration of MethFinder.")

        parser.add_argument('--seed', '-s', default=114514, type=int)
        
        # Dataset argument: Choose from a list of available datasets
        parser.add_argument('--data', '-d', default='4mC_C.equisetifolia', type=str, 
                            choices=['4mC_C.equisetifolia', '4mC_F.vesca', '4mC_S.cerevisiae', '4mC_Tolypocladium', 
                                     '5hmC_H.sapiens', '5hmC_M.musculus', '6mA_A.thaliana', '6mA_C.elegans', 
                                     '6mA_C.equisetifolia', '6mA_D.melanogaster', '6mA_F.vesca', '6mA_H.sapiens', 
                                     '6mA_R.chinensis', '6mA_S.cerevisiae', '6mA_T.thermophile', 
                                     '6mA_Tolypocladium', '6mA_Xoc_BLS256'])

        # Training configuration arguments
        parser.add_argument('--epochs', '-ep', default=100, type=int, help='Number of training epochs.')
        parser.add_argument('--batch_size', '-bs', default=512, type=int, help='Batch size for training.')
        parser.add_argument('--learning_rate', '-lr', default=1e-4, type=float, help='Learning rate for optimizer.')
         
        
        # Device configuration argument: Use GPU if available
        parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, 
                            help='Device to use for training (cuda or cpu).')
        
        # Plotting flag argument
        parser.add_argument('--is_drawing_plot', '-dp', action='store_true', 
                            help='Flag to indicate whether to draw plots. Default is False.')
        
        # Umap flag argument
        parser.add_argument('--is_umap', '-iu', action='store_true',
                            help='Flag to indicate whether to draw umap. Default is False.')

        '''Below parameters are used for testing purposes only; you don't need to use them.'''
        # Test arguments (used for debugging or testing purposes)
        parser.add_argument('--loop', '-l', default=114514, type=int, help='Loop parameter for testing.')
        parser.add_argument('--combined', '-c', default=False, type=bool, help='Combined flag for testing.')
        parser.add_argument('--model', '-m', default='4mC_dzy.pth', type=str, help='Pre-trained model name for testing.')

        # Parse all arguments
        args = parser.parse_args()

        # Assign parsed arguments to class variables
        self.epochs = args.epochs if type(args.epochs) != str else eval(args.epochs)  # Convert epochs to int if passed as string
        self.lr = args.learning_rate
        self.batch_size = args.batch_size
        self.data_name = args.data
        self.data_path = './combined_data/'  # Hardcoded path for combined data
        self.loop = args.loop
        self.combined = args.combined
        self.model = args.model
        self.device = args.device
        self.is_drawing_plot = args.is_drawing_plot
        self.is_umap = args.is_umap
        self.seed = args.seed

# Create a config object
config = Config()
