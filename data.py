import torch
import torch.utils.data as Data
from config import config
import pandas as pd
import numpy as np

def get_dataloader(train_path=None, test_path=None) -> list:
    if train_path:
        train_data = pd.read_csv(train_path, sep='\t')
        print("train_data is :", train_data.shape)
        train_dataset = MyDataset(train_data['text'], train_data['label'])
        train_loader = Data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    else:
        train_loader = None

    if test_path:
        test_data = pd.read_csv(test_path, sep='\t')
        test_dataset = MyDataset(test_data['text'], test_data['label'])
        test_loader = Data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    else:
        test_loader = None

    return (train_loader, test_loader)

class MyDataset(Data.Dataset):
    def process_features(self, sequences):
        mapping = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        onehot_encoded_sequences = []
        
        for sequence in sequences:
            seq_numbers = [mapping[nucleotide] for nucleotide in sequence]
            seq_array = np.array(seq_numbers)
            onehot_encoded_array = np.eye(4)[seq_array]
            onehot_encoded_sequences.append(onehot_encoded_array)
        
        onehot_encoded_sequences_array = np.array(onehot_encoded_sequences)
        onehot_tensor = torch.tensor(onehot_encoded_sequences_array, dtype=torch.float32, device=config.device)
        
        return onehot_tensor

    def process_data(self, features, labels):
        features = self.process_features(features)
        labels = torch.tensor(labels, device=config.device, dtype=torch.long)
        return features, labels

    def __init__(self, features:list, labels:list) -> None:
        super().__init__()
        self.features, self.labels = self.process_data(features, labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return (self.features[index], self.labels[index])

# ----------------------- Load unlabeled data for prediction -----------------------

class MyDatasetTest(Data.Dataset):
    """
    Dataset class for prediction (no labels).
    Input TSV should contain only one column: 'text'
    """
    def __init__(self, sequences):
        super().__init__()
        self.sequences = sequences
        self.features = self.process_features(sequences)

    def process_features(self, sequences):
        mapping = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        onehot_encoded_sequences = []

        for sequence in sequences:
            seq_numbers = [mapping.get(n, 0) for n in sequence]
            seq_array = np.array(seq_numbers)
            onehot_encoded_array = np.eye(4)[seq_array]
            onehot_encoded_sequences.append(onehot_encoded_array)

        onehot_encoded_sequences_array = np.array(onehot_encoded_sequences)
        onehot_tensor = torch.tensor(
            onehot_encoded_sequences_array, dtype=torch.float32, device=config.device
        )
        return onehot_tensor

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        # Returns the original sequence and its corresponding one-hot encoded tensor
        return self.sequences[index], self.features[index]


def get_test_dataloader(test_path, batch_size=1024):
    """
    Create a DataLoader for unlabeled prediction data (one-column .tsv).
    """
    test_data = pd.read_csv(test_path, sep='\t')

    # Automatically identifies column names or defaults to using the first column
    if 'text' in test_data.columns:
        sequences = test_data['text']
    else:
        sequences = test_data.iloc[:, 0]

    print(f"[INFO] Loaded {len(sequences)} sequences for prediction.")
    test_dataset = MyDatasetTest(sequences)
    test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


