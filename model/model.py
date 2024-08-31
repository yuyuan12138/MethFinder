import torch
import torch.nn as nn
from .Conv1d_weight import Conv1d_location_specific as Spec_Conv1d
from .Transformer_weight import Transformer_Different_Attention_Encoder as Spec_transformerEncoder


class Net(nn.Module):
    def __init__(self, in_channels, out_channels, use_spectic_conv1d=False, use_spectic_transformer=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initialize Conv1d layer, using a location-specific convolution if specified
        if use_spectic_conv1d:
            self.conv1D = Spec_Conv1d(in_channels=4, out_channels=64, kernel_size=3, padding=1, stride=1, weight_learning=True)
        else:
            self.conv1D = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1, stride=1)

        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

        # Conv2D layer for processing sequences after Conv1d
        self.conv2D = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 3), padding=(0, 1))
        self.bn_2 = nn.BatchNorm1d(64)

        # Second Conv1d layer, again supporting location-specific convolutions
        if use_spectic_conv1d:
            self.conv1D_2 = Spec_Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, weight_learning=True)
        else:
            self.conv1D_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

        self.bn_3 = nn.BatchNorm1d(64)

        # Transformer encoder for sequence modeling, with or without specific attention mechanism
        if use_spectic_transformer:
            self.transformer = Spec_transformerEncoder(d_model=64, norm_in_channels=64, N=1, dim_feedforward=8, nhead=2, weight=True)
        else:
            self.transformer = nn.Sequential(
                nn.TransformerEncoderLayer(d_model=64, nhead=2, dim_feedforward=8, batch_first=True),
            )

        self.dropout = nn.Dropout(0.5)
        self.flat = nn.Flatten()

        # Fully connected layer for final classification
        self.fc = nn.Sequential(
            nn.Linear(1312 * 2, 64),  # Input size 1312*2; adjust this if needed
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, self.out_channels),
        )

        # Flags to determine the use of specific Conv1d or Transformer layers
        self.use_spectic_conv1d = use_spectic_conv1d
        self.use_spectic_transformerEncoder = use_spectic_transformer

    def forward(self, x):
        # Transpose to match the expected input format for Conv1d
        x = x.transpose(1, 2)

        # Create a flipped copy of the input for bidirectional processing
        y = torch.flip(x, dims=[0])

        # First Conv1d layer, with location-specific options
        if self.use_spectic_conv1d:
            x = self.conv1D(x, non_important_site=10, important_site=31)
        else:
            x = self.conv1D(x)

        # Same Conv1d operation on the flipped input
        if self.use_spectic_conv1d:
            y = self.conv1D(y, non_important_site=10, important_site=31)
        else:
            y = self.conv1D(y)

        # Batch normalization and activation for both directions
        x = self.bn(x)
        y = self.bn(y)
        x = self.relu(x)
        y = self.relu(y)

        # Stack the forward and flipped outputs for 2D convolution
        x = torch.stack([x, y], dim=1)
        x = x.transpose(1, 2)  # Adjust dimensions for Conv2D

        # 2D convolution
        x = self.conv2D(x)
        x = x.squeeze(2)  # Squeeze to reduce unnecessary dimensions

        # Batch normalization and activation
        x = self.bn_2(x)
        x = self.relu(x)

        # Second Conv1d layer with optional location-specific convolutions
        if self.use_spectic_conv1d:
            x = self.conv1D_2(x, non_important_site=10, important_site=31)
        else:
            x = self.conv1D_2(x)

        # Batch normalization and activation
        x = self.bn_3(x)
        x = self.relu(x)
        x = x.transpose(1, 2)  # Adjust for transformer input

        # Transformer encoder for capturing long-term dependencies
        x = self.transformer(x)

        # Dropout regularization
        x = self.dropout(x)

        # Flatten the output for the fully connected layers
        x = self.flat(x)

        # Final fully connected layers for classification
        umap_features = x  
        x = self.fc(x)

        # Return output and attention probabilities from the transformer (if specific transformer is used)
        return x, self.transformer.layers[0].attn.attention_probs, umap_features
