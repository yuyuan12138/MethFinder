import torch
import torch.nn as nn
from Conv1d_weight import Conv1d_location_specific as Spec_Conv1d
from Transformer_weight import Transformer_Different_Attention_Encoder as Spec_transfomerEncoder


class Net(nn.Module):
    def __init__(self, in_channels, out_channels, use_spectic_conv1d=False, use_spectic_transformer=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if use_spectic_conv1d:
            self.conv1D = Spec_Conv1d(in_channels=4, out_channels=64, kernel_size=3, padding=1, stride=1,
                                      weight_learning=True)
        else:
            self.conv1D = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1, stride=1)

        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.conv2D = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 3), padding=(0, 1))
        self.bn_2 = nn.BatchNorm1d(64)

        if use_spectic_conv1d:
            self.conv1D_2 = Spec_Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0,
                                        weight_learning=True)
        else:
            self.conv1D_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        # self.conv1D_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1)

        self.bn_3 = nn.BatchNorm1d(64)

        if use_spectic_transformer:
            self.transformer = Spec_transfomerEncoder(d_model=64,
                                                      norm_in_channels=64,
                                                      N=1,
                                                      dim_feedforward=8,
                                                      nhead=2,
                                                      weight=True)
        else:
            self.transformer = nn.Sequential(
                nn.TransformerEncoderLayer(d_model=64, nhead=2, dim_feedforward=8, batch_first=True),
                # nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=8, batch_first=True),
            )
        # self.Bide = nn.LSTM(input_size=64, dropout=0.5, num_layers=8, hidden_size=64, batch_first=True)

        self.dropout = nn.Dropout(0.5)
        self.flat = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(1312 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, self.out_channels),
        )

        self.use_spectic_conv1d = use_spectic_conv1d
        self.use_spectic_transformerEncoder = use_spectic_transformer


    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2)
        y = torch.flip(x, dims=[0])

        # Spec Conv
        if self.use_spectic_conv1d:
            x = self.conv1D(x, non_important_site=10, important_site=31)
            y = self.conv1D(y, non_important_site=10, important_site=31)
        else:
            x = self.conv1D(x)
            y = self.conv1D(y)

        x = self.bn(x)
        y = self.bn(y)
        x = self.relu(x)
        y = self.relu(y)
        x = torch.stack([x, y], dim=1)
        x = x.transpose(1, 2)

        # Adver Conv
        x = self.conv2D(x)
        x = x.squeeze(2)
        x = self.bn_2(x)
        x = self.relu(x)
        
        # Spec Conv 2
        if self.use_spectic_conv1d:
            x = self.conv1D_2(x, non_important_site=10, important_site=31)
        else:
            x = self.conv1D_2(x)
        x = self.bn_3(x)
        x = self.relu(x)

        x = x.transpose(1, 2)

        # Transformer
        x = self.transformer(x)
        
        x = self.dropout(x)
        x = self.flat(x)
        y = x
        x = self.fc(x)
        
        return x, y
