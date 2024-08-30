from torch import nn
import torch

class Conv1d_location_specific(nn.Module):
    """
    Custom 1D convolutional layer with position-specific weighting.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, weight_learning=False):
        super(Conv1d_location_specific, self).__init__()

        # Initialize weight parameter if weight learning is enabled
        self.weight = None
        if weight_learning:
            # Learnable weights initialized with predefined values
            self.weight = nn.Parameter(torch.tensor([0.18, 0.64, 0.18]))

        self.kernel_size = kernel_size
        self.stride = stride
        self.weight_learning = weight_learning

        # Three separate Conv1d layers for different regions of input
        self.conv1d_1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)  
        self.conv1d_2 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv1d_3 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, vector, non_important_site, important_site):
        """
        Forward pass for location-specific Conv1d with weighted outputs.
        """
        if self.weight_learning:
            # Ensure indices are valid
            num = vector.size(2)
            assert 0 <= non_important_site <= important_site <= num, "Invalid site indices"

            # Apply convolutions to different sections of the input
            output_1 = self.conv1d_1(vector[:, :, 0:non_important_site])
            output_2 = self.conv1d_2(vector[:, :, non_important_site:important_site])
            output_3 = self.conv1d_3(vector[:, :, important_site:num])

            # Concatenate weighted outputs along the feature dimension
            output = torch.cat([self.weight[0] * output_1,
                                self.weight[1] * output_2,
                                self.weight[2] * output_3], dim=2)
        else:
            # Default behavior or raise error if not handled
            raise ValueError("Weight learning is disabled, provide an alternative behavior.")

        return output
