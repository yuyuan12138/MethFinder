import copy
import math
import torch
from torch import nn
import torch.nn.functional as F

class different_Self_Attention(nn.Module):
    """
    Custom self-attention layer with optional position-specific weighting.
    """
    def __init__(self, d_model, nhead, weight=False):
        super(different_Self_Attention, self).__init__()

        # Initialize parameters and layers
        self.attention_probs = None
        self.nhead = nhead  # Number of attention heads
        self.weight = weight  # Whether to apply position-specific weights

        # Linear layers to compute Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Softmax for attention scores normalization
        self.softmax = nn.Softmax(dim=1)

        # Normalization factor for scaled dot-product attention
        self._norm_fact = 1 / math.sqrt(d_model // nhead)

    def forward(self, encoder_outputs, func_lr=None):
        """
        Forward pass for the custom self-attention layer.
        """
        # Get batch size, sequence length (num), and model dimensionality (d_model)
        batch, num, d_model = encoder_outputs.shape
        num_heads = self.nhead
        d_head_model = d_model // num_heads

        # Calculate Q, K, V matrices
        Q = self.w_q(encoder_outputs)
        K = self.w_k(encoder_outputs)
        V = self.w_v(encoder_outputs)

        # Apply position-specific weights if specified
        if self.weight:
            weight_artificial = nn.Parameter(torch.tensor([0.18, 0.64, 0.18]))
            # Split Q, K, V into non-important and important sections
            Q_b_1 = Q[:, 0:10, :]
            Q_a = Q[:, 10:31, :]
            Q_b_2 = Q[:, 31:41, :]
            K_b_1 = K[:, 0:10, :]
            K_a = K[:, 10:31, :]
            K_b_2 = K[:, 31:41, :]
            V_b_1 = V[:, 0:10, :]
            V_a = V[:, 10:31, :]
            V_b_2 = V[:, 31:41, :]

            # Concatenate weighted Q, K, V
            Q = torch.concat([weight_artificial[0] * Q_b_1,
                              weight_artificial[1] * Q_a,
                              weight_artificial[2] * Q_b_2], dim=1)
            K = torch.concat([weight_artificial[0] * K_b_1,
                              weight_artificial[1] * K_a,
                              weight_artificial[2] * K_b_2], dim=1)
            V = torch.concat([weight_artificial[0] * V_b_1,
                              weight_artificial[1] * V_a,
                              weight_artificial[2] * V_b_2], dim=1)

        # Reshape Q, K, V for multi-head attention and transpose for batch processing
        Q = Q.reshape(batch, num, num_heads, d_head_model).transpose(1, 2)
        K = K.reshape(batch, num, num_heads, d_head_model).transpose(1, 2)
        V = V.reshape(batch, num, num_heads, d_head_model).transpose(1, 2)

        # Compute scaled dot-product attention scores
        attention_sorces = torch.matmul(Q, K.transpose(-1, -2)) * self._norm_fact

        # Apply softmax to get attention probabilities
        self.attention_probs = nn.Softmax(dim=-1)(attention_sorces)

        # Compute attention output
        out = torch.matmul(self.attention_probs, V)
        out = out.transpose(1, 2).reshape(batch, num, d_model)  # Reshape back to original size

        return out

class FeedForward(nn.Module):
    """
    Position-wise FeedForward layer with dropout and ReLU activation.
    """
    def __init__(self, hidden_size, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, hidden_size)

    def forward(self, x):
        # Apply feedforward network with dropout and ReLU activation
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class Norm(nn.Module):
    """
    Layer normalization module.
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()

        # Initialize parameters for normalization
        self.size = hidden_size
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        # Apply layer normalization
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class Transformer_Different_Attention_EncoderLayer(nn.Module):
    """
    Custom Transformer Encoder layer with different self-attention.
    """
    def __init__(self, d_model, norm_in_channels, dim_feedforward, nhead, dropout=0.1, threshold_value=False, weight=False):
        super().__init__()
        self.norm_1 = Norm(norm_in_channels)  # First layer normalization
        self.norm_2 = Norm(norm_in_channels)  # Second layer normalization
        self.attn = different_Self_Attention(d_model, nhead, weight=weight)  # Custom self-attention layer
        self.ff = FeedForward(norm_in_channels, dim_feedforward)  # Feedforward network
        self.dropout_1 = nn.Dropout(dropout)  # Dropout for attention output
        self.dropout_2 = nn.Dropout(dropout)  # Dropout for feedforward output
        self.threshold_value = threshold_value  # Apply threshold to output if True

    def forward(self, x, func_lr=None):
        # Normalize input and apply self-attention
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, func_lr=func_lr))

        # Normalize output and apply feedforward network
        x2 = self.norm_2(x)

        # Apply threshold if specified
        if self.threshold_value:
            mask = x2 < 0.15
            x2[mask] = 0

        # Add feedforward output to input
        x = x + self.dropout_2(self.ff(x2))
        return x

def get_clones(module, N):
    """
    Create N identical layers (deep copies).
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Transformer_Different_Attention_Encoder(nn.Module):
    """
    Transformer Encoder with different self-attention mechanism.
    """
    def __init__(self, d_model, norm_in_channels, N, dim_feedforward, nhead, threshold_value=False, weight=False):
        super().__init__()
        self.N = N  # Number of encoder layers
        # Create N encoder layers with custom self-attention
        self.layers = get_clones(
            Transformer_Different_Attention_EncoderLayer(d_model, norm_in_channels, dim_feedforward, nhead,
                                                         threshold_value=threshold_value, weight=weight), N)
        self.norm = Norm(norm_in_channels)  # Final normalization layer

    def forward(self, x, func_lr=None):
        # Pass input through each encoder layer
        for i in range(self.N):
            x = self.layers[i](x, func_lr=func_lr)
        # Apply final normalization
        x = self.norm(x)
        return x
