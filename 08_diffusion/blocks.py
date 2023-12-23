import time
import numpy as np
import torch
import math

import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Two convolutional blocks with a resudual connection and with time and class conditioning.

    Args:
      in_channels (int):  Number of input channels.
      out_channels (int): Number of output channels.
      time_emb_dim (int): Time embedding dimension, None for no conditioning on time.
      num_classes (int): Number of classes, None for no conditioning on classes.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim=None, num_classes=None, dropout=0.1):
        super(ResidualBlock, self).__init__()

        num_groups = 32
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.time_bias = nn.Linear(time_emb_dim, out_channels) if time_emb_dim is not None else None
        self.class_bias = nn.Embedding(num_classes, out_channels) if num_classes is not None else None

        if in_channels != out_channels:
            self.residual_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_connection = nn.Identity()
    
    def forward(self, x, time_emb=None, y=None):
        """
        Args:
          x of shape (batch_size, in_channels, H, W): Inputs.
          time_emb of shape (batch_size, time_emb_dim): Time embedding tensor or None if there is no conditioning on time.
          y of shape (batch_size,): Classes tensor or None if there is no conditioning on time.
          
        Returns:
          out of shape (batch_size, out_channels, H, W): Outputs.
        """
        out = F.silu(self.norm1(x))
        out = self.conv1(out)

        if self.time_bias is not None:
            assert time_emb is not None, "time_emb should be passed"
            out += self.time_bias(F.silu(time_emb))[:, :, None, None]

        if self.class_bias is not None:
            assert y is not None, "y should be passed"
            out += self.class_bias(y)[:, :, None, None]

        out = F.silu(self.norm2(out))
        out = self.dropout(out)
        out = self.conv2(out) + self.residual_connection(x)

        return out


class Downsample(nn.Module):
    """Downsamples input 2d maps by a factor of 2 using a strided convolution.

    Args:
      in_channels (int): Number of input channels.
    """
    def __init__(self, in_channels):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)
    
    def forward(self, x, *args):
        """
        Args:
          x of shape (batch_size, in_channels, H, W): Inputs.
          args: Ignored arguments.
          
        Returns:
          out of shape (batch_size, in_channels, H/2, W/2): Outputs.
        """
        return self.conv(x)


class Upsample(nn.Module):
    """Upsamples 2d maps by a factor of 2.
    
    Args:
      in_channels (int): Number of input channels.
    """
    def __init__(self, in_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
    
    def forward(self, x, *args):
        """
        Args:
          x of shape (batch_size, in_channels, H, W): Inputs.
          args: Ignored arguments.
          
        Returns:
          out of shape (N, in_channels, H * 2, W * 2): Outputs.
        """
        out = self.upsample(x)
        out = self.conv(out)
        return out


class PositionalEmbedding(nn.Module):
    """Positional embedding of timesteps.

    Args:
      dim (int): Embedding dimension.
    """
    def __init__(self, dim):
        super(PositionalEmbedding, self).__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = 1.0

    def forward(self, x):
        """
        Args:
          x of shape (batch_size,): Timesteps.
        
        Returns:
          emb of shape (batch_size, dim): Positional embedding of timesteps.
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

