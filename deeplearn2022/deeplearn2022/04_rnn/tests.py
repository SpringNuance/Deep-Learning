import os
import numpy as np
import numpy.testing as npt

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_weights_encoder(net):
    for child in net.children():
        if isinstance(child, (nn.GRU, nn.GRUCell)):
            if isinstance(child, nn.GRU):
                weight_ih = child.weight_ih_l0
                weight_hh = child.weight_hh_l0
                bias_ih = child.bias_ih_l0
                bias_hh = child.bias_hh_l0
            else:
                weight_ih = child.weight_ih
                weight_hh = child.weight_hh
                bias_ih = child.bias_ih
                bias_hh = child.bias_hh

            weight_ih.data.fill_(0)
            weight_ih.data[3, :].fill_(0)  # output gate: >0 => old state, <0 => new value
            weight_ih.data[5, :].fill_(0.1)  # new value (tanh is applied after that)

            weight_hh.data.fill_(0.1)
            weight_hh.data[:, 1:].fill_(-0.1)

            bias_ih.data.fill_(0)
            bias_hh.data.fill_(0)

        elif isinstance(child, nn.Embedding):
            child.weight.data = child.weight.data = torch.tensor([
                [  1.,  -1.],
                [ -0.5,  0.2],
                [  0.1, -0.4],
                [ -0.3,  0.4],
                [ -0.1,  0.8],
            ])


def set_weights_decoder(net):
    for child in net.children():
        #print(child)
        if isinstance(child, (nn.GRU, nn.GRUCell)):
            if isinstance(child, nn.GRU):
                weight_ih = child.weight_ih_l0
                weight_hh = child.weight_hh_l0
                bias_ih = child.bias_ih_l0
                bias_hh = child.bias_hh_l0
            else:
                weight_ih = child.weight_ih
                weight_hh = child.weight_hh
                bias_ih = child.bias_ih
                bias_hh = child.bias_hh

            weight_ih.data.fill_(0)
            weight_ih.data[3, :].fill_(0)  # output gate: >0 => old state, <0 => new value
            weight_ih.data[5, :].fill_(0.1)  # new value (tanh is applied after that)

            weight_hh.data.fill_(0.1)
            weight_hh.data[:, 1:].fill_(-0.1)

            bias_ih.data.fill_(0)
            bias_hh.data.fill_(0)

        elif isinstance(child, nn.Embedding):
            child.weight.data = child.weight.data = torch.tensor([
                [ 0.,  1.], #[   1.,  1.],
                [ 1.,  0.4],
                [ 0.5, 0.2],
                [ 0.3, 0.],
                [ 0.1, 0.8],
            ])
        elif isinstance(child, nn.Linear):
            child.weight.data = torch.tensor([
                [ 0.6,  0.],
                [ 0.8, 0.2],
                [ 1.,  0.8],
                [ 0.,  .9],
                [ 0.5, 1.],
            ])
            child.bias.data.fill_(0)

