import os
import numpy as np

import copy
import torch
import matplotlib.pyplot as plt
from IPython import display

import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as utils


def test_generator_loss(generator_loss):
    class MyDiscriminator(nn.Module):
        def forward(self, x):
            out = torch.Tensor([0.1, 0.5, 0.8])
            return out
    
    with torch.no_grad():
        netD = MyDiscriminator()
        fake_images = torch.zeros(3, 1, 28, 28)
        loss = generator_loss(netD, fake_images)
        expected = torch.tensor(1.0729585886001587)
        print('loss:', loss)
        print('expected:', expected)
        assert torch.allclose(loss, expected), "out does not match expected value."
    print('Success')


def test_discriminator_loss(discriminator_loss):
    with torch.no_grad():
        class MyDiscriminator(nn.Module):
            def forward(self, x):
                if (x == 0).all():
                    return torch.Tensor([0.1, 0.2, 0.3])
                elif (x == 1).all():
                    return torch.Tensor([0.6, 0.7, 0.8])
        
        netD = MyDiscriminator()

        fake_images = torch.zeros(3, 1, 28, 28)
        real_images = torch.ones(3, 1, 28, 28)
        d_loss_real, D_real, d_loss_fake, D_fake = discriminator_loss(netD, real_images, fake_images)
        expected = torch.tensor(0.36354804039001465)
        print('d_loss_real:', d_loss_real)
        print('expected d_loss_real:', expected)
        assert torch.allclose(d_loss_real, expected), "d_loss_real does not match expected value."
        
        expected = 0.699999988079071
        print('D_real:', D_real)
        print('expected D_real:', expected)
        assert np.allclose(D_real, expected), "D_real does not match expected value."
        
        expected = torch.tensor(0.22839301824569702)
        print('d_loss_fake:', d_loss_fake.item())
        print('expected d_loss_fake:', expected)
        assert np.allclose(d_loss_fake, expected), "d_loss_fake does not match expected value."

        expected = 0.20000000298023224
        print('D_fake:', D_fake)
        print('expected D_fake:', expected)
        assert np.allclose(D_fake, expected), "D_fake does not match expected value."
        
    print('Success')


def test_gradient_penalty(gradient_penalty):
    class MyDiscriminator(nn.Module):
        def __init__(self):
            super(MyDiscriminator, self).__init__()
            self.linear = nn.Linear(28*28, 1, bias=False)
            self.linear.weight.data.fill_(1)
        def forward(self, x):
            out = self.linear(x.view(-1, 28*28))
            return out.view(-1)

    netD = MyDiscriminator()

    batch_size = 5
    real = torch.zeros(batch_size, 1, 28, 28)
    fake = torch.ones(batch_size, 1, 28, 28)
    loss, x = gradient_penalty(netD, real, fake)
    assert ((0 < x) & (x < 1)).all(), "All values in x should be between real and fake samples."

    expected = torch.tensor(729.)
    print('loss:', loss)
    print('expected:', expected)
    assert loss == expected, "Wrong calculations of gradient penalties."

    loss.backward()
    assert (netD.linear.weight.grad != 0).all(), "The gradients do not propagate through the gradent penalty."
    print('Success')
