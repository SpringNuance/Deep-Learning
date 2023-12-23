import os
import numpy as np
from scipy import linalg

import torch
import torch.nn as nn
from torch.nn import functional as F


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.fc1 = nn.Linear(32*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    @classmethod
    def pretrained(cls):
        model = cls()
        filename = os.path.join(os.path.dirname(__file__), 'fd_classifier.pth')
        model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
        return model

    def forward(self, x):
        """
        Args:
          x of shape (batch_size, 1, 28, 28): Input images.
        
        Returns:
          y of shape (batch_size, 10): Outputs of the network.
        """
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        x = self.pool(F.relu(x))
        x = x.view(-1, self.fc1.in_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_activations(self, x):
        """
        Args:
          x of shape (batch_size, 1, 28, 28): Input images.
        
        Returns:
          y of shape (batch_size, 10): Outputs of the network.
        """
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        x = self.pool(F.relu(x))
        x = x.view(-1, self.fc1.in_features)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        return x


class Classifier32(nn.Module):
    def __init__(self):
        super(Classifier32, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3, padding=1)
        self.conv1n = nn.BatchNorm2d(num_features=20)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, padding=1)
        self.conv2n = nn.BatchNorm2d(num_features=20)
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, padding=1)
        self.conv3n = nn.BatchNorm2d(num_features=20)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, padding=1)
        self.conv4n = nn.BatchNorm2d(num_features=40)
        self.conv5 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3, padding=1)
        self.conv5n = nn.BatchNorm2d(num_features=40)
        self.conv6 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3, padding=1)
        self.conv6n = nn.BatchNorm2d(num_features=40)

        self.conv7 = nn.Conv2d(in_channels=40, out_channels=60, kernel_size=3)
        self.conv7n = nn.BatchNorm2d(num_features=60)
        self.conv8 = nn.Conv2d(in_channels=60, out_channels=40, kernel_size=1)
        self.conv8n = nn.BatchNorm2d(num_features=40)
        self.conv9 = nn.Conv2d(in_channels=40, out_channels=20, kernel_size=1)
        self.conv9n = nn.BatchNorm2d(num_features=20)

        size = 28 // 4
        kernel_size = size - 2
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size)
        #in_features = n_channels * (size-kernel_size) * (size-kernel_size)
        self.fc = nn.Linear(in_features=20, out_features=10)

    @classmethod
    def pretrained(cls):
        model = cls()
        filename = os.path.join(os.path.dirname(__file__), 'fd_vgg_mnist.pth')
        model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
        return model

    def forward(self, x):
        """
        Args:
          x of shape (batch_size, 1, 28, 28): Input images.
        
        Returns:
          y of shape (batch_size, 10): Outputs of the network.
        """
        x = F.relu(self.conv1n(self.conv1(x)))
        x = F.relu(self.conv2n(self.conv2(x)))
        x = F.relu(self.conv3n(self.conv3(x)))
        x = self.max_pool(x)
        x = F.relu(self.conv4n(self.conv4(x)))
        x = F.relu(self.conv5n(self.conv5(x)))
        x = F.relu(self.conv6n(self.conv6(x)))
        x = self.max_pool(x)
        x = F.relu(self.conv7n(self.conv7(x)))
        x = F.relu(self.conv8n(self.conv8(x)))
        x = F.relu(self.conv9n(self.conv9(x)))
        x = self.avg_pool(x)
        x = x.view(-1, self.fc.in_features)
        x = self.fc(x)
        return x
        
    def get_activations(self, x):
        """
        Args:
          x of shape (batch_size, 1, 28, 28): Input images.
        
        Returns:
          y of shape (batch_size, 10): Outputs of the network.
        """
        x = F.relu(self.conv1n(self.conv1(x)))
        x = F.relu(self.conv2n(self.conv2(x)))
        x = F.relu(self.conv3n(self.conv3(x)))
        x = self.max_pool(x)
        x = F.relu(self.conv4n(self.conv4(x)))
        x = F.relu(self.conv5n(self.conv5(x)))
        x = F.relu(self.conv6n(self.conv6(x)))
        x = self.max_pool(x)
        x = F.relu(self.conv7n(self.conv7(x)))
        x = F.relu(self.conv8n(self.conv8(x)))
        x = F.relu(self.conv9n(self.conv9(x)))
        x = self.avg_pool(x)
        x = x.view(-1, self.fc.in_features)
        return x


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


class FDScore(nn.Module):
    def __init__(self, mu=None, sigma=None, imsize=28):
        super(FDScore, self).__init__()
        self.imsize = imsize
        if imsize == 28:
            self.net = net = Classifier.pretrained()
        else:
            self.net = net = Classifier32.pretrained()
        self.net.eval()
        self.mu = mu
        self.sigma = sigma

    @classmethod
    def pretrained(cls, imsize=28):
        filename = 'fd_mnist_musigma.pt' if imsize == 28 else 'fd_mnist_musigma_32.pt'
        filename = os.path.join(os.path.dirname(__file__), filename)
        musigma = torch.load(filename)
        mu, sigma = musigma['mu'], musigma['sigma']
        return cls(mu, sigma, imsize)
    
    def save(self):
        filename = 'fd_mnist_musigma.pt' if imsize == 28 else 'fd_mnist_musigma_32.pt'
        filename = os.path.join(os.path.dirname(__file__), filename)
        musigma = {'mu': self.mu, 'sigma': self.sigma}
        torch.save(musigma, filename)

    def train(self, trainset, batch_size):
        'Compute statistic from training set'
        dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
        device = next(self.net.parameters()).device
        
        n = 0
        self.mu = 0
        self.sigma = 0
        for batch in dataloader:
            samples = batch[0].to(device)
            activations = self.get_activations(samples)

            batch_size = samples.size(0)
            w_old = n / (n + batch_size)
            w_new = batch_size / (n + batch_size)

            self.mu = w_old * self.mu + w_new * np.mean(activations, axis=0)
            self.sigma = w_old * self.sigma + w_new * np.cov(activations, rowvar=False)
            n = n + batch_size

    def get_activations(self, samples):
        with torch.no_grad():
            return self.net.get_activations(samples).cpu().numpy()

    def calculate(self, samples):
        minmax = np.array([samples.cpu().numpy().min(), samples.cpu().numpy().max()])
        if not np.allclose(minmax, np.array([-1., 1.]), rtol=0.1, atol=0.1):
            print('Warning: the pixel values should be in the range [-1, 1]. The range in the samples is %s.' % minmax)
        act_fake = self.get_activations(samples)
        mu_fake = np.mean(act_fake, axis=0)
        sigma_fake = np.cov(act_fake, rowvar=False)

        return frechet_distance(mu_fake, sigma_fake, self.mu, self.sigma)