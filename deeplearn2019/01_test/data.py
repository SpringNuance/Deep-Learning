import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset
from torchvision.datasets.utils import download_and_extract_archive


class WineQuality(TensorDataset):
    download_url_prefix = 'https://users.aalto.fi/~alexilin/dle'
    zip_filename = 'winequality.zip'
    
    def __init__(self, root, normalize=True, train=None):
        self.root = root
        self._folder = folder = os.path.join(root, 'winequality')
        self._fetch_data(root)

        df = pd.concat([
            pd.read_csv(os.path.join(folder, 'winequality-red.csv'), delimiter=';'),
            pd.read_csv(os.path.join(folder, 'winequality-white.csv'), delimiter=';')
        ])
        split_file = os.path.join(folder, 'winequality_split.pt')

        x = torch.Tensor(df.loc[:, df.columns != 'quality'].values)
        quality = torch.Tensor(df['quality'].values)
        
        if train is not None:
            #np.random.seed(1)
            #rp = np.random.permutation(x.size(0))
            #torch.save(torch.tensor(rp), split_file)
            rp = torch.load(split_file)
            n_train = int(x.size(0) * 0.8)
            
            if train:
                # Training set
                x = x[rp[:n_train]]
                quality = quality[rp[:n_train]]
            else:
                x = x[rp[n_train:]]
                quality = quality[rp[n_train:]]
                
        if normalize:
            self.x_mean = x.mean(dim=0)
            self.x_std = x.std(dim=0)
            x = (x - self.x_mean) / self.x_std
        
        #self.y = torch.LongTensor(df['quality'].values >= 7)  # Convert to a binary classification problem
        #dataset = torch.utils.data.TensorDataset(x, y)
        super(WineQuality, self).__init__(x, quality)
    
    def _check_integrity(self):
        files = ['winequality-red.csv', 'winequality-white.csv', 'winequality_split.pt']
        for file in files:
            if not os.path.isfile(os.path.join(self._folder, file)):
                return False
        return True

    def _fetch_data(self, data_dir):
        if self._check_integrity():
            return
        
        url = self.download_url_prefix + '/' + self.zip_filename
        download_and_extract_archive(url, data_dir, filename=self.zip_filename, remove_finished=True)
