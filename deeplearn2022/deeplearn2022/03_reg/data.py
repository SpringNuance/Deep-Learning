import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset
from torchvision.datasets.utils import download_and_extract_archive


class RatingsData(TensorDataset):
    download_url_prefix = 'https://users.aalto.fi/~alexilin/dle'
    zip_filename = 'ratings.zip'
    
    def __init__(self, root, train=True):
        self.root = root
        self._folder = folder = os.path.join(root, 'Ratings')
        self._fetch_data(root)
        
        self.n_users = 943
        self.n_items = 1682

        if train:
            filename = os.path.join(folder, 'train.tsv')
        else:
            filename = os.path.join(folder, 'test.tsv')

        cols = ['user_ids', 'item_ids', 'ratings', 'timestamps']
        df = pd.read_csv(filename, sep='\t', names=cols).astype(int)
        user_ids = torch.LongTensor(df.user_ids.values)
        item_ids = torch.LongTensor(df.item_ids.values)
        ratings = torch.LongTensor(df.ratings.values)

        super(RatingsData, self).__init__(user_ids, item_ids, ratings)
    
    def _check_integrity(self):
        files = ['train.tsv', 'test.tsv']
        for file in files:
            if not os.path.isfile(os.path.join(self._folder, file)):
                return False
        return True

    def _fetch_data(self, data_dir):
        if self._check_integrity():
            return
        
        url = self.download_url_prefix + '/' + self.zip_filename
        download_and_extract_archive(url, data_dir, filename=self.zip_filename, remove_finished=True)



    