import os
import torch
from torch.utils.data import TensorDataset
from torchvision.datasets.utils import download_and_extract_archive


class Sudoku(TensorDataset):
    download_url_prefix = 'https://users.aalto.fi/~alexilin/dle'
    zip_filename = 'sudoku.zip'

    def __init__(self, root, train=True):
        self.root = root
        self._folder = folder = os.path.join(root, 'sudoku')
        self._fetch_data(root)

        X = torch.load(os.path.join(folder, 'features.pt'))
        Y = torch.load(os.path.join(folder, 'labels.pt'))

        X = X.view(-1, 9*9, 9)
        Y = Y.view(-1, 9*9, 9)
        Y = Y.argmax(dim=2)

        n_train = 9000
        if train:
            x, y = X[:n_train], Y[:n_train]
        else:
            x, y = X[n_train:], Y[n_train:]
        super(Sudoku, self).__init__(x, y)

    def _check_integrity(self):
        files = ['features.pt', 'labels.pt']
        for file in files:
            if not os.path.isfile(os.path.join(self._folder, file)):
                return False
        return True

    def _fetch_data(self, data_dir):
        if self._check_integrity():
            return
        
        url = self.download_url_prefix + '/' + self.zip_filename
        download_and_extract_archive(url, data_dir, filename=self.zip_filename, remove_finished=True)
