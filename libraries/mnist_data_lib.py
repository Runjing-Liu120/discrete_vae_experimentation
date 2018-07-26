import os

import torch

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.utils.data import DataLoader

from torch.utils.data.sampler import Sampler

def load_mnist_data(data_dir = '../mnist_data/'):
    assert os.path.exists(data_dir)

    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (1.0,))])

    train_set = dset.MNIST(root=data_dir, train=True,
                            transform=trans, download=False)
    test_set = dset.MNIST(root=data_dir, train=False,
                            transform=trans, download=False)

    return train_set, test_set


class DataSubsetter(Sampler):
    def __init__(self, mask):
        self.mask = mask

    def __iter__(self):
        return (i for i in self.mask)

    def __len__(self):
        return len(self.mask)

def subsample_mnist_data(data_set, propn_sample, batch_size):
    # when we want to work with a smaller sample of the
    # MNIST dataset, this returns a loader class
    # that only uses part of th dataset

    n_samples = round(propn_sample * len(data_set))

    mask = torch.Tensor([i for i in range(n_samples)])
    sampler = DataSubsetter(mask)

    return DataLoader(data_set, batch_size=batch_size,
                        sampler = sampler,
                        shuffle=False,
                        num_workers=2)
