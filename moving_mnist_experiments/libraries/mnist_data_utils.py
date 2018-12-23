import os

import torch

import numpy as np

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.sampler import Sampler

def load_mnist_data(data_dir = '../mnist_data/', train = True):
    assert os.path.exists(data_dir)

    # trans = transforms.Compose([transforms.ToTensor(),
    #                             transforms.Normalize((0.5,), (1.0,))])

    trans = lambda x: transforms.ToTensor()(x).bernoulli()

    # sigmoid = torch.nn.Sigmoid()
    # trans = lambda x: (sigmoid(transforms.ToTensor()(x)) - 0.5) * 2

    data = dset.MNIST(root=data_dir, train=train,
                            transform=trans, download=True)

    return data

class MNISTDataSet(Dataset):

    def __init__(self, data_dir = '../mnist_data/',
                    propn_sample = 1.0,
                    indices = None,
                    train_set = True):

        super(MNISTDataSet, self).__init__()

        # Load MNIST dataset
        assert os.path.exists(data_dir)

        # This is the full dataset
        self.mnist_data_set = load_mnist_data(data_dir = data_dir, train = train_set)

        if train_set:
            n_image_full = len(self.mnist_data_set.train_labels)
        else:
            n_image_full = len(self.mnist_data_set.test_labels)

        # we may wish to subset
        if indices is None:
            self.num_images = round(n_image_full * propn_sample)
            self.sample_indx = np.random.choice(n_image_full, self.num_images, replace = False)
        else:
            self.num_images = len(indices)
            self.sample_indx = indices

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        return {'image' : self.mnist_data_set[self.sample_indx[idx]][0].squeeze(),
                'label' : self.mnist_data_set[self.sample_indx[idx]][1].squeeze()}

def get_mnist_dataset(data_dir = '../mnist_data/',
                    propn_sample = 1.0):
    train_set = MNISTDataSet(data_dir = data_dir,
                            propn_sample = propn_sample,
                            train_set = True)

    test_set = MNISTDataSet(data_dir = data_dir,
                            propn_sample = propn_sample,
                            train_set = False)

    return train_set, test_set
