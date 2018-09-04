import os

import torch

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.sampler import Sampler

import numpy as np

def load_mnist_data(data_dir = '../mnist_data/', train = True):
    assert os.path.exists(data_dir)

    # trans = transforms.Compose([transforms.ToTensor(),
    #                             transforms.Normalize((0.5,), (1.0,))])

    # trans = lambda x: transforms.ToTensor()(x).bernoulli()

    sigmoid = torch.nn.Sigmoid()
    trans = lambda x: (sigmoid(transforms.ToTensor()(x)) - 0.5) * 2

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

        # if train_set:
        #     self.images = mnist_data_set.train_data[sample_indx, :, :].float()
        #     self.labels = mnist_data_set.train_labels[sample_indx].float()
        # else:
        #     self.images = mnist_data_set.test_data[sample_indx, :, :].float()
        #     self.labels = mnist_data_set.test_labels[sample_indx].float()

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

def get_mnist_dataset_semisupervised(data_dir = '../mnist_data/',
                    propn_sample = 1.0, propn_labeled = 0.1):
    total_num_train_images = 60000 # is there way to read this in?

    # subsample training set if desired
    num_train_images = round(propn_sample * total_num_train_images)
    subs_train_set = np.random.choice(total_num_train_images, \
                        num_train_images,
                        replace = False)

    # split training set into labeled and unlabled images
    num_labeled_images = round(num_train_images * propn_labeled)
    train_set_labeled = MNISTDataSet(data_dir = data_dir,
                            indices = subs_train_set[:num_labeled_images],
                            train_set = True)
    train_set_unlabeled = MNISTDataSet(data_dir = data_dir,
                            indices = subs_train_set[num_labeled_images:],
                            train_set = True)

    # get test set as usual
    test_set = MNISTDataSet(data_dir = data_dir,
                            propn_sample = propn_sample,
                            train_set = False)

    return train_set_labeled, train_set_unlabeled, test_set


# class DataSubsetter(Sampler):
#     def __init__(self, mask):
#         self.mask = mask
#
#     def __iter__(self):
#         return (i for i in self.mask)
#
#     def __len__(self):
#         return len(self.mask)
#
# def subsample_mnist_data(data_set, propn_sample, batch_size):
#     # when we want to work with a smaller sample of the
#     # MNIST dataset, this returns a loader class
#     # that only uses part of th dataset
#
#     n_samples = round(propn_sample * len(data_set))
#
#     mask = torch.Tensor([i for i in range(n_samples)])
#     sampler = DataSubsetter(mask)
#
#     return DataLoader(data_set, batch_size=batch_size,
#                         sampler = sampler,
#                         shuffle=False,
#                         num_workers=2)
