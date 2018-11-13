import os

import torch

import matplotlib.pyplot as plt

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.sampler import Sampler

import numpy as np

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

CIFAR100_MEAN_TENSOR = torch.Tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
CIFAR100_STD_TENSOR = torch.Tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR10_MEAN_TENSOR = torch.Tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
CIFAR10_STD_TENSOR = torch.Tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)



def load_cifar_data(cifar100 = True, data_dir = '../cifar100_data/',
                    train = True):
    # adapted from
    # https://github.com/meliketoy/wide-resnet.pytorch/blob/master/main.py

    assert os.path.exists(data_dir)

    if cifar100:
        cifar_mean, cifar_std = (CIFAR100_MEAN, CIFAR100_STD)
    else:
        cifar_mean, cifar_std = (CIFAR10_MEAN, CIFAR10_STD)

    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ]) # meanstd transformation
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ])

    print("| Preparing CIFAR dataset...")
    # sys.stdout.write("| ")
    if cifar100:
        cifar_data = dset.CIFAR100(root=data_dir, train=train,
                                    download=True, transform=transform)
    else:
        cifar_data = dset.CIFAR10(root=data_dir, train=train,
                                    download=True, transform=transform)

    return cifar_data

class CIFARDataSet(Dataset):

    def __init__(self, cifar100, data_dir,
                    propn_sample = 1.0,
                    indices = None,
                    train_set = True):

        super(CIFARDataSet, self).__init__()

        # Load MNIST dataset
        assert os.path.exists(data_dir)

        # This is the full dataset
        self.cifar_dataset = load_cifar_data(cifar100,
                                            data_dir = data_dir,
                                            train = train_set)

        if train_set:
            n_image_full = len(self.cifar_dataset.train_labels)
        else:
            n_image_full = len(self.cifar_dataset.test_labels)

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
        return {'image' : self.cifar_dataset[self.sample_indx[idx]][0].squeeze(),
                'label' : self.cifar_dataset[self.sample_indx[idx]][1]}


def load_semisupervised_cifar_dataset(cifar100, data_dir,
                                        propn_sample = 1.0,
                                        propn_labeled = 0.1):

    total_num_train_images = 50000 # is there way to read this in?

    # subsample training set if desired
    num_train_images = round(propn_sample * total_num_train_images)
    subs_train_set = np.random.choice(total_num_train_images, \
                        num_train_images,
                        replace = False)

    # split training set into labeled and unlabled images
    num_labeled_images = round(num_train_images * propn_labeled)
    train_set_labeled = CIFARDataSet(cifar100, data_dir = data_dir,
                            indices = subs_train_set[:num_labeled_images],
                            train_set = True)
    if propn_labeled == 1:
        train_set_unlabeled = None
    else:
        train_set_unlabeled = CIFARDataSet(cifar100, data_dir = data_dir,
                                indices = subs_train_set[num_labeled_images:],
                                train_set = True)

    # get test set as usual
    test_set = CIFARDataSet(cifar100, data_dir = data_dir,
                            propn_sample = propn_sample,
                            train_set = False)

    return train_set_labeled, train_set_unlabeled, test_set


# Functions to examine class accuracies and reconstruction_loss
# TODO: this only works for cifar-100 at the moment.
# LOAD TRUE LABELS
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cifar100_labels_legend_all = unpickle('../cifar100_data/cifar-100-python/meta')
cifar100_fine_labels_legend_ = cifar100_labels_legend_all[b'fine_label_names']
cifar100_fine_labels_legend = [str(cifar100_fine_labels_legend_[i])[2:-1] for i in range(len(cifar100_fine_labels_legend_))]

# cifar10_labels_legend_all = unpickle('../cifar10_data/cifar-10-python/meta')
# cifar10_fine_labels_legend_ = cifar10_labels_legend_all[b'fine_label_names']
# cifar10_fine_labels_legend = [str(cifar10_fine_labels_legend_[i])[2:-1] for i in range(len(cifar10_fine_labels_legend_))]

def get_topk_labels(class_weights, topk):
    assert len(class_weights.shape) == 1, 'this is implemented only for a vector of class weights'
    probs, indx = torch.topk(class_weights, k = topk)

    labels = []
    for i in range(topk):
        labels.append(fine_labels_legend[indx[i]])

    return probs, labels
