import os

import torch

import matplotlib.pyplot as plt

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.sampler import Sampler

import numpy as np

CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD = (0.2675, 0.2565, 0.2761)

CIFAR_MEAN_TENSOR = torch.Tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
CIFAR_STD_TENSOR = torch.Tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)


def load_cifar100_data(data_dir = '../cifar100_data/', train = True):
    # adapted from
    # https://github.com/meliketoy/wide-resnet.pytorch/blob/master/main.py

    assert os.path.exists(data_dir)

    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]) # meanstd transformation
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    print("| Preparing CIFAR-100 dataset...")
    # sys.stdout.write("| ")
    cifar_data = dset.CIFAR100(root=data_dir, train=train,
                                download=True, transform=transform)

    return cifar_data

class CIFAR100DataSet(Dataset):

    def __init__(self, data_dir = '../cifar100_data/',
                    propn_sample = 1.0,
                    indices = None,
                    train_set = True):

        super(CIFAR100DataSet, self).__init__()

        # Load MNIST dataset
        assert os.path.exists(data_dir)

        # This is the full dataset
        self.cifar100_dataset = load_cifar100_data(data_dir = data_dir, train = train_set)

        if train_set:
            n_image_full = len(self.cifar100_dataset.train_labels)
        else:
            n_image_full = len(self.cifar100_dataset.test_labels)

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
        return {'image' : self.cifar100_dataset[self.sample_indx[idx]][0].squeeze(),
                'label' : self.cifar100_dataset[self.sample_indx[idx]][1]}


def load_semisupervised_cifar_dataset(data_dir = '../cifar100_data/',
                    propn_sample = 1.0, propn_labeled = 0.1):

    total_num_train_images = 50000 # is there way to read this in?

    # subsample training set if desired
    num_train_images = round(propn_sample * total_num_train_images)
    subs_train_set = np.random.choice(total_num_train_images, \
                        num_train_images,
                        replace = False)

    # split training set into labeled and unlabled images
    num_labeled_images = round(num_train_images * propn_labeled)
    train_set_labeled = CIFAR100DataSet(data_dir = data_dir,
                            indices = subs_train_set[:num_labeled_images],
                            train_set = True)
    train_set_unlabeled = CIFAR100DataSet(data_dir = data_dir,
                            indices = subs_train_set[num_labeled_images:],
                            train_set = True)

    # get test set as usual
    test_set = CIFAR100DataSet(data_dir = data_dir,
                            propn_sample = propn_sample,
                            train_set = False)

    return train_set_labeled, train_set_unlabeled, test_set


# Functions to examine class accuracies and reconstruction_loss

# LOAD TRUE LABELS
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

labels_legend_all = unpickle('../cifar100_data/cifar-100-python/meta')
fine_labels_legend_ = labels_legend_all[b'fine_label_names']
fine_labels_legend = [str(fine_labels_legend_[i])[2:-1] for i in range(len(fine_labels_legend_))]

def get_topk_labels(class_weights, topk):
    assert len(class_weights.shape) == 1, 'this is implemented only for a vector of class weights'
    probs, indx = torch.topk(class_weights, k = topk)

    labels = []
    for i in range(topk):
        labels.append(fine_labels_legend[indx[i]])

    return probs, labels
