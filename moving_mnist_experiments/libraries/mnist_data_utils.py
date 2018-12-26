import os

import torch

import numpy as np

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as f
from torch.distributions import Categorical

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
        return {'image' : self.mnist_data_set[self.sample_indx[idx]][0],
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

#######################
# Classes and functions for moving mnist

def pixel_1d_to_2d(sout, padding, pixel_1d):
    # converts 1d pixel coordinates to 2d

    pixel_2d_row = pixel_1d.div(sout - 2 * padding)
    pixel_2d_col = pixel_1d.remainder(sout - 2 * padding)
    pixel_2d = torch.stack([pixel_2d_row, pixel_2d_col], dim=1)

    return pixel_2d + padding

def pad_image(image, pixel_2d, sout):
    # image is the input image
    # pixel 2d are the coordinates, from 0 to sout
    # first coordinate is x, second coordinate is y
    # sout is the side length of the output image

    # image should be N x slen x slen
    assert len(image.shape) == 3

    # assert there is a coordinate for each image
    assert image.shape[0] == pixel_2d.shape[0]

    batchsize = image.shape[0]
    sin = image.shape[-1]

    r0 = (sout - 1) / 2
    # TODO: cache this
    grid_out = torch.FloatTensor(np.mgrid[0:sout, 0:sout].transpose() - r0)

    grid1 = grid_out.unsqueeze(0).expand([pixel_2d.size(0), -1, -1, -1])

    grid2 = grid1 - pixel_2d.float().unsqueeze(1).unsqueeze(1)
    grid3 = (grid2.float() + (sout // 2)) / (sin // 2)

    # grid sample only works with 4D inputs
    image_ = image.view(batchsize, 1, sin, sin)
    padded = f.grid_sample(image_, grid3)

    return padded.squeeze()

def crop_image(image, pixel_2d, sin):
    # image should be N x slen x slen
    assert len(image.shape) == 3

    # assert there is a coordinate for each image
    assert image.shape[0] == pixel_2d.shape[0]

    batchsize, h, _ = image.shape

    r = sin // 2

    # TODO: this can be cached, too
    grid0 = torch.from_numpy(np.mgrid[(-r):(r+1), (-r):(r+1)].transpose([2, 1, 0]))

    grid1 = grid0.unsqueeze(0).expand([image.size(0), -1, -1, -1])
    grid2 = grid1 + pixel_2d.view(image.size(0), 1, 1, 2) - (h - 1) / 2
    grid3 = grid2.float() / ((h - 1) / 2)

    # grid sample only works with 4D inputs
    image_ = image.view(batchsize, 1, h, h)

    return f.grid_sample(image_, grid3).squeeze()


class MovingMNISTDataSet(Dataset):

    def __init__(self, slen = 68,
                    padding = 14,
                    data_dir = '../mnist_data/',
                    propn_sample = 1.0,
                    indices = None,
                    train_set = True):

        # slen is the side length of the image on which an mnist digit (28 x 28)
        # is placed. Padding is the width of the border of the full image

        super(MovingMNISTDataSet, self).__init__()

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

        # set up parameters for moving MNIST
        self.slen = slen
        self.padding = padding
        # number of possible pixel locations
        self.n_pixel_1d = (slen -  2 * padding) ** 2
        # define uniform categorical variable over pixels
        unif_probs = torch.ones(self.n_pixel_1d) / self.n_pixel_1d
        unif_probs = unif_probs.view(-1, self.n_pixel_1d)
        self.categorical = Categorical(unif_probs)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):

        # draw pixel locations
        pixel_1d = self.categorical.sample()
        pixel_2d = pixel_1d_to_2d(self.slen, self.padding, pixel_1d)

        # get translated image
        image = self.mnist_data_set[self.sample_indx[idx]][0]
        image_translated = pad_image(image, pixel_2d, self.slen)

        label = self.mnist_data_set[self.sample_indx[idx]][1].squeeze()

        return {'image' : image_translated,
                'label' : label,
                'pixel_2d': pixel_2d.squeeze()}

def get_moving_mnist_dataset(data_dir = '../mnist_data/',
                    propn_sample = 1.0):
    train_set = MovingMNISTDataSet(data_dir = data_dir,
                            propn_sample = propn_sample,
                            train_set = True)

    test_set = MovingMNISTDataSet(data_dir = data_dir,
                            propn_sample = propn_sample,
                            train_set = False)

    return train_set, test_set
