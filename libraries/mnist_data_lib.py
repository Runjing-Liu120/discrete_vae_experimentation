import os

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F

def load_mnist_data(data_dir = '../mnist_data/'):
    assert os.path.exists(data_dir)

    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (1.0,))])

    train_set = dset.MNIST(root=data_dir, train=True,
                            transform=trans, download=False)
    test_set = dset.MNIST(root=data_dir, train=False,
                            transform=trans, download=False)

    return train_set, test_set
