import numpy as np

import os
import torch
import torch.nn as nn

import torch.optim as optim

from torch.distributions import Normal, Categorical
import torch.nn.functional as F

from cifar_vae_lib import CIFARConditionalVAE
import cifar_data_utils
import mnist_utils

import timeit

import sys
sys.path.insert(0, '../../partial_marginalization_experiments/libraries/')
import partial_marginalization_lib as pm_lib
import common_utils as pm_common_utils

import semisupervised_vae_lib as ss_vae_lib

from torch.optim import lr_scheduler

from copy import deepcopy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def get_cifar_semisuperivsed_vae(image_config = {'slen': 32,
#                                                  'channel_num': 3,
#                                                  'n_classes': 100},
#                                 cond_vae_config = {'kernel_num': 128,
#                                                    'z_size': 32},
#                                 classifier_config = {'depth': 28,
#                                                      'widen_factor': 10,
#                                                      'dropout_rate': 0.3}):
#
#     cond_vae = CIFARConditionalVAE(slen = image_config['slen'],
#                                     channel_num = image_config['channel_num'],
#                                     kernel_num = cond_vae_config['kernel_num'],
#                                     z_size = cond_vae_config['z_size'],
#                                     n_classes = image_config['n_classes'])
#
#     classifier = Wide_ResNet(slen = image_config['slen'],
#                              depth = classifier_config['depth'],
#                              widen_factor = classifier_config['widen_factor'],
#                              dropout_rate = classifier_config['dropout_rate'],
#                              n_classes = image_config['n_classes'])
#
#     cifar_loglik = lambda image, image_mean, image_var: \
#                             nn.BCELoss(size_average=False)(image_mean, image)
#
#     return ss_vae_lib.SemiSupervisedVAE(cond_vae, classifier, cifar_loglik)



sys.path.insert(0, '../../../pytorch-cifar-models/models/')
sys.path.insert(0, '../../../pytorch-cifar-models/')
from models import densenet_cifar

class MyClassifier(nn.Module):
    def __init__(self, depth = 10, k = 12, n_classes = 100, slen = 28):
        super(MyClassifier, self).__init__()

        self.classfier = densenet_cifar.densenet_BC_cifar(depth = depth,
                                                k = k,
                                                num_classes = n_classes)

        self.n_classes = n_classes
        self.slen = slen

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, image):
        out = self.classfier(image)

        return self.log_softmax(out)

def cifar_loglik(image, image_mean, image_var, use_cifar100):
    if use_cifar100:
        # we are doing cifar100
        CIFAR_STD_TENSOR = cifar_data_utils.CIFAR100_STD_TENSOR
        CIFAR_MEAN_TENSOR = cifar_data_utils.CIFAR100_MEAN_TENSOR
    else:
        CIFAR_STD_TENSOR = cifar_data_utils.CIFAR10_STD_TENSOR
        CIFAR_MEAN_TENSOR = cifar_data_utils.CIFAR10_MEAN_TENSOR

    image_unscaled = image * CIFAR_STD_TENSOR.to(device) + \
                                CIFAR_MEAN_TENSOR.to(device)

    return mnist_utils.get_bernoulli_loglik(image_mean, image_unscaled)


def get_cifar_semisuperivsed_vae(image_config = {'use_cifar100': True,
                                                 'slen': 32,
                                                 'channel_num': 3,
                                                 'n_classes': 100},
                                cond_vae_config = {'kernel_num': 128,
                                                   'z_size': 32},
                                classifier_config = {'depth': 10,
                                                    'k': 12}):

    cond_vae = CIFARConditionalVAE(slen = image_config['slen'],
                                    channel_num = image_config['channel_num'],
                                    kernel_num = cond_vae_config['kernel_num'],
                                    z_size = cond_vae_config['z_size'],
                                    n_classes = image_config['n_classes'],
                                    use_cifar100 = image_config['use_cifar100'])

    classifier = MyClassifier(depth = classifier_config['depth'],
                                k = classifier_config['k'],
                                n_classes = image_config['n_classes'],
                                slen = image_config['slen'])

    cifar_spec_loglik = lambda image, image_mean, image_var : \
                                cifar_loglik(image, image_mean, image_var,
                                                image_config['use_cifar100'])

    return ss_vae_lib.SemiSupervisedVAE(cond_vae, classifier, cifar_spec_loglik)
