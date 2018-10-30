import numpy as np

import os
import torch
import torch.nn as nn

import torch.optim as optim

from torch.distributions import Normal, Categorical
import torch.nn.functional as F

from cifar_vae_lib import CIFARConditionalVAE
from cifar_classifier_lib import Wide_ResNet

import timeit

import sys
sys.path.insert(0, '../../partial_marginalization_experiments/libraries/')
import partial_marginalization_lib as pm_lib
import common_utils as pm_common_utils

import semisupervised_vae_lib as ss_vae_lib

from torch.optim import lr_scheduler

from copy import deepcopy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_cifar_semisuperivsed_vae(image_config = {'slen': 32,
                                                 'channel_num': 3,
                                                 'n_classes': 100},
                                cond_vae_config = {'kernel_num': 128,
                                                   'z_size': 32},
                                classifier_config = {'depth': 28,
                                                     'widen_factor': 10,
                                                     'dropout_rate': 0.3}):

    cond_vae = CIFARConditionalVAE(slen = image_config['slen'],
                                    channel_num = image_config['channel_num'],
                                    kernel_num = cond_vae_config['kernel_num'],
                                    z_size = cond_vae_config['z_size'],
                                    n_classes = image_config['n_classes'])

    classifier = Wide_ResNet(slen = image_config['slen'],
                             depth = classifier_config['depth'],
                             widen_factor = classifier_config['widen_factor'],
                             dropout_rate = classifier_config['dropout_rate'],
                             n_classes = image_config['n_classes'])

    cifar_loglik = lambda image, image_mean, image_var: \
                            nn.BCELoss(size_average=False)(image_mean, image)

    return ss_vae_lib.SemiSupervisedVAE(cond_vae, classifier, cifar_loglik)