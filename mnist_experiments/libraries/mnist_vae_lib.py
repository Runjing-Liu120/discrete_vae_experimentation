import numpy as np

import os
import torch
import torch.nn as nn

import torch.optim as optim

from torch.distributions import Normal, Categorical
import torch.nn.functional as F

import mnist_utils
import timeit

import sys
sys.path.insert(0, '../../partial_marginalization_experiments/libraries/')
import partial_marginalization_lib as pm_lib
import common_utils as pm_common_utils

from copy import deepcopy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MLPEncoder(nn.Module):
    def __init__(self, latent_dim = 5,
                    slen = 28,
                    n_classes = 10):
        # the encoder returns the mean and variance of the latent parameters
        # given the image and its class (one hot encoded)

        super(MLPEncoder, self).__init__()

        # image / model parameters
        self.n_pixels = slen ** 2
        self.latent_dim = latent_dim
        self.slen = slen
        self.n_classes = n_classes

        # define the linear layers
        self.fc1 = nn.Linear(self.n_pixels + self.n_classes, 256) # 128 hidden nodes; two more layers
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, latent_dim * 2)

    def forward(self, image, z):

        # feed through neural network
        h = image.view(-1, self.n_pixels)
        h = torch.cat((h, z), dim = 1)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = self.fc4(h)
        # h = self.fc4(h)
        # h = self.fc5(h)

        # get means, std, and class weights
        indx1 = self.latent_dim
        indx2 = 2 * self.latent_dim
        # indx3 = 2 * self.latent_dim + self.n_classes

        latent_means = h[:, 0:indx1]
        latent_std = torch.exp(h[:, indx1:indx2])
        # free_class_weights = h[:, indx2:]
        # class_weights = mnist_utils.get_symplex_from_reals(free_class_weights)

        return latent_means, latent_std #, class_weights


class Classifier(nn.Module):
    def __init__(self, slen = 28, n_classes = 10):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(Classifier, self).__init__()

        self.slen = slen
        self.n_pixels = slen ** 2
        self.n_classes = n_classes

        self.fc1 = nn.Linear(self.n_pixels, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, n_classes)

        self.log_softmax = nn.LogSoftmax(dim=1)


    def forward(self, image):
        h = image.view(-1, self.n_pixels)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = self.fc4(h)

        return self.log_softmax(h)

class BaselineLearner(nn.Module):
    def __init__(self, slen = 28):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(BaselineLearner, self).__init__()

        self.slen = slen
        self.n_pixels = slen ** 2

        self.fc1 = nn.Linear(self.n_pixels, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, image):
        h = image.view(-1, self.n_pixels)

        h = F.relu(self.fc1(h))
        h = self.fc2(h)

        return h

class MLPConditionalDecoder(nn.Module):
    def __init__(self, latent_dim = 5,
                        n_classes = 10,
                        slen = 28):

        # This takes the latent parameters and returns the
        # mean and variance for the image reconstruction

        super(MLPConditionalDecoder, self).__init__()

        # image/model parameters
        self.n_pixels = slen ** 2
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.slen = slen

        self.fc1 = nn.Linear(latent_dim + n_classes, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, self.n_pixels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, latent_params, z):
        assert latent_params.shape[1] == self.latent_dim
        assert z.shape[1] == self.n_classes # z should be one hot encoded
        assert latent_params.shape[0] == z.shape[0]

        h = torch.cat((latent_params, z), dim = 1)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = self.fc4(h)

        h = h.view(-1, self.slen, self.slen)

        # image_mean = h[:, 0, :, :]
        # image_std = torch.exp(h[:, 1, :, :])
        image_mean = self.sigmoid(h)

        return image_mean # , image_std

class HandwritingVAE(nn.Module):

    def __init__(self, latent_dim = 5,
                    n_classes = 10,
                    slen = 28):

        super(HandwritingVAE, self).__init__()

        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.slen = slen

        self.encoder = MLPEncoder(latent_dim = latent_dim,
                                    slen = slen)

        self.classifier = Classifier(n_classes = n_classes, slen = slen)

        self.decoder = MLPConditionalDecoder(latent_dim = latent_dim,
                                                n_classes = n_classes,
                                                slen = slen)

        # self.use_baseline = use_baseline
        # if self.use_baseline:
        #     self.baseline_learner = BaselineLearner(slen = self.slen)

    def encoder_forward(self, image, one_hot_z):
        assert one_hot_z.shape[0] == image.shape[0]
        assert one_hot_z.shape[1] == self.n_classes

        latent_means, latent_std = self.encoder(image, one_hot_z)

        latent_samples = torch.randn(latent_means.shape).to(device) * latent_std + latent_means

        return latent_means, latent_std, latent_samples

    def decoder_forward(self, latent_samples, one_hot_z):
        assert one_hot_z.shape[0] == latent_samples.shape[0]
        assert one_hot_z.shape[1] == self.n_classes

        image_mean = self.decoder(latent_samples, one_hot_z)

        return image_mean # , image_std

    def forward_conditional(self, image, z):
        # z is a vector of class labels
        # (integers, not one-hot-encoding)
        assert len(z) == image.shape[0]

        # one hot encode z
        one_hot_z  = mnist_utils.get_one_hot_encoding_from_int(z, self.n_classes)

        # pass through encoder
        latent_means, latent_std, latent_samples = \
            self.encoder_forward(image, one_hot_z)

        # pass through decoder
        image_mu = self.decoder_forward(latent_samples, one_hot_z)

        return image_mu, latent_means, latent_std, latent_samples

    def get_conditional_loss(self, image, z):
        # Returns the expectation of the objective conditional
        # on the class label z

        # z is a vector of class labels
        assert len(z) == image.shape[0]

        image_mu, latent_means, latent_std, latent_samples = \
            self.forward_conditional(image, z)

        # likelihood term
        # loglik_z = mnist_utils.get_normal_loglik(image, image_mu,
        #                                         image_std, scale = False)

        loglik_z = mnist_utils.get_bernoulli_loglik(image_mu, image)

        if not(np.all(np.isfinite(loglik_z.detach().cpu().numpy()))):
            print(z)
            print(image_mu)
            assert np.all(np.isfinite(loglik_z.detach().cpu().numpy()))

        # entropy term for the latent dimension
        kl_q_latent = mnist_utils.get_kl_q_standard_normal(latent_means, \
                                                            latent_std)

        assert np.all(np.isfinite(kl_q_latent.detach().cpu().numpy()))

        return -loglik_z + kl_q_latent

    def get_unlabeled_pm_loss(self, image, topk = 0, use_baseline = True,
                                    true_labels = None):

        # true labels for debugging only:
        if true_labels is None:
            log_q = self.classifier(image)
        else:
            # print('using true labels')
            batch_size = image.shape[0]
            q = torch.zeros((batch_size, self.n_classes)) + 1e-12
            seq_tensor = torch.LongTensor([i for i in range(batch_size)])
            q[seq_tensor, true_labels] = 1 - 1e-12 * (self.n_classes - 1)
            log_q = torch.log(q).to(device)

        f_z = lambda z : self.get_conditional_loss(image, z)

        pm_loss_z = pm_lib.get_partial_marginal_loss(f_z, log_q,
                                    alpha = 0.0,
                                    topk = topk,
                                    use_baseline = use_baseline,
                                    use_term_one_baseline = True)

        # print(pm_loss_z)

        # kl term for class weights
        # (assuming uniform prior)
        kl_q_z = (-torch.exp(log_q) * log_q).sum()

        # sampled loss:
        map_weights = torch.argmax(log_q, dim = 1)
        map_loss = f_z(map_weights)

        return pm_loss_z + kl_q_z, map_loss.sum()

    # def get_labeled_loss(self, image, true_class_labels):
    #     # class_weights = mnist_utils.get_one_hot_encoding_from_int(true_class_labels)
    #
    #     return self.get_conditional_loss(image, true_class_labels)

    def get_class_label_cross_entropy(self, log_class_weights, labels):
        assert np.all(log_class_weights.detach().cpu().numpy() <= 0)
        assert log_class_weights.shape[0] == len(labels)
        assert log_class_weights.shape[1] == self.n_classes

        return torch.sum(
            -log_class_weights * \
            mnist_utils.get_one_hot_encoding_from_int(labels, self.n_classes))

    def get_semisupervised_loss(self, unlabeled_images,
                                    labeled_images, labels,
                                    use_baseline = True,
                                    alpha = 1.0, topk = 0,
                                    true_labels = None):

        # unlabeled loss
        unlabeled_pm_loss, unlabeled_map_loss = \
            self.get_unlabeled_pm_loss(unlabeled_images,
                                        topk = topk, 
                                        use_baseline = use_baseline,
                                        true_labels = true_labels)

        # labeled loss
        labeled_loss = self.get_conditional_loss(labeled_images, labels)

        # cross entropy term
        log_q_labeled = self.classifier(labeled_images)
        cross_entropy_term = \
            self.get_class_label_cross_entropy(log_q_labeled, labels)

        return unlabeled_pm_loss.sum() + labeled_loss.sum() + \
                    alpha * cross_entropy_term.sum(), unlabeled_map_loss

    def eval_vae(self, train_loader, labeled_images, labels, \
                        optimizer = None,
                        train = False,
                        use_baseline = True,
                        topk = 0,
                        alpha = 1.0,
                        use_true_labels = False):
        if train:
            self.train()
            assert optimizer is not None
        else:
            self.eval()

        avg_loss = 0.0

        num_images = train_loader.dataset.__len__()
        i = 0

        for batch_idx, data in enumerate(train_loader):
            unlabeled_images = data['image'].to(device)

            if use_true_labels:
                true_labels = data['label'].to(device)
            else:
                true_labels = None

            if optimizer is not None:
                optimizer.zero_grad()

            batch_size = unlabeled_images.size()[0]

            loss, unlabeled_map_loss = \
                self.get_semisupervised_loss(unlabeled_images,
                                                labeled_images, labels,
                                                use_baseline = use_baseline,
                                                alpha = alpha, topk = topk,
                                                true_labels = true_labels)

            if train:
                loss.backward()
                optimizer.step()

            avg_loss += unlabeled_map_loss.data  / num_images

        return avg_loss

######################################
# FUNCTIONS TO TRAIN SEMI-SUPERVISED MODEL
######################################
# TODO: integrate this into the class ...
def eval_classification_accuracy(classifier, loader):
    accuracy = 0.0
    n_images = 0.0

    for batch_idx, data in enumerate(loader):

        # if torch.cuda.is_available():
        image = data['image'].to(device)
        label = data['label'].to(device)
        # else:
        #     image = data['image']
        #     label = data['label']

        log_class_weights = classifier(image)

        z_ind = torch.argmax(log_class_weights, dim = 1)

        accuracy += torch.sum(z_ind == label).float()

        n_images += len(z_ind)

    return accuracy / n_images

def train_semisupervised_model(vae, train_loader_unlabeled, labeled_images, labels,
                    test_loader, alpha = 1.0, topk = 0, use_baseline = True,
                    outfile = './mnist_vae_semisupervised',
                    n_epoch = 200, print_every = 10, save_every = 20,
                    weight_decay = 1e-6, lr = 0.001,
                    save_final_enc = True,
                    train_classifier_only = False,
                    use_true_labels = False):

    # define optimizer
    if train_classifier_only:
        # for debugging only
        optimizer = optim.Adam([
                {'params': vae.classifier.parameters(), 'lr': lr}],
                weight_decay=weight_decay)

    else:
        optimizer = optim.Adam([
                {'params': vae.classifier.parameters(), 'lr': lr},
                {'params': vae.encoder.parameters(), 'lr': lr},
                {'params': vae.decoder.parameters(), 'lr': lr}],
                weight_decay=weight_decay)

    iter_array = []
    train_loss_array = []
    test_loss_array = []
    train_class_accuracy_array = []
    test_class_accuracy_array = []

    # get losses
    train_loss = vae.eval_vae(train_loader_unlabeled, labeled_images, labels, use_baseline = False)
    test_loss = vae.eval_vae(test_loader, labeled_images, labels, use_baseline = False)
    print('  * init train recon loss: {:.10g};'.format(train_loss))
    print('  * init test recon loss: {:.10g};'.format(test_loss))

    # get test accuracies
    train_class_accuracy = eval_classification_accuracy(vae.classifier, train_loader_unlabeled)
    test_class_accuracy = eval_classification_accuracy(vae.classifier, test_loader)

    print('  * init train class accuracy: {:.4g};'.format(train_class_accuracy))
    print('  * init test class accuracy: {:4g};'.format(test_class_accuracy))

    iter_array.append(0)
    train_loss_array.append(train_loss.detach().cpu().numpy())
    test_loss_array.append(test_loss.detach().cpu().numpy())
    train_class_accuracy_array.append(train_class_accuracy.detach().cpu().numpy())
    test_class_accuracy_array.append(test_class_accuracy.detach().cpu().numpy())

    for epoch in range(1, n_epoch + 1):
        start_time = timeit.default_timer()

        unlabeled_loss = \
                vae.eval_vae(train_loader_unlabeled,
                                labeled_images = labeled_images,
                                labels = labels,
                                optimizer = optimizer,
                                train = True,
                                use_baseline = use_baseline,
                                alpha = alpha,
                                topk = topk,
                                use_true_labels = use_true_labels)

        elapsed = timeit.default_timer() - start_time
        print('[{}] unlabeled_loss: {:.10g}  \t[{:.1f} seconds]'.format(\
                    epoch, unlabeled_loss, elapsed))

        if epoch % print_every == 0:
            train_loss = vae.eval_vae(train_loader_unlabeled, labeled_images, labels, use_baseline = False)
            test_loss = vae.eval_vae(test_loader, labeled_images, labels, use_baseline = False)

            print('  * train recon loss: {:.10g};'.format(train_loss))
            print('  * test recon loss: {:.10g};'.format(test_loss))

            train_class_accuracy = eval_classification_accuracy(vae.classifier, train_loader_unlabeled)
            test_class_accuracy = eval_classification_accuracy(vae.classifier, test_loader)

            print('  * train class accuracy: {:.4g};'.format(train_class_accuracy))
            print('  * test class accuracy: {:4g};'.format(test_class_accuracy))

            iter_array.append(epoch)
            train_loss_array.append(train_loss.detach().cpu().numpy())
            test_loss_array.append(test_loss.detach().cpu().numpy())
            train_class_accuracy_array.append(train_class_accuracy.detach().cpu().numpy())
            test_class_accuracy_array.append(test_class_accuracy.detach().cpu().numpy())

        if epoch % save_every == 0:
            outfile_every = outfile + '_enc_epoch' + str(epoch)
            print("writing the encoder parameters to " + outfile_every + '\n')
            torch.save(vae.encoder.state_dict(), outfile_every)

            outfile_every = outfile + '_dec_epoch' + str(epoch)
            print("writing the decoder parameters to " + outfile_every + '\n')
            torch.save(vae.decoder.state_dict(), outfile_every)

            outfile_every = outfile + '_classifier_epoch' + str(epoch)
            print("writing the classifier parameters to " + outfile_every + '\n')
            torch.save(vae.classifier.state_dict(), outfile_every)

            loss_array = np.zeros((5, len(iter_array)))
            loss_array[0, :] = iter_array
            loss_array[1, :] = train_loss_array
            loss_array[2, :] = test_loss_array
            loss_array[3, :] = train_class_accuracy_array
            loss_array[4, :] = test_class_accuracy_array

            np.savetxt(outfile + 'loss_array.txt', loss_array)


    if save_final_enc:
        outfile_final = outfile + '_enc_final'
        print("writing the encoder parameters to " + outfile_final + '\n')
        torch.save(vae.encoder.state_dict(), outfile_final)

        outfile_final = outfile + '_dec_final'
        print("writing the decoder parameters to " + outfile_final + '\n')
        torch.save(vae.decoder.state_dict(), outfile_final)

        outfile_final = outfile + '_classifier_final'
        print("writing the classifier parameters to " + outfile_final + '\n')
        torch.save(vae.classifier.state_dict(), outfile_final)


#####################
# Some functions to examine VAE results
def get_classification_accuracy(loader, classifier,
                                    return_wrong_images = False,
                                    max_images = 1000):

    n_images = 0.0
    accuracy = 0.0

    wrong_images = torch.zeros((0, classifier.slen, classifier.slen))
    wrong_labels = torch.LongTensor(0)

    for batch_idx, data in enumerate(loader):
        class_weights = classifier(data['image'])

        z_ind = torch.argmax(class_weights, dim = 1)

        accuracy += torch.sum(z_ind == data['label']).float()

        if return_wrong_images:
            wrong_indx = 1 - (z_ind == data['label'])
            wrong_images = torch.cat((wrong_images,
                                    data['image'][wrong_indx, :, :]),
                                    dim = 0)
            wrong_labels = torch.cat((wrong_labels,
                                data['label'][wrong_indx]))
        else:
            wrong_images = None
            wrong_labels = None

        n_images += len(z_ind)
        if n_images > 1000:
            break

    return accuracy / n_images, wrong_images, wrong_labels

def get_reconstructions(vae, image):
    class_weights = vae.classifier(image)

    z_ind = torch.argmax(class_weights, dim = 1)
    z_ind_one_hot = \
        mnist_utils.get_one_hot_encoding_from_int(z_ind, vae.n_classes)

    latent_means, latent_std, latent_samples = \
        vae.encoder_forward(image, z_ind_one_hot)

    image_mu = vae.decoder_forward(latent_means, z_ind_one_hot)

    return image_mu, z_ind
