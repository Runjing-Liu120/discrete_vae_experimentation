import numpy as np

import os
import torch
import torch.nn as nn

import torch.optim as optim

from torch.distributions import Normal
import torch.nn.functional as F

import common_utils
import timeit

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
        self.fc1 = nn.Linear(self.n_pixels + self.n_classes, 128) # 128 hidden nodes; two more layers
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, self.n_pixels)
        self.fc5 = nn.Linear(self.n_pixels, latent_dim * 2)

    def forward(self, image, z):

        # feed through neural network
        h = image.view(-1, self.n_pixels)
        h = torch.cat((h, z), dim = 1)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        h = self.fc5(h)

        # get means, std, and class weights
        indx1 = self.latent_dim
        indx2 = 2 * self.latent_dim
        # indx3 = 2 * self.latent_dim + self.n_classes

        latent_means = h[:, 0:indx1]
        latent_std = torch.exp(h[:, indx1:indx2])
        # free_class_weights = h[:, indx2:]
        # class_weights = common_utils.get_symplex_from_reals(free_class_weights)

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

        self.fc1 = nn.Linear(self.n_pixels, 128)
        self.fc2 = nn.Linear(128, n_classes - 1)

    def forward(self, image):
        h = image.view(-1, self.n_pixels)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))

        return common_utils.get_symplex_from_reals(h)

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

        self.fc1 = nn.Linear(latent_dim + n_classes, self.n_pixels)
        self.fc2 = nn.Linear(self.n_pixels, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, self.n_pixels * 2)


    def forward(self, latent_params, z):
        assert latent_params.shape[1] == self.latent_dim
        assert z.shape[1] == self.n_classes # z should be one hot encoded
        assert latent_params.shape[0] == z.shape[0]

        h = torch.cat((latent_params, z), dim = 1)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        h = self.fc5(h)

        h = h.view(-1, 2, self.slen, self.slen)

        image_mean = h[:, 0, :, :]
        image_std = torch.exp(h[:, 1, :, :])

        return image_mean, image_std

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

    def encoder_forward(self, image, one_hot_z):
        assert one_hot_z.shape[0] == image.shape[0]
        assert one_hot_z.shape[1] == self.n_classes

        latent_means, latent_std = self.encoder(image, one_hot_z)

        latent_samples = torch.randn(latent_means.shape).to(device) * latent_std + latent_means

        return latent_means, latent_std, latent_samples #, class_weights

    def decoder_forward(self, latent_samples, one_hot_z):
        assert one_hot_z.shape[0] == latent_samples.shape[0]
        assert one_hot_z.shape[1] == self.n_classes

        image_mean, image_std = self.decoder(latent_samples, one_hot_z)

        return image_mean, image_std

    def forward(self, image):
        # Note this forward module is not differentiable
        # bc we sample a discrete class
        assert 1 == 2, 'not implemented yet '

    def loss(self, image, true_class_labels = None):

        # latent_means, latent_std, latent_samples, computed_class_weights = \
        #     self.encoder_forward(image)

        computed_class_weights = self.classifier(image)

        if true_class_labels is not None:
            # print('setting true class label')
            true_class_weights_np = np.zeros(computed_class_weights.shape)
            true_class_weights_np[np.arange(computed_class_weights.shape[0]),
                            true_class_labels] = 1

            class_weights = torch.Tensor(true_class_weights_np).to(device)
        else:
            class_weights = computed_class_weights

        # likelihood term
        loss = 0.0
        for z in range(self.n_classes):
            batch_z = torch.ones(image.shape[0]).to(device) * z

            one_hot_batch_z = common_utils.get_one_hot_encoding_from_int(batch_z, self.n_classes)

            latent_means, latent_std, latent_samples = \
                self.encoder_forward(image, one_hot_batch_z)

            image_mu, image_std = self.decoder_forward(latent_samples, one_hot_batch_z)

            # likelihood term
            normal_loglik_z = common_utils.get_normal_loglik(image, image_mu,
                                                    image_std, scale = False)

            if not(np.all(np.isfinite(normal_loglik_z.detach().cpu().numpy()))):
                print(z)
                print(image_std)
                assert np.all(np.isfinite(normal_loglik_z.detach().cpu().numpy()))

            loss -= (class_weights[:, z] * normal_loglik_z).sum()
            
            # entropy term
            kl_q_latent = common_utils.get_kl_q_standard_normal(latent_means, \
                                                                latent_std)
            assert np.all(np.isfinite(kl_q_latent.detach().cpu().numpy()))

            loss += (class_weights[:, z] * kl_q_latent).sum()

            # print('log like', loss / image.size()[0])
        # kl term for latent parameters
        # (assuming standard normal prior)

        # print('kl q latent', kl_q_latent / image.size()[0])

        # entropy term for class weights
        # (assuming uniform prior)
        if true_class_labels is not None:
            kl_q_z = 0.0
        else:
            kl_q_z = -common_utils.get_multinomial_entropy(class_weights)
            # print('kl q z', kl_q_z / image.size()[0])
            if not np.isfinite(kl_q_z.detach().cpu().numpy()):
                print(class_weights)
                assert np.isfinite(kl_q_z.detach().cpu().numpy())

        loss += kl_q_z

        return loss / image.size()[0], computed_class_weights

    def get_class_label_cross_entropy(self, class_weights, labels):
        return torch.sum(
            -torch.log(class_weights + 1e-8) * \
            common_utils.get_one_hot_encoding_from_int(labels, self.n_classes))

    def get_semisupervised_loss(self, labeled_images, labels, unlabeled_images, alpha):
        unlabeled_loss = self.loss(unlabeled_images)[0]
        labeled_loss, computed_class_weights = \
            self.loss(labeled_images, true_class_labels = labels)

        cross_entropy_term = self.get_class_label_cross_entropy(computed_class_weights, labels)

        return unlabeled_loss * unlabeled_images.size()[0] + \
                labeled_loss * labeled_images.size()[0] + \
                alpha * cross_entropy_term

    def eval_vae(self, train_loader, optimizer = None, train = False,
                    set_true_class_label = False):
        if train:
            self.train()
            assert optimizer is not None
        else:
            self.eval()

        avg_loss = 0.0

        num_images = train_loader.dataset.__len__()
        i = 0

        for batch_idx, data in enumerate(train_loader):
            # first entry of data is the actual image
            # the second entry is the true class label
            if torch.cuda.is_available():
                image = data['image'].to(device)
            else:
                image = data['image']

            # i+=1; print('batch {}'.format(i))

            if optimizer is not None:
                optimizer.zero_grad()

            batch_size = image.size()[0]

            if set_true_class_label:
                true_class_labels = data['label'].numpy()
            else:
                true_class_labels = None

            loss = self.loss(image, true_class_labels)[0]  * (batch_size / num_images)

            if train:
                loss.backward()
                optimizer.step()

            avg_loss += loss.data

        return avg_loss

    def train_module(self, train_loader, test_loader,
                    set_true_class_label = False,
                    outfile = './mnist_vae',
                    n_epoch = 200, print_every = 10, save_every = 20,
                    weight_decay = 1e-6, lr = 0.001,
                    save_final_enc = True):

        optimizer = optim.Adam(self.parameters(), lr=lr,
                                weight_decay=weight_decay)

        iter_array = []
        train_loss_array = []
        test_loss_array = []

        train_loss = self.eval_vae(train_loader, set_true_class_label = set_true_class_label)
        test_loss = self.eval_vae(test_loader, set_true_class_label = set_true_class_label)
        print('  * init train recon loss: {:.10g};'.format(train_loss))
        print('  * init test recon loss: {:.10g};'.format(test_loss))

        iter_array.append(0)
        train_loss_array.append(train_loss.detach().cpu().numpy())
        test_loss_array.append(test_loss.detach().cpu().numpy())

        for epoch in range(1, n_epoch + 1):
            start_time = timeit.default_timer()

            batch_loss = self.eval_vae(train_loader,
                                        optimizer = optimizer,
                                        train = True,
                                        set_true_class_label = set_true_class_label)

            elapsed = timeit.default_timer() - start_time
            print('[{}] loss: {:.10g}  \t[{:.1f} seconds]'.format(epoch, batch_loss, elapsed))

            if epoch % print_every == 0:
                train_loss = self.eval_vae(train_loader, set_true_class_label = set_true_class_label)
                test_loss = self.eval_vae(test_loader, set_true_class_label = set_true_class_label)

                print('  * train recon loss: {:.10g};'.format(train_loss))
                print('  * test recon loss: {:.10g};'.format(test_loss))

                iter_array.append(epoch)
                train_loss_array.append(train_loss.detach().cpu().numpy())
                test_loss_array.append(test_loss.detach().cpu().numpy())

            if epoch % save_every == 0:
                outfile_every = outfile + '_enc_epoch' + str(epoch)
                print("writing the encoder parameters to " + outfile_every + '\n')
                torch.save(self.encoder.state_dict(), outfile_every)

                outfile_every = outfile + '_dec_epoch' + str(epoch)
                print("writing the decoder parameters to " + outfile_every + '\n')
                torch.save(self.decoder.state_dict(), outfile_every)


        if save_final_enc:
            outfile_final = outfile + '_enc_final'
            print("writing the encoder parameters to " + outfile_final + '\n')
            torch.save(self.encoder.state_dict(), outfile_final)

            outfile_final = outfile + '_dec_final'
            print("writing the decoder parameters to " + outfile_final + '\n')
            torch.save(self.decoder.state_dict(), outfile_final)

            loss_array = np.zeros((3, len(iter_array)))
            loss_array[0, :] = iter_array
            loss_array[1, :] = train_loss_array
            loss_array[2, :] = test_loss_array
            np.savetxt(outfile + 'loss_array.txt', loss_array)


######################################
# FUNCTIONS TO TRAIN SEMI-SUPERVISED MODEL
######################################
# TODO: integrate this into the class ...
def train_semi_supervised_loss(vae, train_loader_unlabeled, labeled_images, labels, optimizer, alpha):
    assert labeled_images.shape[0] == len(labels)

    vae.train()

    avg_loss = 0.0

    num_images = len(train_loader_unlabeled)

    i = 0
    for batch_idx, data in enumerate(train_loader_unlabeled):

        if torch.cuda.is_available():
            unlabeled_images = data['image'].to(device)
            labeled_images = labeled_images.to(device)
            labels = labels.to(device)
        else:
            unlabeled_images = data['image']

        optimizer.zero_grad()

        batch_size = unlabeled_images.size()[0]

        semi_super_loss = vae.get_semisupervised_loss(labeled_images, labels, unlabeled_images, alpha) / num_images

        semi_super_loss.backward()
        optimizer.step()

        avg_loss += semi_super_loss.data

    return avg_loss

def train_semisupervised_model(vae, train_loader_unlabeled, labeled_images, labels,
                    test_loader, alpha = 0.01,
                    outfile = './mnist_vae_semisupervised',
                    n_epoch = 200, print_every = 10, save_every = 20,
                    weight_decay = 1e-6, lr = 0.001,
                    save_final_enc = True):

    # define optimizer
    optimizer = optim.Adam(vae.parameters(), lr=lr,
                            weight_decay=weight_decay)

    iter_array = []
    train_loss_array = []
    test_loss_array = []

    train_loss = vae.eval_vae(train_loader_unlabeled)
    test_loss = vae.eval_vae(test_loader)
    print('  * init train recon loss: {:.10g};'.format(train_loss))
    print('  * init test recon loss: {:.10g};'.format(test_loss))

    iter_array.append(0)
    train_loss_array.append(train_loss.detach().cpu().numpy())
    test_loss_array.append(test_loss.detach().cpu().numpy())

    for epoch in range(1, n_epoch + 1):
        start_time = timeit.default_timer()

        batch_loss = train_semi_supervised_loss(vae,
                                train_loader_unlabeled,
                                labeled_images, labels, optimizer, alpha)

        elapsed = timeit.default_timer() - start_time
        print('[{}] loss: {:.10g}  \t[{:.1f} seconds]'.format(epoch, batch_loss, elapsed))

        if epoch % print_every == 0:
            train_loss = vae.eval_vae(train_loader_unlabeled)
            test_loss = vae.eval_vae(test_loader)

            print('  * train recon loss: {:.10g};'.format(train_loss))
            print('  * test recon loss: {:.10g};'.format(test_loss))

            iter_array.append(epoch)
            train_loss_array.append(train_loss.detach().cpu().numpy())
            test_loss_array.append(test_loss.detach().cpu().numpy())

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


    if save_final_enc:
        outfile_final = outfile + '_enc_final'
        print("writing the encoder parameters to " + outfile_final + '\n')
        torch.save(vae.encoder.state_dict(), outfile_final)

        outfile_final = outfile + '_dec_final'
        print("writing the decoder parameters to " + outfile_final + '\n')
        torch.save(vae.decoder.state_dict(), outfile_final)

        outfile_final = outfile + '_classifier_final' + str(epoch)
        print("writing the classifier parameters to " + outfile_final + '\n')
        torch.save(vae.classifier.state_dict(), outfile_final)


        loss_array = np.zeros((3, len(iter_array)))
        loss_array[0, :] = iter_array
        loss_array[1, :] = train_loss_array
        loss_array[2, :] = test_loss_array
        np.savetxt(outfile + 'loss_array.txt', loss_array)
