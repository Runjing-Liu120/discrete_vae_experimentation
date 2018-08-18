#!/bin/bash

python ../libraries/run_mnist_semisupervised_training.py \
			--epochs 200 \
			--save_every 10 \
			--batch_size 256 \
			--seed 901 \
			--outfilename '../mnist_vae_results_aws/bernoulli_losses/mnist_vae2_semisuper_reinforce' \
			--alpha 1.0 \
			--propn_labeled 0.1 \
			--reinforce True \
			--latent_dim 32 

