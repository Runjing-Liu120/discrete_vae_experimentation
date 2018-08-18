#!/bin/bash

python ../libraries/run_mnist_semisupervised_training.py \
			--epochs 100 \
			--save_every 1000 \
			--batch_size 64 \
			--propn_sample 0.005 \
			--seed 901 \
			--outfilename '../mnist_vae_results_aws/bernoulli_losses/test' \
			--alpha 1.0 \
			--propn_labeled 0.1 \
			--reinforce True \
			--latent_dim 32 

