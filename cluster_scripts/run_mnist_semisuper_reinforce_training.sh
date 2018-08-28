#!/bin/bash

python ../libraries/run_mnist_semisupervised_training.py \
			--epochs 200 \
			--save_every 50 \
			--batch_size 256 \
			--seed 901 \
			--outfilename '../mnist_vae_results_aws/mnist_vae_semisuper_reinforce_trained_baseline' \
			--alpha 1.0 \
			--propn_labeled 0.1 \
			--reinforce True \
			--use_baseline True \
			--latent_dim 32 

