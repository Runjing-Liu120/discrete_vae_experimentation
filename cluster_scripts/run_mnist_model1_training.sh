#!/bin/bash

python ../libraries/run_mnist_model1_training.py \
			--epochs 5 \
			--save_every 2 \
			--propn_sample 0.0001 \
			--batch_size 12 \
			--seed 901 \
			--outfilename '../mnist_vae_results_aws/testing' \
			--latent_dim 36 \
			--learning_rate 1e-4
