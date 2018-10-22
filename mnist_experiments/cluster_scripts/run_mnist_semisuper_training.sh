#!/bin/bash

python ../libraries/run_mnist_semisupervised_training.py \
			--epochs 300 \
			--save_every 50 \
			--batch_size 64 \
			--seed 901 \
			--outdir '../mnist_vae_results/'\
			--outfilename 'mnist_vae' \
			--alpha 1.0 \
			--propn_labeled 0.01 \
			--propn_sample 1.0 \
			--latent_dim 32 \
			--learning_rate 1e-3
