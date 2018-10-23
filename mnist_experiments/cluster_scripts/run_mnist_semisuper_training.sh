#!/bin/bash

python ../libraries/run_mnist_semisupervised_training.py \
			--epochs 50 \
			--save_every 10 \
			--batch_size 256 \
			--seed 901 \
			--outdir '../mnist_vae_results/'\
			--outfilename 'mnist_vae' \
			--alpha 1.0 \
			--propn_labeled 0.05 \
			--propn_sample 1.0 \
			--latent_dim 50 \
			--learning_rate 1e-3
