#!/bin/bash

python ../libraries/run_mnist_semisupervised_training.py \
			--epochs 200 \
			--save_every 10 \
			--batch_size 256 \
			--seed 9033 \
			--outfilename '../mnist_vae_results_aws/mnist_vae_semisuper_reinforce' \
			--propn_labeled 0.1 \
			--reinforce True 

