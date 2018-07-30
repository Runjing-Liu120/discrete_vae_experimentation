#!/bin/bash

python ../libraries/run_mnist_semisupervised_training.py \
			--epochs 100 \
			--save_every 10 \
			--batch_size 256 \
			--seed 9033 \
			--outfilename '../mnist_vae_results_aws/mnist_vae_semisupervised_alpha1' \
			--alpha 1.0 
