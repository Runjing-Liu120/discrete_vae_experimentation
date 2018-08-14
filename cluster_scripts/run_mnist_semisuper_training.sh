#!/bin/bash

python ../libraries/run_mnist_semisupervised_training.py \
			--epochs 100 \
			--save_every 10 \
			--batch_size 256 \
			--seed 9033 \
			--outfilename '../mnist_vae_results_aws/mnist_vae5_semisupervised' \
			--alpha 0.0 \
			--propn_labeled 0.0001  
