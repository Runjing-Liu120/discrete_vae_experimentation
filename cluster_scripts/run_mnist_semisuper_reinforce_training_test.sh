#!/bin/bash

python ../libraries/run_mnist_semisupervised_training.py \
			--epochs 5 \
			--save_every 2 \
			--batch_size 5 \
			--seed 9033 \
			--outfilename '../mnist_vae_results_aws/test' \
			--propn_sample 0.001 \
			--reinforce True
