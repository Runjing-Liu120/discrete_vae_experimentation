#!/bin/bash

python ../libraries/run_mnist_training.py \
			--epochs 100 \
			--save_every 10 \
			--batch_size 256 \
			--seed 9033 \
			--outfilename '../mnist_vae_results_aws/mnist_vae_set_true_class' \
			--set_true_class_label True \
			--subsample_data False	
