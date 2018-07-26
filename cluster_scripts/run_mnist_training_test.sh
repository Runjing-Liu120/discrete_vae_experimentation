#!/bin/bash

python ../libraries/run_mnist_training.py \
			--epochs 5 \
			--save_every 2 \
			--batch_size 512 \
			--seed 9033 \
			--outfilename '../mnist_vae_results_aws/test' \
			--subsample_data True 		
