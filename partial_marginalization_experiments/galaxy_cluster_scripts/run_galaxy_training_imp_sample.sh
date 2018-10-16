#!/bin/bash

python ../libraries/train_galaxy_vae_imp_sampling.py \
			--epochs 31 \
			--save_every 10 \
			--batchsize 64 \
			--seed 904 \
			--use_importance_sample True \
			--vae_outdir '../galaxy_results/imp_sampled/' 
