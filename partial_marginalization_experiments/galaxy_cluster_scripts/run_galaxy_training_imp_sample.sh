#!/bin/bash

python ../libraries/train_galaxy_vae_imp_sampling.py \
			--epochs 51 \
			--save_every 10 \
			--batchsize 64 \
			--seed 904 \
			--use_baseline False \
			--use_importance_sample True \
			--vae_outdir '../galaxy_results/imp_sampled/' \
			--vae_outfilename 'galaxy_vae_imp_sampl_off'
# 			--vae_outfilename 'galaxy_vae'

