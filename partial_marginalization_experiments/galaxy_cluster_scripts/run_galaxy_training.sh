#!/bin/bash

python ../libraries/train_galaxy_vae.py \
			--slen 20 \
			--epochs 201 \
			--save_every 10 \
			--batchsize 64 \
			--seed 904 \
			--topk 1 \
			--n_samples 1 \
			--vae_outdir '../galaxy_results_sandbox/topk1/' \
                        --vae_warm_start False \
			--vae_outfilename 'galaxy_vae_topk1' 
			
