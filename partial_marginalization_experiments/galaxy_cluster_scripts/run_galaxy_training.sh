#!/bin/bash

python ../libraries/train_galaxy_vae.py \
			--epochs 501 \
			--save_every 5 \
			--batchsize 64 \
			--seed 901 \
			--topk 0 \
			--n_samples 50 \
			--vae_outdir '../galaxy_results/testing_reinforce/' \
			--vae_warm_start True \
			--vae_init_file '../galaxy_results/topk0/galaxy_vae_topk0_epoch80.dat'\
			--vae_outfilename 'galaxy_vae_reinforce_nsamples50' 
