#!/bin/bash

python ../libraries/train_galaxy_vae.py \
			--epochs 501 \
			--save_every 10 \
			--batchsize 64 \
			--seed 901 \
			--topk 0 \
			--n_samples 2 \
			--vae_outdir '../galaxy_results/reinforce_nsamples2/' \
			--galaxy_enc_warm_start False \
			--galaxy_dec_warm_start False \
			--vae_outfilename 'galaxy_vae_reinforce_nsamples2' 
