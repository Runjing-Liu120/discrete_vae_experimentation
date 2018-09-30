#!/bin/bash

python ../libraries/train_galaxy_vae.py \
			--epochs 1001 \
			--save_every 50 \
			--batchsize 64 \
			--seed 901 \
			--vae_outdir '../galaxy_results/' \
			--vae_outfilename 'galaxy_vae' 
