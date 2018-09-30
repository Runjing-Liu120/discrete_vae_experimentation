#!/bin/bash

python ../libraries/train_galaxy_vae.py \
			--epochs 1001 \
			--save_every 20 \
			--batchsize 64 \
			--seed 901 \
			--vae_outdir '../galaxy_results/' \
			--galaxy_enc_warm_start True \
			--galaxy_dec_warm_start True \
			--vae_outfilename 'galaxy_vae_warm_encdec_init' 
