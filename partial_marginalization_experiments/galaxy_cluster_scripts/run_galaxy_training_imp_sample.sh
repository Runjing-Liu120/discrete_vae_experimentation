#!/bin/bash

python ../libraries/train_galaxy_vae_imp_sampling.py \
			--epochs 201 \
			--save_every 20 \
			--batchsize 64 \
			--seed 904 \
			--max_detections 3\
			--use_baseline True \
			--use_importance_sample False \
			--vae_outdir '../galaxy_results/multiple_detections/' 
  			--vae_outfilename 'galaxy_vae_3detections_bs_on_imp_sample_off'
#                         --vae_outfilename 'galaxy_vae_3detections'

