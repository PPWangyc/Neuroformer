#!/bin/bash

conda activate neuroformer
cd ..
python neuroformer_train.py \
       --dataset ibl \
       --config configs/Visnav/lateral/mconf_pretrain.yaml
cd script
conda deactivate