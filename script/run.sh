#!/bin/bash

conda activate neuroformer
cd ..
python neuroformer_train.py \
       --dataset ibl \
       --config configs/ibl/pretrain.yaml
cd script
conda deactivate