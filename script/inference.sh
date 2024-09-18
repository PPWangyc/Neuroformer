#!/bin/bash

conda activate neuroformer
cd ..

python neuroformer_inference.py --dataset ibl \
                                --ckpt_path "models/NF.15/Visnav_VR_Expt/ibl/Neuroformer/None/(state_history=6,_state=6,_stimulus=6,_behavior=6,_self_att=6,_modalities=(n_behavior=25))/25" \
                                --predict_modes speed phi th

cd script