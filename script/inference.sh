#!/bin/bash

eid=${1}
mode=${2} # 'pretrain' or 'finetune'

conda activate neuroformer
cd ..

ckpt_path="models/NF.15/Visnav_VR_Expt/ibl/Neuroformer/None/(state_history=6,_state=6,_stimulus=6,_behavior=6,_self_att=6,_modalities=(n_behavior=25))/25/${eid}/${mode}"
python neuroformer_inference.py --dataset ibl \
                                --ckpt_path ${ckpt_path} \
                                --predict_modes wheel_speed whisker_energy \
                                --eid ${eid} 

cd script