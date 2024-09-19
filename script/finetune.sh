#!/bin/bash

#SBATCH --account=col169
#SBATCH --partition=gpu-shared
#SBATCH --job-name="mm"
#SBATCH --output="mm.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH --mem 150000
#SBATCH --gpus=1
#SBATCH -t 2-00
#SBATCH --export=ALL
module load gpu
module load slurm
. ~/.bashrc
conda activate neuroformer
cd ..

python neuroformer_train.py --dataset ibl \
                            --finetune  \
                            --loss_bprop wheel_speed whisker_energy \
                            --resume "./models/NF.15/Visnav_VR_Expt/ibl/Neuroformer/None/(state_history=6,_state=6,_stimulus=6,_behavior=6,_self_att=6,_modalities=(n_behavior=25))/25/pretrain/model.pt" \
                            --config ./configs/ibl/predict_behav.yaml
cd script