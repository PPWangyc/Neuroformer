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

eid=${1}

module load gpu
module load slurm
. ~/.bashrc
conda activate neuroformer
cd ..

resume_path="./models/NF.15/Visnav_VR_Expt/ibl/Neuroformer/None/(state_history=6,_state=6,_stimulus=6,_behavior=6,_self_att=6,_modalities=(n_behavior=25))/25/${eid}/pretrain/model.pt"

echo "Finetuning model with resume path: ${resume_path}"
python neuroformer_train.py --dataset ibl \
                            --finetune  \
                            --loss_bprop wheel_speed whisker_energy \
                            --resume ${resume_path} \
                            --config ./configs/ibl/predict_behav.yaml \
                            --eid ${eid}
cd script