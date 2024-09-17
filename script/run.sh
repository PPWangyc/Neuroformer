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
python neuroformer_train.py \
       --dataset ibl \
       --config configs/ibl/pretrain.yaml
cd script
conda deactivate