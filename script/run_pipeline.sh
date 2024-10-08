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

# script to run pretrain, finetune, and evaluate on IBL dataset
# session eid
eid=${1}

source pretrain.sh ${eid}

source finetune.sh ${eid}

# use pretrain model for spike inference
source inference.sh ${eid} pretrain
# use behavior finetune model for behavior inference
source inference.sh ${eid} finetune