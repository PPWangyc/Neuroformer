#!/bin/bash

# run 10 single session

while IFS= read -r line
do
    echo "Submit Pipeline job on ses eid: $line"
    sbatch run_pipeline.sh $line
done < "../data/train_eids.txt"
