#!/bin/bash

#SBATCH -p normal_q
#SBATCH -N 1  # this requests 1 node, 1 core. 
#SBATCH --mem=50G
#SBATCH -t 72:00:00
#SBATCH --account=nmayhall_group
## SBATCH --exclusive # this requests exclusive access to node for interactive jobs

module reset

source activate pyconda

cd $SLURM_SUBMIT_DIR

echo $LEVEL
echo $BATCH

python run_opt.py $LEVEL $BATCH

exit;
