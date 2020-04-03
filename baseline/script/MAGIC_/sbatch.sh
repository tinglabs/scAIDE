#!/usr/bin/env bash
#SBATCH -J MAGIC_sim_sparsity
#SBATCH -o /home/yhuang/MAGIC_sim_sparsity.txt
#SBATCH -n 20
#SBATCH -N 1
#SBATCH -t 72:00:00

PROJ_PATH='/home/yhuang/LSH/project/baseline/project/magic/python'
BASELINE_PATH='/home/yhuang/LSH/project/baseline'

export PYTHONPATH=${PROJ_PATH}:$PYTHONPATH
export PYTHONPATH=${BASELINE_PATH}:$PYTHONPATH

echo 'PYTHONPATH:' ${PYTHONPATH}
python --version
python $BASELINE_PATH/script/MAGIC_/run.py --magic_pca 256 --n_jobs 20
python $BASELINE_PATH/script/MAGIC_/run.py --magic_pca 100 --n_jobs 20
