#!/usr/bin/env bash
#SBATCH -J ZIFA_2
#SBATCH -o /home/yhuang/ZIFA_2.txt
#SBATCH -n 20
#SBATCH -N 1
#SBATCH -t 72:00:00

PROJ_PATH='/home/yhuang/LSH/project/baseline/project/ZIFA'
BASELINE_PATH='/home/yhuang/LSH/project/baseline'

export PYTHONPATH=${PROJ_PATH}:$PYTHONPATH
export PYTHONPATH=${BASELINE_PATH}:$PYTHONPATH

echo 'PYTHONPATH:' ${PYTHONPATH}
python --version
python $BASELINE_PATH/script/ZIFA_/run.py --hidden_dim 2 --cell_norm 1
