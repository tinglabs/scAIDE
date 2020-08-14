#!/usr/bin/env bash
#SBATCH -J sim_sparse_raw
#SBATCH -o /home/yhuang/sim_sparse_raw.txt
#SBATCH -n 20
#SBATCH -N 1
#SBATCH -t 72:00:00

AIDE_PATH='/home/yhuang/LSH/project/aide'
RPH_KMEANS_PATH='/home/yhuang/LSH/project/rph_kmeans'
SC_CLUSTER_PATH='/home/yhuang/LSH/scAIDE/sc_cluster/src'

export PYTHONPATH=${AIDE_PATH}:$PYTHONPATH
export PYTHONPATH=${RPH_KMEANS_PATH}:$PYTHONPATH

echo 'PYTHONPATH:' ${PYTHONPATH}
cd $SC_CLUSTER_PATH
#python dim_reduce_clt_eval.py
python dim_origin_clt_eval.py