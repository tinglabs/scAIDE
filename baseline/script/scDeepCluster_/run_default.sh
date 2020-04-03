#!/usr/bin/env bash

REPEAT=5
PROJ_PATH="/home/yhuang/LSH/baseline/project/scDeepCluster"
CODE_PATH="${PROJ_PATH}/code"
DATA_PATH="${PROJ_PATH}/scRNA_seq_data"
LAYERS="256,64,32" #"1024,512,256"
GPU_USE=1

DATA_NAME="10X_PBMC"
for i in $(seq 1 $REPEAT); do
  echo "$DATA_NAME $i ====================================="
  python ${CODE_PATH}/scDeepCluster.py --data_file ${DATA_PATH}/10X_PBMC.h5 --n_clusters 8 --gpu $GPU_USE --layers $LAYERS
done

DATA_NAME="mouse_bladder_cell"
for i in $(seq 1 $REPEAT); do
  echo "$DATA_NAME $i ====================================="
  python ${CODE_PATH}/scDeepCluster.py --data_file ${DATA_PATH}/mouse_bladder_cell.h5 --n_clusters 16 --gpu $GPU_USE --layers $LAYERS
done

DATA_NAME="mouse_ES_cell"
for i in $(seq 1 $REPEAT); do
  echo "$DATA_NAME $i ====================================="
  python ${CODE_PATH}/scDeepCluster.py --data_file ${DATA_PATH}/mouse_ES_cell.h5 --n_clusters 4 --gpu $GPU_USE --layers $LAYERS
done

DATA_NAME="worm_neuron_cell"
for i in $(seq 1 $REPEAT); do
  echo "$DATA_NAME $i ====================================="
  python ${CODE_PATH}/scDeepCluster.py --data_file ${DATA_PATH}/worm_neuron_cell.h5 --n_clusters 10 --gpu $GPU_USE --layers $LAYERS
done

DATA_NAME="Shekhar_mouse_retina"
for i in $(seq 1 $REPEAT); do
  echo "$DATA_NAME $i ====================================="
  python ${CODE_PATH}/scDeepCluster.py --data_file ${DATA_PATH}/large_real_datasets/Shekhar_mouse_retina_raw_data.h5 --n_clusters 19 --gpu $GPU_USE --layers $LAYERS
done

DATA_NAME="PBMC_68k"
for i in $(seq 1 $REPEAT); do
  echo "$DATA_NAME $i ====================================="
  python ${CODE_PATH}/scDeepCluster.py --data_file ${DATA_PATH}/large_real_datasets/PBMC_68k.h5 --n_clusters 10 --gpu $GPU_USE --layers $LAYERS
done

DATA_NAME="sc_brain"
for i in $(seq 1 $REPEAT); do
  echo "$DATA_NAME $i ====================================="
  python ${CODE_PATH}/scDeepCluster.py --data_file ${DATA_PATH}/large_real_datasets/sc_brain.h5 --n_clusters 7 --pretrain_epochs 40 --gpu $GPU_USE --layers $LAYERS
done

#sparsity=(60 70 75 80 85 90 93)
#for sp in ${sparsity[@]}; do
#  DATA_NAME="sim_sparsity_${sp}"
#  for i in $(seq 1 $REPEAT); do
#    echo "$DATA_NAME $i ====================================="
#    python ${CODE_PATH}/scDeepCluster.py --data_file ${DATA_PATH}/sim_sparsity/${DATA_NAME}.h5 --n_clusters 10 --gpu $GPU_USE --layers $LAYERS
#  done
#done

#REPEAT=3
#n_samples_list=(1000 5000 10000 50000 100000)
#for n_samples in ${n_samples_list[@]}; do
#  DATA_NAME="1M_neurons-${n_samples}-samples"
#  for i in $(seq 1 $REPEAT); do
#    echo "$DATA_NAME $i ====================================="
#    python ${CODE_PATH}/scDeepCluster.py --data_file ${DATA_PATH}/1M_neurons_samples/${DATA_NAME}.h5 --n_clusters 10 --gpu $GPU_USE --layers $LAYERS
#  done
#done




