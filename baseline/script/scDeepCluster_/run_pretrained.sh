#!/usr/bin/env bash

REPEAT=5
PROJ_PATH="/home/yhuang/LSH/baseline/project/scDeepCluster"
CODE_PATH="${PROJ_PATH}/code"
DATA_PATH="${PROJ_PATH}/scRNA_seq_data"
WEIGHT_PATH="${PROJ_PATH}/model_weights"

DATA_NAME="10X_PBMC"
for i in $(seq 1 $REPEAT); do
  echo "$DATA_NAME $i ====================================="
  python ${CODE_PATH}/scDeepCluster.py --data_file ${DATA_PATH}/10X_PBMC.h5 --n_clusters 8 --ae_weights ${WEIGHT_PATH}/10X_PBMC_full_set/ae_weights.h5
done

DATA_NAME="mouse_bladder_cell"
for i in $(seq 1 $REPEAT); do
  echo "$DATA_NAME $i ====================================="
  python ${CODE_PATH}/scDeepCluster.py --data_file ${DATA_PATH}/mouse_bladder_cell.h5 --n_clusters 16 --ae_weights ${WEIGHT_PATH}/Mouse_bladder_cells_full_set/ae_weights.h5
done

DATA_NAME="mouse_ES_cell"
for i in $(seq 1 $REPEAT); do
  echo "$DATA_NAME $i ====================================="
  python ${CODE_PATH}/scDeepCluster.py --data_file ${DATA_PATH}/mouse_ES_cell.h5 --n_clusters 4 --ae_weights ${WEIGHT_PATH}/Mouse_ES_cells_full_set/ae_weights.h5
done

DATA_NAME="worm_neuron_cell"
for i in $(seq 1 $REPEAT); do
  echo "$DATA_NAME $i ====================================="
  python ${CODE_PATH}/scDeepCluster.py --data_file ${DATA_PATH}/worm_neuron_cell.h5 --n_clusters 10 --ae_weights ${WEIGHT_PATH}/Worm_neuron_cells_full_set/ae_weights.h5
done






