#!/bin/bash

# Launches model for preprocessing from within the container


####################################################################
## Exceute preprocess script for your model with the CMD arguments ##
####################################################################
## preprocess="/usr/local/Paccmann_MCA/preprocess.py

IMPROVE_MODEL_DIR=${IMPROVE_MODEL_DIR:-$( dirname -- "$0" )}
CANDLE_MODEL=preprocess_csa.py
CANDLE_MODEL=${IMPROVE_MODEL_DIR}/${CANDLE_MODEL}

CUDA_VISIBLE_DEVICES=$1
CANDLE_DATA_DIR=$2
SPLIT=$3
TRAIN_SOURCE=$4
TEST_SOURCE=$5

CMD="python3 ${CANDLE_MODEL}"


echo "using container "
echo "using CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
echo "using CANDLE_DATA_DIR ${CANDLE_DATA_DIR}"
echo "running command ${CMD}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} CANDLE_DATA_DIR=${CANDLE_DATA_DIR} SPLIT=${SPLIT} TRAIN_SOURCE=${TRAIN_SOURCE} TEST_SOURCE=${TEST_SOURCE} $CMD
