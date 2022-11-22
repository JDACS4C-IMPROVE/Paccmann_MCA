#!/bin/bash

#############################################################################################################
### THIS IS A TEMPLATE FILE. SUBSTITUTE '???' ENTRIES WITH THE APPROPRiATE INFORMATION FOR YOUR CONTAINER ###
#############################################################################################################


# arg 1 CUDA_VISIBLE_DEVICES
# arg 2 CANDLE_DATA_DIR
# arg 3 CANDLE_CONFIG

### Path to your CANDLEized model's main Python script###
CANDLE_MODEL=/usr/local/Paccmann_MCA/train.py

# Reading command line arguments
CUDA_VISIBLE_DEVICES=$1
CANDLE_DATA_DIR=$2
CANDLE_CONFIG=$3

# Check if CANDLE_CONFIG exists if user passed this parameter and prepare 
if [ ! -z "${CANDLE_CONFIG}" ]; then
        if [ ! -f "$CANDLE_CONFIG" ]; then
	    echo "Usage: train.sh CUDA_VISIBLE_DEVICES CANDLE_DATA_DIR CANDLE_CONFIG"
			exit -1
		else
			CMD="python3 ${CANDLE_MODEL} --config_file ${CANDLE_CONFIG}"
        fi
else
	CMD="python3 ${CANDLE_MODEL}"
fi

# Display runtime arguments
echo "using container "
echo "using CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
echo "using CANDLE_DATA_DIR ${CANDLE_DATA_DIR}"
echo "using CANDLE_CONFIG ${CANDLE_CONFIG}"
echo "running command ${CMD}"

# Set up environmental variables and execute model
source /opt/conda/bin/activate /usr/local/conda_envs/Paccmann_MCA
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} CANDLE_DATA_DIR=${CANDLE_DATA_DIR} $CMD
