
echo SETTINGS

# General Settings
export PROCS=4
export PPN=4
export WALLTIME=03:00:00
export NUM_ITERATIONS=1
export POPULATION_SIZE=2

# GA Settings
export STRATEGY='mu_plus_lambda'
export OFF_PROP=0.5
export MUT_PROB=0.8
export CX_PROB=0.2
export MUT_INDPB=0.5
export CX_INDPB=0.5
export TOURNAMENT_SIZE=3

# Lambda Settings
export CANDLE_CUDA_OFFSET=1
export CANDLE_DATA_DIR=/tmp/pvasanthakumari

# Polaris Settings
# export QUEUE="debug"
# export CANDLE_DATA_DIR=/home/weaverr/output

