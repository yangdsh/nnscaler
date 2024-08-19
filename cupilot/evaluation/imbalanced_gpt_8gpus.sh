#!/bin/bash
LOG_DIR=logs
LOGS=${LOG_DIR}/gpt
mkdir -p $LOGS

GPU=V100
NGPUS=8
TOTAL_GPUS=8

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH=.:$PYTHONPATH
export OMP_NUM_THREADS=4

# cube flags
export USE_JIT_PARSER=1
export DISABLE_INTER_RVD=1
export SINGLE_DEV_MODE=0
export TOTAL_GPUS=$TOTAL_GPUS
# export ASYNC_COMM=1

GBS=256 # global batch size


# ================================= gpt =============================

LAYERS=40
HEADS=8
HIDDEN=2048

POLICY=cupilot

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

if [ $SINGLE_DEV_MODE -gt 1 ]; then
python        examples/gpt/train.py \
        --mbs 4 --gbs $GBS --policy $POLICY --dev-mode 1\
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 2048\
        --max-pp 4 --max-dp 32 --dev0-mem-limit 16 --mem-frac 0.85\
        --recompute --db-cache gpt_${GPU}_db.json --save-spec temp.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$POLICY.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.log
else
#nsys profile -o gpt_profile_report 
torchrun --nproc_per_node=$NGPUS examples/gpt/train.py\
        --mbs 4 --gbs $GBS --policy $POLICY --dev-mode 0\
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 1024\
        --max-pp 4 --max-dp 32 --dev0-mem-limit 16 --mem-frac 0.85\
        --recompute --db-cache gpt_${GPU}_db.json --save-spec temp.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$POLICY.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.log
fi

# ========================================================================================