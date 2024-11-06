#!/bin/bash
LOG_DIR=logs
LOGS=${LOG_DIR}/gpt
mkdir -p $LOGS

GPU=V100
TOTAL_GPUS=8

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH=.:$PYTHONPATH
export OMP_NUM_THREADS=4

# cube flags
export USE_JIT_PARSER=1
export DISABLE_INTER_RVD=1
if [ "$TOTAL_GPUS" -ge 9 ]; then
export SINGLE_DEV_MODE=1
else
export SINGLE_DEV_MODE=0
fi
export TOTAL_GPUS=$TOTAL_GPUS
export VISUALIZE_PLAN=0
# export ASYNC_COMM=1

GBS=128 # global batch size


# ================================= gpt =============================

LAYERS=8
HEADS=32
HIDDEN=4096

POLICY=cupilot

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TORCH_DISTRIBUTED_DEBUG=OFF

if [ $SINGLE_DEV_MODE -gt 0 ]; then
echo "Single device mode"
python        examples/gpt/train.py \
        --mbs 4 --gbs $GBS --policy $POLICY --dev-mode 1\
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen $HIDDEN\
        --max-pp 4 --max-dp 32 --dev0-mem-limit 16 --mem-frac 0.85\
        --recompute --db-cache gpt_${GPU}_db.json --save-spec temp.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$POLICY.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.log
else
echo "Multi device mode"
#nsys profile -o gpt_profile_report --load-spec gpt-1-layers-2048-dim-8-heads.json
/home/ubuntu/miniconda3/envs/cupilot/bin/torchrun --nproc_per_node=$TOTAL_GPUS examples/gpt/train.py\
        --mbs 4 --gbs $GBS --policy $POLICY --dev-mode 0 --max-block-num 25\
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 1024\
        --max-pp 2 --max-dp 32 --dev0-mem-limit 16 --mem-frac 0.8\
        --recompute --db-cache gpt_${GPU}_db.json --save-spec temp.json\
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$POLICY.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.log
fi

# ========================================================================================