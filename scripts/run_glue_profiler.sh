#!/bin/bash
## General
export MASTER_ADDR=127.0.0.1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export UPPER_BOUND=63000                     #Upper Range
export LOWER_BOUND=20000                     #Lower Range
export DIFF=$((UPPER_BOUND-LOWER_BOUND+1))   #+1 to inlcude upper limit
export MASTER_PORT=$(($(($RANDOM%$DIFF))+LOWER_BOUND))

export RANK=0
export NODE_RANK=0
export LOCAL_RANK=0
export USE_FP16=true
export N_NODES=1
export N_GPU_NODE=1
export WORLD_SIZE=1
## Distributed training
export CUDA_VISIBLE_DEVICES=1
export USE_DEEPSPEED=true


## CACHE
export HF_HOME=/media/data/yassir/huggingface/
export TRANSFORMERS_CACHE=$HF_HOME

## Datasets
export DATASET=super_glue
#export TASK_NAME=CoLA
export TASK_NAME=cola



## Model
export MODEL_TYPE=EleutherAI/gpt-j-6B
export MODEL_NAME_OR_PATH=EleutherAI/gpt-j-6B
#export MODEL_TYPE=gpt2
#export MODEL_NAME_OR_PATH=/media/data/yassir/original_models/gpt2-xl


## Training hyper-parameters
export EPOCHS=10
export MAX_SEQ_LEN=128
export BATCH_SIZE=2
export EVAL_BATCH_SIZE=1
export GRAD_ACC=1
export SEED=42

## Training settings
export RUNNER=/home/yassir/gpt_project/runners/run_glue_forward_backward_profiler.py
export OUTPUT_DIR=/media/data/yassir/output/$MODEL_NAME_OR_PATH/$DATASET/$TASK_NAME

## Training
python $RUNNER \
    --seed $SEED \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --cache_dir $HF_HOME \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $EPOCHS \
    --max_length $MAX_SEQ_LEN \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE 

echo "<<<<<< Done!!! ";
