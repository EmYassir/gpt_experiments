#!/bin/bash
echo "############## Starting training script... ";

## General
export MASTER_ADDR=127.0.0.1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export NUM_PROC=1

## Deepspeed
export INCLUDE="localhost:5"
export DS_HOSTFILE=/home/yassir/gpt_project/network/hostfile
export DS_CONFIG=/home/yassir/gpt_project/configs/zero-3.json
export UPPER_BOUND=63000                     #Upper Range
export LOWER_BOUND=20000                     #Lower Range
export DIFF=$((UPPER_BOUND-LOWER_BOUND+1))   #+1 to inlcude upper limit
export MASTER_PORT=$(($(($RANDOM%$DIFF))+LOWER_BOUND))

## CACHE
export HF_HOME=/media/nvme/yassir/huggingface/
export TRANSFORMERS_CACHE=$HF_HOME

## Datasets
export TASK_NAME=cola
export DATASET_NAME=glue
export DATASET_CONFIG_NAME=glue

## Distributed training
export CUDA_VISIBLE_DEVICES=0

## Model
export MODEL_TYPE=EleutherAI/gpt-j-6B


## Evaluation hyper-parameters
export MAX_SEQ_LEN=32
export EVAL_BATCH_SIZE=1

## Training settings
export RUNNER=/home/yassir/gpt_project/runners/run_glue.py
export OUTPUT_DIR=/media/nvme/yassir/output/gpt_project/$DATASET_NAME/$TASK_NAME/$MODEL_TYPE

## Training
deepspeed  --hostfile $DS_HOSTFILE --include=$INCLUDE --master_port=$MASTER_PORT $RUNNER  \
    --deepspeed  $DS_CONFIG \
    --fp16 \
    --tokenizer_name $MODEL_TYPE \
    --model_name_or_path $MODEL_TYPE  \
    --cache_dir $HF_HOME \
    --dataset_name $DATASET_NAME \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR  \
    --logging_dir $OUTPUT_DIR/logs \
    --dataloader_num_workers $NUM_PROC \
    --max_seq_length $MAX_SEQ_LEN \
    --do_eval \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --overwrite_output_dir;

echo "<<<<<< Done!!! ";
