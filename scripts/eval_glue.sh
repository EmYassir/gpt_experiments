#!/bin/sh
echo "############## Starting training script... ";

## General
export NODE_RANK=0
export N_NODES=1
export N_GPU_NODE=2
export WORLD_SIZE=2
export MASTER_ADDR=127.0.0.1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export NUM_PROC=1

## CACHE
export HF_HOME=/media/nvme/yassir/huggingface/
export TRANSFORMERS_CACHE=$HF_HOME

## Datasets
export TASK_NAME=cola
export DATASET_NAME=glue
export DATASET_CONFIG_NAME=glue
## Distributed training
export CUDA_VISIBLE_DEVICES=5

## Model
export MODEL_TYPE=EleutherAI/gpt-j-6B


## Training hyper-parameters
export MAX_SEQ_LEN=64
export EVAL_BATCH_SIZE=1

## Training settings
export RUNNER=/home/yassir/gpt_project/runners/run_glue.py
export OUTPUT_DIR=/media/nvme/yassir/output/glue/$TASK/$MODEL_TYPE

## Training
python $RUNNER \
    --fp16 \
    --tokenizer_name $MODEL_TYPE \
    --model_name_or_path $MODEL_TYPE  \
    --cache_dir $HF_HOME \
    --dataset_name $DATASET_NAME \
    --dataset_config_name $DATASET_CONFIG_NAME \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR  \
    --logging_dir $OUTPUT_DIR/logs \
    --max_seq_length $MAX_SEQ_LEN \
    --dataloader_num_workers $NUM_PROC \
    --do_eval \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --overwrite_output_dir;

echo "<<<<<< Done!!! ";
