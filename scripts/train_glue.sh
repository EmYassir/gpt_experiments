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
export NUM_PROC=4

## CACHE
export HF_HOME=/media/data/yassir/huggingface/
export TRANSFORMERS_CACHE=$HF_HOME

## Datasets
export TASK_NAME=cola
export DATASET_NAME=glue
export DATASET_CONFIG_NAME=glue

## Distributed training
export CUDA_VISIBLE_DEVICES=4

## Model
export MODEL_TYPE=EleutherAI/gpt-j-6B


## Training hyper-parameters
export EPOCHS=3
export MAX_SEQ_LEN=128
export BATCH_SIZE=1
export EVAL_BATCH_SIZE=1
export GRAD_ACC=64

## Training settings
export RUNNER=/home/yassir/gpt_project/runners/run_glue.py
export OUTPUT_DIR=/media/data/yassir/output/gpt_project/$DATASET_NAME/$TASK_NAME/$MODEL_TYPE

## Training
python $RUNNER \
    --fp16 \
    --tokenizer_name $MODEL_TYPE \
    --model_name_or_path $MODEL_TYPE  \
    --cache_dir $HF_HOME \
    --dataset_name $DATASET_NAME \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR  \
    --logging_dir $OUTPUT_DIR/logs \
    --do_train \
    --num_train_epochs $EPOCHS \
    --max_seq_length $MAX_SEQ_LEN \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC \
    --do_eval \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --metric_for_best_model eval_loss \
    --evaluation_strategy "epoch" \
    --overwrite_output_dir;

echo "<<<<<< Done!!! ";
