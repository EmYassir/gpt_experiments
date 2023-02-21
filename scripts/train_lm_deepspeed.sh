#!/bin/bash
echo "############## Starting training script... ";

## General
export MASTER_ADDR=127.0.0.1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export NUM_PROC=8

## Deepspeed
export INCLUDE="localhost:5"
export DS_HOSTFILE=/home/yassir/gpt_project/network/hostfile
export DS_CONFIG=/home/yassir/gpt_project/configs/zero-3.json
#export MASTER_PORT=$((((RANDOM<<15)|RANDOM)%63001+2000))
export UPPER_BOUND=63000                     #Upper Range
export LOWER_BOUND=20000                     #Lower Range
export DIFF=$((UPPER_BOUND-LOWER_BOUND+1))   #+1 to inlcude upper limit
export MASTER_PORT=$(($(($RANDOM%$DIFF))+LOWER_BOUND))

## CACHE
export HF_HOME=/media/data/yassir/huggingface/
export TRANSFORMERS_CACHE=$HF_HOME

## Datasets
export TASK_NAME=wikitext-103
export DATASET_NAME=wikitext-103


## Model
export MODEL_TYPE=EleutherAI/gpt-j-6B

## Datasets
export TRAIN_FILE=/media/data/yassir/datasets/wikitext103/wiki.train.raw.txt
export VALIDATION_FILE=/media/data/yassir/datasets/wikitext103/wiki.valid.raw.txt

## Model
export MODEL_TYPE=EleutherAI/gpt-j-6B

## Training settings
export RUNNER=/home/yassir/gpt_project/runners/run_clm.py
export OUTPUT_DIR=/media/data/yassir/output/gpt_project/$DATASET_NAME/$MODEL_TYPE


## Training hyper-parameters
export EPOCHS=6
export BLOCK_SIZE=1024
export BATCH_SIZE=1
export EVAL_BATCH_SIZE=4
export GRAD_ACC=8

## Training
deepspeed  --hostfile $DS_HOSTFILE --include=$INCLUDE --master_port=$MASTER_PORT $RUNNER  \
    --deepspeed  $DS_CONFIG \
    --tokenizer_name $MODEL_TYPE \
    --model_name_or_path $MODEL_TYPE \
    --cache_dir $HF_HOME \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --logging_dir $OUTPUT_DIR/logs \
    --preprocessing_num_workers $NUM_PROC \
    --dataloader_num_workers $NUM_PROC \
    --fp16 \
    --do_train \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC \
    --do_eval \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --metric_for_best_model eval_loss \
    --evaluation_strategy "epoch" \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir;

echo "******************* DONE !!!";
