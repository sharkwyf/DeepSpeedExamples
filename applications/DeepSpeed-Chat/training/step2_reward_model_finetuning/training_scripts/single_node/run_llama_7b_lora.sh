#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
export HF_HOME=/cpfs01/shared/LVLM/transformers
export LLAMA_PATH=/cpfs01/shared/LVLM/transformers/hub/llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348
# export LLAMA_PATH=decapoda-research/llama-7b-hf
# rm -rf /tmp/data_files/*
# export CUDA_VISIBLE_DEVICES=0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi
mkdir -p $OUTPUT

deepspeed main.py \
   --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets \
   --data_split 2,4,4 \
   --model_name_or_path $LLAMA_PATH \
   --num_padding_at_beginning 0 \
   --per_device_train_batch_size 12 \
   --per_device_eval_batch_size 12 \
   --max_seq_len 512 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_train_epochs 1 \
   --disable_dropout \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --lora_dim 128 \
   --gradient_checkpointing \
   --lora_module_name layers. \
   --deepspeed \
   --output_dir $OUTPUT \
