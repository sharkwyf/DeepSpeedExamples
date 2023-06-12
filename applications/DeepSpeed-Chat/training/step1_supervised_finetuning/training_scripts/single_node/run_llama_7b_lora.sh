#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
export HF_HOME=/cpfs01/shared/LVLM/transformers
export LLAMA_PATH=/cpfs01/shared/LVLM/transformers/hub/llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348
rm -rf /tmp/data_files/*

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_llama7b_t
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi
mkdir -p $OUTPUT
#./training_scripts/single_node/llama-7b 
#bash training_scripts/single_node/run_llama_7b.sh
#Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets Dahoas/rm-static \
#stanfordnlp/SHP
# Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets openai/webgpt_comparisons
# Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets openai/webgpt_comparisons stanfordnlp/SHP
deepspeed main.py \
   --data_path Dahoas/full-hh-rlhf  \
   --data_split 2,4,4 \
   --model_name_or_path $LLAMA_PATH  \
   --per_device_train_batch_size 6 \
   --per_device_eval_batch_size 6 \
   --max_seq_len 512 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 2 \
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
   --use_coh \
