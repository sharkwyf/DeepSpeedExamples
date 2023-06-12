#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
export HF_HOME=/cpfs01/shared/LVLM/transformers
# rm -rf /tmp/data_files/*

# DeepSpeed Team
# ACTOR_MODEL_PATH=$1
# CRITIC_MODEL_PATH=$2
ACTOR_MODEL_PATH=/cpfs01/shared/LVLM/transformers/hub/llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348
CRITIC_MODEL_PATH=/cpfs01/user/wangyuanfu/workspace/sft/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/output_llama_7b_lora
ACTOR_ZERO_STAGE=$3
CRITIC_ZERO_STAGE=$4
OUTPUT=$5
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=3
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=3
fi
mkdir -p $OUTPUT

Num_Padding_at_Beginning=0 # this is model related

Actor_Lr=9.65e-6
Critic_Lr=5e-6

deepspeed --master_port 12346 main.py \
   --data_path Dahoas/rm-static \
   --data_split 2,4,4 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 0 \
   --per_device_train_batch_size 4 \
   --per_device_mini_train_batch_size 4 \
   --generation_batch_numbers 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --actor_gradient_checkpointing \
   --disable_actor_dropout \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --output_dir $OUTPUT \
    # &> $OUTPUT/training.log
#    --enable_hybrid_engine \
