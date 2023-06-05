# in evaluation dir, run  'bash evaluation_scripts/eval.sh'
#export HF_HOME="/mnt/nfs2/yangchao/HF/huggingface"
export HF_HOME=/cpfs01/shared/LVLM/transformers
#ACTOR_MODEL_PATH="/mnt/nfs2/liuzhixuan/DeepspeedChat/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/output/actor"
#ACTOR_MODEL_PATH="/mnt/nfs2/wangyuanfu/workspace/sft/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output_6.7b_lora"
#ACTOR_MODEL_PATH="/mnt/nfs2/liuzhixuan/DeepspeedChat/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output"
#ACTOR_MODEL_PATH="facebook/opt-6.7b"

#ACTOR_MODEL_PATH="facebook/opt-1.3b"
#ACTOR_MODEL_PATH="/mnt/nfs2/wangyuanfu/workspace/sft/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output_6.7b_improve_2_4_4_loss_mask+1"

#ACTOR_MODEL_PATH="/mnt/nfs2/wangyuanfu/workspace/sft/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output_6.7b_improve_2_4_4_loss_mask_all"
#ACTOR_MODEL_PATH="/mnt/nfs2/wangyuanfu/workspace/sft/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output_6.7b_improve_v1"
#ACTOR_MODEL_PATH="/mnt/nfs2/wangyuanfu/workspace/sft/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output_6.7b_improve_2_4_4_first_only"
#ACTOR_MODEL_PATH="/mnt/nfs2/wangyuanfu/workspace/sft/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output_6.7b_official_2_4_4"
#ACTOR_MODEL_PATH="/mnt/nfs2/wangyuanfu/workspace/sft/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output_6.7b_improve_2_4_4_loss_on_reject_first_only"
#ACTOR_MODEL_PATH="/mnt/nfs2/liuzhixuan/DeepspeedChat/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output_6.7b_v3"
#ACTOR_MODEL_PATH="/mnt/nfs2/wangyuanfu/workspace/sft/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output_6.7b_improve_10_0_0_loss_on_reject_first_only"
#ACTOR_MODEL_PATH="/mnt/nfs2/wangyuanfu/workspace/sft/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output_6.7b_official_10_0_0"
#ACTOR_MODEL_PATH="/cpfs01/user/liuzhixuan/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output_6.7blora"
#ACTOR_MODEL_PATH="/cpfs01/user/liuzhixuan/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output_6.7blora_epoch16"
#ACTOR_MODEL_PATH="/cpfs01/user/liuzhixuan/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/output/actor"
#ACTOR_MODEL_PATH="/mnt/nfs2/wangyuanfu/workspace/sft/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output_6.7b_improve_2_4_4"
#ACTOR_MODEL_PATH="/mnt/nfs2/wangyuanfu/workspace/sft/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output_6.7b_improve_2_4_4_loss_on_reject"
#ACTOR_MODEL_PATH="/mnt/nfs2/wangyuanfu/workspace/sft/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output_6.7b_improve_10_0_0_first_only"
#ACTOR_MODEL_PATH="/mnt/nfs2/liuzhixuan/DeepspeedChat/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/output_6.7b_6gpu/actor"
ACTOR_MODEL_PATH="/cpfs01/user/liuzhixuan/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output_6.7blora_epoch12"

CRITIC_MODEL_PATH=""

#CRITIC_MODEL_PATH="Dahoas/gptj-rm-static"

#cache/cache__cpfs01_user_liuzhixuan_DeepSpeedExamples_applications_DeepSpeed-Chat_training_step1_supervised_finetuning_output_6.7blora_epoch121
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed eval.py \
    --critic_model_name_or_path=$CRITIC_MODEL_PATH \
    --actor_model_name_or_path=$ACTOR_MODEL_PATH \
    --generate_method=1