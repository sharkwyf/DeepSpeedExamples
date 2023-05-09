
NFS_PATH=/mnt/nfs2/

docker run --gpus all --ipc=host --network=bridge --expose 80 --rm -itd\
    --cap-add SYS_ADMIN --device /dev/fuse \
    --mount src=$NFS_PATH,dst=$NFS_PATH,type=bind \
    --name deepspeed \
    --env NFS_PATH=$NFS_PATH \
    -w /home/user sharkwyf/deepspeed:latest \
    bash -c "
    bash
    sleep 1000000000000000

    # example of use
    echo 'alias proxy_on=\"export http_proxy=http://172.16.1.135:3128/; export https_proxy=http://172.16.1.135:3128/; export HTTP_PROXY=http://172.16.1.135:3128/; export HTTPS_PROXY=http://172.16.1.135:3128/\"' >> ~/.bashrc
    echo 'alias proxy_off=\"unset http_proxy;unset https_proxy;unset HTTP_PROXY;unset HTTPS_PROXY;\"' >> ~/.bashrc
    cp ${NFS_PATH}/wangyuanfu/.netrc ~/
    
    export PROJECT_PATH=${NFS_PATH}/wangyuanfu/workspace/sft
    export HF_HOME=${NFS_PATH}/yangchao/HF/huggingface
    
    # training
    proxy_on
    cd ${PROJECT_PATH}/applications/DeepSpeed-Chat/training/step1_supervised_finetuning
    # clear && rm -rf /tmp/* && CUDA_VISIBLE_DEVICES=7 bash training_scripts/single_node/run_1.3b_lora_with_pretrain_loss.sh
    clear && rm -rf /tmp/* && bash training_scripts/single_node/run_6.7b_lora.sh

    # evaluation
    cd ${PROJECT_PATH}/applications/DeepSpeed-Chat/training/step1_supervised_finetuning
    CUDA_VISIBLE_DEVICES=7 python prompt_eval.py \
        --model_name_or_path_baseline facebook/opt-1.3b \
        --model_name_or_path_finetune ${PROJECT_PATH}/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output/
    
    # chat
    cd ${PROJECT_PATH}/applications/DeepSpeed-Chat/
    CUDA_VISIBLE_DEVICES=7 python chat.py --path='${PROJECT_PATH}/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output/'
    "

