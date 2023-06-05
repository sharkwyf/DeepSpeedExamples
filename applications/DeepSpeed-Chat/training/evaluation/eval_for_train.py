#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import torch
import deepspeed
import numpy as np
import json

from transformers import AutoTokenizer
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from reward_func import create_reward_fn

from utils.model.model_utils import create_hf_model
from utils.data.data_utils import create_prompt_dataset, MiniDataset, get_unsupervised_data, DataCollatorReward, DataCollatorRLHF, get_raw_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

from utils.model.model_utils import create_critic_model
from utils.utils import to_device
from tqdm import tqdm


def get_cache_file_name(model_path, epoch, generate_method=1):
    model_path = os.path.abspath(model_path)
    cache_file_template = '/cpfs01/user/liuzhixuan/DeepSpeedExamples/applications/DeepSpeed-Chat/training/evaluation/cache/cache_{}'
    cache_file = cache_file_template.format(model_path.replace('/', '_'))
    cache_file += str(generate_method) + "_epoch{}".format(epoch)
    return cache_file

def change_prompt(prompts, generate_method=1):
    if generate_method == 1:
        return prompts
    if generate_method == 2:
        for i, prompt in enumerate(prompts):
            prompts[i] = prompts[i][:-10] + "good answer:\nAssistance: "  
        return prompts



def prepare_eval_dataset(tokenizer):

    eval_batch_size = 4
    # just -1
    local_rank = 0
    # data_path: you need add what you want to test.
    data_path = ["Dahoas/rm-static",]
    # 
    data_output_path = "/tmp/data_files/"
    
    # train_phase = 1: prompt and chosen response
    # train_phase = 2: prompt and chosen response, prompt and rejected response
    # train_phase = 3: only prompt
    train_phase = 3
    
    # similar with step1 sft run_1.3b.sh
    seed = 1234
    data_split = "8,1,1"
    max_seq_len = 512
    sft_only_data_path = []

    _, prompt_eval_dataset = create_prompt_dataset(
        local_rank,
        data_path,
        data_split,
        data_output_path,
        train_phase,
        seed,
        tokenizer,
        max_seq_len,
        sft_only_data_path=sft_only_data_path)
    prompt_eval_sampler = DistributedSampler(prompt_eval_dataset)

    data_collator = DataCollatorRLHF(max_seq_len, 1)
    prompt_eval_dataloader = DataLoader(prompt_eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=prompt_eval_sampler,
                                 batch_size=eval_batch_size)
    return prompt_eval_dataloader

def run_single_evaluation(model, tokenizer, model_output_path, epoch, rank, generate_method=1):
    model.eval()
    if rank <=0 :
        print("Start Evaluation!")
        print("############################################################")
    cache_name = get_cache_file_name(model_output_path, epoch, generate_method)
    if rank <=0 :
        print("cache dir is {}".format(cache_name))
    if not os.path.exists(cache_name) and rank <= 0:
        os.mkdir(cache_name)
    
    device = torch.device("cuda", rank)

    response_path = os.path.join(cache_name, "response.json")
    # if rank <=0 :
    #     print("Start generate response!")
    #     print("############################################################")

    
    
    generation_config = GenerationConfig(
        temperature=0.8,
        num_beam_groups=4,
        diversity_penalty=1.0,
        num_beams=4,
        min_length=1,
        max_new_tokens=128,
        num_return_sequences=4,
    )

    eval_dataloader = prepare_eval_dataset(tokenizer)

    responses = []

    def decode(batch):
        return tokenizer.batch_decode(batch,
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)


    for step, batch in enumerate(tqdm(eval_dataloader)):
        # print(f"rank:{rank} step:{step} start!")
        batch = batch['prompt']

        info = {}
        prompts = tokenizer.batch_decode(batch,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
        info['prompts'] = list(prompts)
        prompts = change_prompt(prompts, generate_method)
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        
        # print(f"rank:{rank} step:{step} start generate!")

        # print("model.generate_pre")
        torch.distributed.barrier()

        output_sequences = model.generate(**inputs, 
                                    generation_config=generation_config,
                                    #stopping_criteria=stopping_criteria,
                                )
        # print(f"rank:{rank} step:{step} end generate!")

        # print("model.generate_fix")
        torch.distributed.barrier()

        output_sequences = torch.nn.functional.pad(output_sequences, (0, 1024 - output_sequences.shape[1]), mode='constant', value=tokenizer.pad_token_id)
        
        batch = torch.nn.functional.pad(batch, (0, 1024 - batch.shape[1]), mode='constant', value=tokenizer.pad_token_id).to(device)

        # torch.distributed.barrier()

        if rank == 0:
            # num_return_sequence * batch_size_per_device
            gather_list = [torch.zeros((4*4, 1024),dtype=torch.int64).to(device) for _ in range(torch.distributed.get_world_size())]
            gather_list_batch = [torch.zeros((4, 1024), dtype=torch.int64).to(device) for _ in range(torch.distributed.get_world_size())]
        else:
            gather_list = []
            gather_list_batch = []

        # print(f"rank:{rank} step:{step} start gather 1!")

        torch.distributed.gather(output_sequences, gather_list, dst=0)

        # print(f"rank:{rank} step:{step} start gather 2!")

        torch.distributed.gather(batch, gather_list_batch, dst=0)


        # print(f"rank:{rank} step:{step} start chulishuju!")

        if rank == 0:
            prompts = decode(torch.stack(gather_list_batch).view(-1, 1024))
            results = tokenizer.batch_decode( torch.stack(gather_list).view(-1, 1024),
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
            inputs_ = change_prompt(prompts)

            for i in range( len(prompts)):
                prompt = prompts[i]
                response = results[4*i:4*i+4]
                response_ = {}
                #input: 真实输入
                response_['input'] = inputs_[i]
                #output: 真实输出 
                response_['output'] = [s[len(inputs_[i]):] for s in response]
                #数据集输入
                response_['prompt'] = prompts[i]
                responses.append(response_)
        # print(f"rank:{rank} step:{step} end step!")
        torch.distributed.barrier()
                   
    torch.distributed.barrier()

    if rank <= 0:
        with open(response_path, 'w') as f:
            json.dump(responses, f, indent=4)

    score_path = os.path.join(cache_name, "scores.json")

    if rank <=0 :
        print("Start generate reward!")
        print("############################################################")

        responses = []
        with open(response_path) as f:
            responses = json.load(f)

        reward_func = create_reward_fn()

        scores = []
        infos = []
        prompts_dict = {}

        for response_ in tqdm(responses):
            
            prompt = response_['prompt']

            if prompt in prompts_dict:
                continue
            prompts_dict[prompt] = True
            
            responses = response_['output']
            rw_input = []
            for response in responses:
                response = response.split("<|endoftext|>")[0]
                response = response.split("Human:")[0]
                response = prompt + response
                rw_input.append(response)

            score = reward_func(rw_input)

            info = {}
            info['prompt'] = prompt
            info['output'] = response_['output']
            info['clean_output'] = [s[len(prompt):] for s in rw_input]
            info['score'] = score
            infos.append(info)

            scores.append(score)

        with open(score_path, 'w') as f:
            json.dump(scores, f, indent=4)
        with open(os.path.join(cache_name, "score_info.json"),"w") as f:
            json.dump(infos, f, indent=4)
        print("average reward: " ,np.mean(scores))
        return np.mean(scores)
    

if __name__ == "__main__":
    pass
