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
from utils.utils import set_random_seed
from utils.ds_utils import get_train_ds_config
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
import torch.distributed as distributed


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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval the finetued reward model")

    # reward_model
    parser.add_argument(
        "--critic_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    # LM
    parser.add_argument(
        "--actor_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    # second LM
    parser.add_argument(
        "--actor_model_name_or_path2",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        default="",
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )


    parser.add_argument(
        "--generate_method",
        type=int,
        default=1,
        help="generate_method of prompt"
    )
    args = parser.parse_args()
    return args

def get_cache_file_name(model_path, generate_method):
    cache_file_template = 'cache/cache_{}'
    cache_file = cache_file_template.format(model_path.replace('/', '_'))
    cache_file += str(generate_method)
    return cache_file

def prepare_eval_dataset(tokenizer, rank=0):
    eval_batch_size = 4
    # just -1
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
        rank,
        data_path,
        data_split,
        data_output_path,
        train_phase,
        seed,
        tokenizer,
        max_seq_len,
        sft_only_data_path=sft_only_data_path)
    prompt_eval_sampler = DistributedSampler(prompt_eval_dataset,rank=rank)

    data_collator = DataCollatorRLHF(max_seq_len, 1)
    prompt_eval_dataloader = DataLoader(prompt_eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=prompt_eval_sampler,
                                 batch_size=eval_batch_size)
    return prompt_eval_dataloader


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stop_contexts):
        super().__init__()
        self.stop_contexts = [stop.to("cuda") for stop in stop_contexts]
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stop_contexts:
            if all([stop in input_ids[i] for i in range(input_ids.shape[0])]):
                return True
        return False

def change_prompt(prompts, generate_method=1):
    if generate_method == 1:
        return prompts
    if generate_method == 2:
        new_prompts= []
        for i, prompt in enumerate(prompts):
            new_prompts.append(prompt + "good answer:\nAssistance: ")
        return new_prompts


def run_single_evaluation(args):
    sys.stdout = sys.__stdout__
    cache_name = get_cache_file_name(args.actor_model_name_or_path, args.generate_method)

    if args.local_rank <= 0:
        print("cache dir is {}".format(cache_name))
        if not os.path.exists(cache_name):
            os.mkdir(cache_name)
    # if args.local_rank <= 0:
    # sys.stdout =  open(os.path.join(cache_name,"eval.log"), 'a')
        
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()
    
    args.global_rank = torch.distributed.get_rank()


    ds_config = get_train_ds_config(offload=False,
                                    stage=2)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size() 

    set_random_seed(1234)

    #torch.distributed.barrier()
    #print(1)

    response_path = os.path.join(cache_name, "response.json")



    if not os.path.exists(response_path):


        if args.global_rank <= 0:
            print("Start generate response!")
            print("############################################################")

        tokenizer = AutoTokenizer.from_pretrained(args.actor_model_name_or_path, fast_tokenizer=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = create_hf_model(AutoModelForCausalLM, args.actor_model_name_or_path, tokenizer, ds_config)
        model.to(device)

        # stop_contexts = [ "<|endoftext|>"]
        # stop_contexts_ids = [
        #     tokenizer(stop_context, return_tensors='pt')['input_ids'].squeeze() for stop_context in stop_contexts]
        # stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stop_contexts=stop_contexts_ids)])



        eval_dataloader = prepare_eval_dataset(tokenizer, args.global_rank)
        
        responses = []

        def decode(batch):
            return tokenizer.batch_decode(batch,
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)



        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = batch['prompt']
            # print(batch.shape)
            info = {}
            prompts = decode(batch)
            info['prompts'] = list(prompts)
            prompts = change_prompt(prompts, args.generate_method)
            
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

            generation_config = GenerationConfig(
                temperature=0.8,
                num_beam_groups=4,
                diversity_penalty=1.0,
                num_beams=4,
                min_length=1,
                max_new_tokens=128,
                num_return_sequences=4,
            )

            output_sequences = model.generate(**inputs, 
                                        generation_config=generation_config,
                                    )


            
            output_sequences = torch.nn.functional.pad(output_sequences, (0, 1024 - output_sequences.shape[1]), mode='constant', value=tokenizer.pad_token_id)
            
            batch = torch.nn.functional.pad(batch, (0, 1024 - batch.shape[1]), mode='constant', value=tokenizer.pad_token_id).to(device)

            # print(batch.shape)

            torch.distributed.barrier()

            if args.global_rank == 0:
                gather_list = [torch.zeros((4*4, 1024),dtype=torch.int64).to(device) for _ in range(torch.distributed.get_world_size())]
                gather_list_batch = [torch.zeros((4, 1024), dtype=torch.int64).to(device) for _ in range(torch.distributed.get_world_size())]
            else:
                gather_list = []
                gather_list_batch = []
            # num_return_sequence * batch_size_per_device
            # print(1)
            torch.distributed.gather(output_sequences, gather_list, dst=0)

            # print(2)
            torch.distributed.gather(batch, gather_list_batch, dst=0)

            if args.global_rank == 0:
                # print(3)
                prompts = decode(torch.stack(gather_list_batch).view(-1, 1024))
                results = tokenizer.batch_decode( torch.stack(gather_list).view(-1, 1024),
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
                inputs_ = change_prompt(prompts)
                # print(len(prompts))

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
                   
        torch.distributed.barrier()
        
        if args.global_rank == 0:
            with open(response_path, 'w') as f:
                json.dump(responses, f, indent=4)

    torch.distributed.barrier()
    score_path = os.path.join(cache_name, "scores.json")


    # torch.distributed.barrier()

    if not os.path.exists(score_path) and args.global_rank <= 0:

        if args.global_rank <= 0:
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

            # if args.global_rank <= 0:
            #     print("==================Eval result============================")
            #     print("prompt: ", prompt)
            #     print( [f"\nresponse {i}: " +response[len(prompt):] for i, response in enumerate(rw_input)])
            #     print()
            #     print("=============Scores========================")
            #     print("response score: ", score)
        if args.global_rank <= 0:
            with open(score_path, 'w') as f:
                json.dump(scores, f, indent=4)
            with open(os.path.join(cache_name, "score_info.json"),"w") as f:
                json.dump(infos, f, indent=4)
    if args.global_rank <= 0:
        with open(score_path) as f:
            scores = json.load(f)
        
        print(len(scores))
        print(np.mean(scores))


if __name__ == "__main__":
    args = parse_args()
    if args.actor_model_name_or_path2 == "":
        run_single_evaluation(args)
