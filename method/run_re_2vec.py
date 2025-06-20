import json
import re
import os
import sys
import argparse
from tqdm import tqdm
import torch
import pdb
import random
import numpy as np
from collections import defaultdict
from copy import copy
import torch.nn.functional as F

from accelerate import Accelerator

RE4R_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)
sys.path.insert(0, RE4R_ROOT_PATH)

RE4R_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, RE4R_ROOT_PATH)

from utils.utils import read_jsonl, write_jsonl, transformodel_name2model_path, load_model_tokenizer, get_model_wrapper
from utils.evaluator import MATHEvaluator
from dataset import get_dataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()

    # ==== model & dataset ====
    parser.add_argument('--model_name', type=str, default='llama3.1-8b-instruct', help='path to config file')
    parser.add_argument('--dataset', type=str, help="string of dataset", default="mathoai")
    parser.add_argument('--max_generated_token', type=int, default=10000, help="max generated token")

    # ==== intervention ====
    parser.add_argument('--module', type=str, default="hidden", help="inject vector to which module, attn / mlp / hidden")
    parser.add_argument('--extract_pos', type=str, default="last", help="extract vector from which position, first / last / random")
    parser.add_argument('--layer', type=int, default=3, help="layer to inject")

    # ==== style_vector ====
    parser.add_argument('--style_inject_method', type=str, default="linear", help="inject method, linear / add / balance")
    parser.add_argument('--style_inject_pos', type=str, default="first", help="inject vector to which position, first / last / random")
    parser.add_argument('--style_strength', type=str, default="0.1,1.0", help="strength to inject")

    # ==== domain_vector ====
    parser.add_argument('--domain_inject_method', type=str, default="linear", help="inject method, linear / add / balance")
    parser.add_argument('--domain_inject_pos', type=str, default="last", help="inject vector to which position, first / last / random")
    parser.add_argument('--domain_strength', type=str, default="0.1,1.0", help="strength to inject")
    parser.add_argument("--domain_level_sample_num", type=int, default=5, help="number of similar domain examples to sample")

    return parser.parse_args()

def sub_vector(dict1, dict2):
    new_dict = {}
    for key in dict1.keys():
        new_dict[key] = {}
        for module in dict1[key].keys():
            new_dict[key][module] = dict1[key][module] - dict2[key][module]
    return new_dict


def get_domain_thought_bank(data, tokenizer, model, model_wrapper, device, instruction, domain_vec_config, layer):
    
    domain_thought_bank_list = []

    for case_id, d in tqdm(enumerate(data), desc="cache vector"):

        problem = d['problem']
        input_str = problem + instruction
        input_str = input_str.strip()

        thought = d['short_form_thought']

        only_problem_list = [
            {'role': 'user', 'content': input_str},
            {'role': 'assistant', 'content': ""}
        ]
        only_problem_str = tokenizer.apply_chat_template(only_problem_list, tokenize=False)

        tokenized_only_problem = tokenizer(only_problem_str, return_tensors='pt').to(device)
        only_problem_len = tokenized_only_problem['input_ids'].shape[1]

        thought_list = [
            {'role': 'user', 'content': input_str},
            {'role': 'assistant', 'content': thought}
        ]

        thought_str = tokenizer.apply_chat_template(thought_list, tokenize=False)
        
        domain_thought_all_latent_dicts = []
        with torch.no_grad():
            with model_wrapper.extract_latent():
                tokenized_thought = tokenizer(thought_str, return_tensors='pt').to(device)
                _ = model(**tokenized_thought)
            domain_thought_all_latent_dicts.append(model_wrapper.latent_dict)
        
        latent_dict = domain_thought_all_latent_dicts[0]
        module = domain_vec_config['module']
        latent_value = latent_dict[layer][module]
        latent_problem_value = latent_value[:, only_problem_len-2, :].squeeze()
        latent_thought_value = latent_value[:, -1, :].squeeze()

        domain_thought_bank_list.append({
            "problem_latent": latent_problem_value,
            "thought_latent": latent_thought_value,
        })

    return domain_thought_bank_list


def main():
    
    # ========== initialize ==========
    
    # get args
    args = get_args()
    accelerator = Accelerator()
    device = accelerator.device
    math_evaluator = MATHEvaluator()
    
    model_name = args.model_name
    dataset = args.dataset
    domain_level_sample_num = args.domain_level_sample_num
    model_path = transformodel_name2model_path(model_name)
    model, tokenizer, model_config, MODEL_CONFIG = load_model_tokenizer(model_path, accelerator, output_hidden_states=True, load_in_8bit=False)
    max_generated_token = args.max_generated_token
    model_wrapper = get_model_wrapper(model_name, model, tokenizer, model_config, accelerator)
    
    style_inject_pos = args.style_inject_pos
    domain_inject_pos = args.domain_inject_pos

    style_strength = args.style_strength.split(',')
    style_strength = [float(s) for s in style_strength]

    domain_strength = args.domain_strength.split(',')
    domain_strength = [float(s) for s in domain_strength]

    style_config = {
        "tok_pos": args.extract_pos,
        "inject_method": args.style_inject_method,
        "inject_pos": style_inject_pos,
        "strength": style_strength,
        'module': args.module,
    }

    domain_config = {
        "tok_pos": args.extract_pos,
        "inject_method": args.domain_inject_method,
        "inject_pos": domain_inject_pos,
        "strength": domain_strength,
        'module': args.module,
    }
    
    
    instruction = "Answer the following question step by step and put the final answer in \\boxed{}:\n"
    
    # ========== process test dataset ==========

    dataset = args.dataset
    BaseDataset = get_dataset(dataset)
    layer = args.layer

    all_problems = BaseDataset.all_problems
    all_inputs = BaseDataset.all_inputs
    all_labels = BaseDataset.all_labels
    all_pred_labels, accuracies = [], []
    
    
    # ========== get domain level thought bank ==========
    if args.dataset in ["mathoai", "aime", "amc"]:
        domain_level_thought_path = "data/demon/longcot/math_long_short_form_thought.jsonl"
        domain_level_data = read_jsonl(domain_level_thought_path)
        domain_level_thought_bank_list = get_domain_thought_bank(domain_level_data, tokenizer, model, model_wrapper, device, instruction, domain_config, layer)
    else:
        domain = args.dataset.split('-')[-1]
        domain_level_thought_path = f"data/demon/longcot/{domain}_long_short_form_thought.jsonl"
        domain_level_data = read_jsonl(domain_level_thought_path)
        domain_level_thought_bank_list = get_domain_thought_bank(domain_level_data, tokenizer, model, model_wrapper, device, instruction, domain_config, layer)

    # ========== get vector ==========
    
    short_form_vector_list = []
    long_form_vector_list = []
    
    long_form_demon_path = "data/demon/longcot/long_short_form_thought.jsonl"
    demon_data = read_jsonl(long_form_demon_path)
    
    for i, d in tqdm(enumerate(demon_data), desc="get vector"):
        problem = d['problem']
        short_form = d['short_form_thought']
        long_form = d['long_form_thought']
        
        # ========== process short form demonstration ==========
        short_form_demon_list = [
            {"role": "user", "content": (instruction + problem).strip()},
            {"role": "assistant", "content": short_form}
        ]
        short_form_demon_str = tokenizer.apply_chat_template(short_form_demon_list, tokenize=False)    
        
        short_form_all_latent_dicts = []
        with torch.no_grad():
            with model_wrapper.extract_latent():
                short_form_demon_token = tokenizer(short_form_demon_str, return_tensors='pt').to(device)
                _ = model(**short_form_demon_token)
            short_form_all_latent_dicts.append(model_wrapper.latent_dict)
        
        short_form_context_vector_dict = model_wrapper.get_context_vector(short_form_all_latent_dicts, style_config)
        short_form_context_vector_dict = {key: value for key, value in short_form_context_vector_dict.items() if int(key) == layer}
        short_form_vector_list.append(short_form_context_vector_dict)
        del short_form_all_latent_dicts
        
        # ========== process long form demonstration ==========
        long_form_demon_list = [
            {"role": "user", "content": (instruction + problem).strip()},
            {"role": "assistant", "content": long_form}
        ]
        long_form_demon_str = tokenizer.apply_chat_template(long_form_demon_list, tokenize=False)
        
        long_form_all_latent_dicts = []
        with torch.no_grad():
            with model_wrapper.extract_latent():
                long_form_demon_token = tokenizer(long_form_demon_str, return_tensors='pt').to(device)
                _ = model(**long_form_demon_token)
            long_form_all_latent_dicts.append(model_wrapper.latent_dict)
            
        long_form_context_vector_dict = model_wrapper.get_context_vector(long_form_all_latent_dicts, style_config)
        long_form_context_vector_dict = {key: value for key, value in long_form_context_vector_dict.items() if int(key) == layer}
        long_form_vector_list.append(long_form_context_vector_dict)
        del long_form_all_latent_dicts
    
    # ========== get contrast vector ==========
    
    contrast_vector_list = []
    aggregated_contrast_vector_dict = defaultdict(lambda: defaultdict(torch.Tensor))
    
    for i, (short_form, long_form) in enumerate(zip(short_form_vector_list, long_form_vector_list)):
        demon_contrast_vector_dict = defaultdict(lambda: defaultdict(torch.Tensor))
        module = style_config['module']
        short_form_layer = short_form[layer][module]
        long_form_layer = long_form[layer][module]
        contrast_vector = long_form_layer - short_form_layer
        demon_contrast_vector_dict[layer][module] = contrast_vector
        contrast_vector_list.append(demon_contrast_vector_dict)

    for demon_contrast_vector_dict in contrast_vector_list:
        for module in demon_contrast_vector_dict[layer]:
            contrast_vector = demon_contrast_vector_dict[layer][module]
            
            if aggregated_contrast_vector_dict[layer][module].numel() == 0:
                aggregated_contrast_vector_dict[layer][module] = contrast_vector
            else:
                aggregated_contrast_vector_dict[layer][module] += contrast_vector

    num_demons = len(contrast_vector_list)
    for module in aggregated_contrast_vector_dict[layer]:
        aggregated_contrast_vector_dict[layer][module] /= num_demons
    
    del contrast_vector_list, short_form_vector_list, long_form_vector_list

    # ========== inject domain-level thought vector ==========
    all_pred_labels, accuracies = [], []

    for sample_id, (input, label) in tqdm(enumerate(zip(all_inputs, all_labels)), desc="inject 2 vector"):

        input_list = [
            {"role": "user", "content": input},
            {"role": "assistant", "content": ""}
        ]
        input_str = tokenizer.apply_chat_template(input_list, tokenize=False)

        current_problem_latent_list = []
        with torch.no_grad():
            with model_wrapper.extract_latent():
                input_token = tokenizer(input_str, return_tensors='pt').to(device)
                _ = model(**input_token)
            current_problem_latent_list.append(model_wrapper.latent_dict)

        current_problem_vector_dict = model_wrapper.get_context_vector(current_problem_latent_list, domain_config)
        current_problem_vector_dict = {key: value for key, value in current_problem_vector_dict.items() if int(key) == layer}
        del current_problem_latent_list

        # ========== find domain-level thought ==========

        sim_domain_level_sample_list = []
        domain_problem_level_vec = [domain_sample['problem_latent'] for domain_sample in domain_level_thought_bank_list]
        domain_problem_embed = torch.stack(domain_problem_level_vec, dim=0)
        
        current_problem_vectors = current_problem_vector_dict[layer][domain_config['module']]
        current_problem_vectors = current_problem_vectors.unsqueeze(0)

        cos_sim = F.cosine_similarity(domain_problem_embed, current_problem_vectors)
        top_k_similarities, top_k_indices = torch.topk(cos_sim, domain_level_sample_num)

        for sim_id in top_k_indices:
            sim_domain_level_sample_list.append(domain_level_thought_bank_list[sim_id.item()])

        sim_domain_level_thought_list = [case['thought_latent'] for case in sim_domain_level_sample_list]

        sim_domain_level_thought_vector = torch.stack(sim_domain_level_thought_list, dim=0)  
        domain_level_mean_vector = sim_domain_level_thought_vector.mean(dim=0)  
        domain_level_vector_dict = {layer: {domain_config['module']: domain_level_mean_vector}}

        with model_wrapper.inject_latent((aggregated_contrast_vector_dict, domain_level_vector_dict), [layer], (style_config, domain_config)):
            input_token = tokenizer(input_str, return_tensors='pt').to(device)
            final_resposne_ids = model.generate(**input_token, max_new_tokens=args.max_generated_token, pad_token_id=tokenizer.eos_token_id, num_return_sequences=1)

        final_response = tokenizer.decode(final_resposne_ids[0][input_token["input_ids"].shape[1]:], skip_special_tokens=True)
        all_pred_labels.append(final_response)
        if args.dataset in ['aime', 'amc', 'mathoai']:
            sample_acc = math_evaluator.score(final_response, label)
            accuracies.append(sample_acc)
        else:
            pred_pattern = r'\\boxed\{(.*)\}'
            pred_matches = re.findall(pred_pattern, final_response)
            if pred_matches:
                pred_text = pred_matches[-1]
            ans_pattern = r'\([ABCD]\)'
            ans_choice = re.findall(ans_pattern, label)[0]
            ans_choice_wobucket = ans_choice.replace('(', '').replace(')', '')
            ans_content = label.replace(ans_choice, '').strip()
            if ans_choice == pred_text or ans_choice_wobucket == pred_text or ans_content in pred_text:
                accuracies.append(1)
            else:
                accuracies.append(0)

    results = []
    for input_str, pred_label, label, acc in zip(all_inputs, all_pred_labels, all_labels, accuracies):
        results.append({'problem': input_str, 'pred_label': pred_label, 'label': label, 'acc': acc})
    
    print('layer', layer, "domain_num", domain_level_sample_num, "acc", sum(accuracies)/len(accuracies))

    save_dir = f"exp-final/method/re/ours/{model_name}/{dataset}"
    os.makedirs(save_dir, exist_ok=True)
    write_jsonl(results, f"{save_dir}/{domain_level_sample_num}_{layer}_{'-'.join([str(s) for s in style_strength])}_{'-'.join([str(s) for s in domain_strength])}.jsonl")
    
if __name__ == "__main__":
    set_seed(42)
    main()
