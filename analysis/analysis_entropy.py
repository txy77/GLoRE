import json
import os
import sys
import argparse
from tqdm import tqdm
import torch
import pdb
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde

from collections import defaultdict
from matplotlib.lines import Line2D

from accelerate import Accelerator


RE4R_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, RE4R_ROOT_PATH)

from src.utils.utils import read_jsonl, write_jsonl, transformodel_name2model_path, load_model_tokenizer, get_model_wrapper
from sklearn.manifold import TSNE

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def compute_matrix_entropy(Z, alpha=1):
    K = Z @ Z.T
    
    eigenvalues, _ = np.linalg.eig(K)
    eigenvalues = np.real(eigenvalues) 
    
    trace_K = np.trace(K)
    
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    sum_term = np.sum((eigenvalues / trace_K) ** alpha)
    
    if alpha == 1:
        S_alpha = -np.sum((eigenvalues / trace_K) * np.log(eigenvalues / trace_K))
    else:
        S_alpha = (1 / (1 - alpha)) * np.log(sum_term)
    
    return S_alpha


def write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama3.1-8b-instruct', help='path to config file')
    parser.add_argument('--module', type=str, default="hidden", help="inject vector to which module, attn / mlp / hidden")
    parser.add_argument('--layers', type=str, default="0,1,2,3", help="layer to inject")
    parser.add_argument('--alpha', type=float, default=1, help="alpha")
    return parser.parse_args()

def main():
    
    # ========== initialize ==========
    
    config = {
        "tok_pos": "last",
    }
    
    # get args
    args = get_args()
    accelerator = Accelerator()
    device = accelerator.device
    layers = args.layers.split(',')
    layers = [int(layer) for layer in layers]
    
    module = args.module
    config['module'] = module
    model_name = args.model_name
    model_path = transformodel_name2model_path(model_name)
    model, tokenizer, model_config, MODEL_CONFIG = load_model_tokenizer(model_path, accelerator, output_hidden_states=True, load_in_8bit=False)
    model_wrapper = get_model_wrapper(model_name, model, tokenizer, model_config, accelerator)
    
    demon_path = "data/demon/longcot/long_short_form_thought.jsonl"
    
    instruction = "Answer the following question step by step and put the final answer in \\boxed{}:\n"
    
    # ========== get vector ==========
    
    pred_res_vector_list = []
    short_form_vector_list = []
    long_form_vector_list = []
    
    demon_data = read_jsonl(demon_path)

    for i, d in tqdm(enumerate(demon_data), desc="get vector"):
        problem = d['problem']
        short_form = d['short_form_thought']
        long_form = d['long_form_thought']
        
        # # ========== process short form demonstration ==========
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
        
        short_form_context_vector_dict = model_wrapper.get_context_vector(short_form_all_latent_dicts, config)
        short_form_context_vector_dict = {key: value for key, value in short_form_context_vector_dict.items() if key in layers}
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
            
        long_form_context_vector_dict = model_wrapper.get_context_vector(long_form_all_latent_dicts, config)
        long_form_context_vector_dict = {key: value for key, value in long_form_context_vector_dict.items() if key in layers}
        long_form_vector_list.append(long_form_context_vector_dict)
        del long_form_all_latent_dicts

    res_dict = defaultdict(dict)

    for layer in tqdm(layers, desc='Evaluate Metric'):
        short_form_vec_layer_module_list = []
        long_form_vec_layer_module_list = []
        for i, (short_form_vec, long_form_vec) in enumerate(zip(short_form_vector_list, long_form_vector_list)):
            short_form_vec_layer_module_list.append(short_form_vec[layer][module].numpy())
            long_form_vec_layer_module_list.append(long_form_vec[layer][module].numpy())
        short_form_vec_layer_module_np = np.array(short_form_vec_layer_module_list)
        long_form_vec_layer_module_np = np.array(long_form_vec_layer_module_list)
        short_entropy = compute_matrix_entropy(short_form_vec_layer_module_np, args.alpha)
        long_entropy = compute_matrix_entropy(long_form_vec_layer_module_np, args.alpha)
        res_dict[layer] = {'short_entropy': float(short_entropy), 'long_entropy': float(long_entropy)}

    for layer, res in res_dict.items():
        print(layer, res)
    
    save_dir = f"exp-final/analysis/entropy/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    write_json(res_dict, f"{save_dir}/{module}_{args.layers}_entropy_{args.alpha}.json")

if __name__ == "__main__":
    set_seed(42)
    main()