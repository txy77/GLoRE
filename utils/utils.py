import json
import os
import torch
import sys
import time

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

RE4R_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, RE4R_ROOT_PATH)

import utils.wrapper as wrapper

def read_jsonl(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
            
def transformodel_name2model_path(model_name):
    model_name2model_path = {
        "llama3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct/",
        "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
        "qwen2.5-32b-instruct": "Qwen/Qwen2.5-32B-Instruct",
    }
    model_path = model_name2model_path[model_name]
    return model_path

def load_model_tokenizer(model_path, accelerator, output_hidden_states=True, load_in_8bit=False):
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    # model
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", output_hidden_states=output_hidden_states, load_in_8bit=load_in_8bit, torch_dtype=torch.float32, trust_remote_code=True).eval()
    model = accelerator.prepare(model)
    # config
    model_config = AutoConfig.from_pretrained(model_path)
    
    if "llama" in model_path.lower():
        MODEL_CONFIG = {
            "head_num": model.config.num_attention_heads,
            "layer_num": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "name_or_path": model.config._name_or_path,
            "attn_hook_names": [f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.config.num_hidden_layers)],
            "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.config.num_hidden_layers)],
            "prepend_bos":True
        }
    elif "qwen" in model_path.lower():
        MODEL_CONFIG = {
            "head_num": model.config.num_attention_heads,
            "layer_num": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "name_or_path": model.config._name_or_path,
            "attn_hook_names": [f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.config.num_hidden_layers)],
            "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.config.num_hidden_layers)],
            "prepend_bos":True
        }
    
    return model, tokenizer, model_config, MODEL_CONFIG


def call_openai_server_func(prompt, model, client, labels=None, temperature=0, max_tokens=10000):    
    
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    
    try:
        response = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
        res_completion = response.choices[0].message.content
        return res_completion
    
    except OSError as e:
        retry_time = 5  # Adjust the retry time as needed
        print(f"Connection error occurred: {e}. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return call_openai_server_func(
            prompt, model=model, labels=labels, temperature=temperature
        )


def get_model_wrapper(model_name, model, tokenizer, model_config, accelerator):
    
    device = accelerator.device
    if 'llama' in model_name:
        model_wrapper = wrapper.LlamaWrapper(model, tokenizer, model_config, device)
    elif 'qwen' in model_name:
        model_wrapper = wrapper.QwenWrapper(model, tokenizer, model_config, device)
    elif 'gpt' in model_name:
        model_wrapper = wrapper.GPTWrapper(model, tokenizer, model_config, device)
    else:
        raise ValueError("only support llama or gpt!")
    return model_wrapper