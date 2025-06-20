import math
import string
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from functools import reduce
import numpy as np
import os
from collections import defaultdict
import sys

RE4R_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, RE4R_ROOT_PATH)

from peft import get_peft_model, PromptTuningConfig
from utils import utils
import pdb
from tqdm import tqdm
import time

class ModelWrapper(nn.Module):
    def __init__(self, model, tokenizer, model_config, device):
        super().__init__()
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.device = device
        self.num_layers = self._get_layer_num()
        self.latent_dict = {}
        self.linear_coef = None
        self.inject_layers = None
        print(f"The model has {self.num_layers} layers:")

    def reset_latent_dict(self):
        self.latent_dict = {}
            
    @contextmanager
    def extract_latent(self):
        handles = []
        self.latent_dict = defaultdict(dict)
        try:
            # attach hook
            for layer_idx in range(self.num_layers):
                handles.append(
                    self._get_nested_attr(self._get_arribute_path(layer_idx, 'attn')).register_forward_hook(self.extract_hook_func(layer_idx, 'attn')))
                handles.append(
                    self._get_nested_attr(self._get_arribute_path(layer_idx, 'mlp')).register_forward_hook(self.extract_hook_func(layer_idx, 'mlp')))
                handles.append(
                    self._get_nested_attr(self._get_arribute_path(layer_idx, 'hidden')).register_forward_hook(self.extract_hook_func(layer_idx, 'hidden')))
            yield
        finally:
            # remove hook
            for handle in handles:
                handle.remove()

    def extract_hook_func(self, layer_idx, target_module):
        if layer_idx not in self.latent_dict:
            self.latent_dict[layer_idx] = {}
        def hook_func(module, inputs, outputs):
            if type(outputs) is tuple:
                outputs = outputs[0]
            self.latent_dict[layer_idx][target_module] = outputs.detach().cpu()
        return hook_func
    
    @contextmanager
    def inject_latent(self, context_vector_dict, inject_layers, config):
        handles = []
        # assert self.inject_layers is not None, "inject_layers is not set!"
        if isinstance(context_vector_dict, tuple):
            style_inject_method, knowledge_inject_method = config[0]['inject_method'], config[1]['inject_method']
            style_inject_pos, knowledge_inject_pos = config[0]['inject_pos'], config[1]['inject_pos']
            style_strength, knowledge_strength = config[0]['strength'], config[1]['strength']
        else:
            inject_method = config['inject_method']
            inject_pos = config['inject_pos']
            strength = config['strength']
        try:
            # attach hook
            if isinstance(context_vector_dict, tuple):
                for layer_idx, layer in enumerate(inject_layers):
                    module = config[0]['module']
                    style_context_vector_container = [context_vector_dict[0][layer][module].to(self.device)]
                    inject_func = self.inject_hook_func(style_context_vector_container, style_strength, style_inject_method, style_inject_pos)
                    handles.append(
                        self._get_nested_attr(self._get_arribute_path(layer, module)).
                        register_forward_hook(inject_func)
                    )
                for layer_idx, layer in enumerate(inject_layers):
                    module = config[1]['module']
                    knowledge_context_vector_container = [context_vector_dict[1][layer][module].to(self.device)]
                    inject_func = self.inject_hook_func(knowledge_context_vector_container, knowledge_strength, knowledge_inject_method, knowledge_inject_pos)
                    handles.append(
                        self._get_nested_attr(self._get_arribute_path(layer, module)).
                        register_forward_hook(inject_func)
                    )
            else:
                for layer_idx, layer in enumerate(inject_layers):
                    module = config['module']
                    context_vector_container = [context_vector_dict[layer][module].to(self.device)]
                    inject_func = self.inject_hook_func(context_vector_container, strength, inject_method, inject_pos)
                    handles.append(
                        self._get_nested_attr(self._get_arribute_path(layer, module)).
                        register_forward_hook(inject_func)
                    )
            yield
            
        finally:
            # remove hook
            print(f"Removing {len(handles)} hooks...")
            for handle in handles:
                handle.remove()

    def inject_hook_func(self, context_vector_container, strength, inject_method, inject_pos):

        def hook_func(module, inputs, outputs):
            if type(outputs) is tuple:
                output = outputs[0]     
            else:
                output = outputs

            # init context_vector
            context_vector = context_vector_container[0]
            # expand inject_value to match output size (b, seq_len, d)
            context_vector = context_vector.expand(output.size(0), output.size(1), context_vector.size(-1))
            
            if inject_method == 'add':
                output = output + F.relu(strength) * context_vector
            elif inject_method == 'linear':
                if inject_pos == 'all':
                    output = strength[1] * output + strength[0] * context_vector
                else:
                    if inject_pos == 'last':
                        content = strength[1] * output[:, -1, :].clone().detach() + strength[0] * context_vector[:, -1, :]
                        content_norm = torch.norm(content, p=2, dim=-1, keepdim=True)
                        output_norm = torch.norm(output[:, -1, :], p=2, dim=-1, keepdim=True)
                        content = (content / content_norm) * output_norm
                        output[:, -1, :] = content
                    elif inject_pos == 'first':
                        content = strength[1] * output[:, 0, :].clone().detach() + strength[0] * context_vector[:, 0, :]
                        content_norm = torch.norm(content, p=2, dim=-1, keepdim=True)
                        output_norm = torch.norm(output[:, 0, :], p=2, dim=-1, keepdim=True)
                        content = (content / content_norm) * output_norm
                        output[:, 0, :] = content
                    else:
                        raise ValueError("only support all, last, first or random!")
                        
            elif inject_method == 'balance':
                a, b = strength[0], strength[1]
                output = ((1.0 - a) * output + a * context_vector) * b
            else:
                raise ValueError("only support add, linear or balance!")
            
            if type(outputs) is tuple:
                outputs = list(outputs)
                outputs[0] = output
                outputs = tuple(outputs)
            else:
                outputs = output
            return outputs
        return hook_func
    

    @contextmanager
    def replace_latent(self, context_vector_dict, target_layers, config):
        handles = []
        try:
            # attach hook
            for _, layer in enumerate(target_layers):
                for _, module in enumerate(config['module']):
                    context_vector_container = [context_vector_dict[layer][module].to(self.device)]
                    inject_func = self.replace_hook_func(context_vector_container)
                    handles.append(
                        self._get_nested_attr(self._get_arribute_path(layer, module)).
                        register_forward_hook(inject_func))
            yield
        finally:
            # remove hook
            print(f"Removing {len(handles)} hooks...")
            for handle in handles:
                handle.remove()

    def replace_hook_func(self, context_vector_container):
        def hook_func(module, inputs, outputs):
            if type(outputs) is tuple:
                output = outputs[0]     
            else:
                output = outputs
            # init context_vector
            context_vector = context_vector_container[0] # (hidden_size)
            
            # replace hidden states of last token position with context_vector
            for i in range(output.size(0)):
                output[i, end_idx, :] = context_vector
            
            if type(outputs) is tuple:
                outputs = list(outputs)
                outputs[0] = output
                outputs = tuple(outputs)
            else:
                outputs = output
                
            return outputs
        return hook_func
    

    def get_context_vector(self, all_latent_dicts, config):
        if len(all_latent_dicts) == 1:
            latent_dict = all_latent_dicts[0]
            output_dict = {}
            module = config['module']
            for layer, sub_dict in latent_dict.items():
                output_dict[layer] = {}
                latent_value = sub_dict[module]
                if config['tok_pos'] == 'last':
                    latent_value = latent_value[:, -1, :].squeeze()
                elif config['tok_pos'] == 'first':
                    latent_value = latent_value[:, 0, :].squeeze()
                elif config['tok_pos'] == 'random':
                    latent_value = latent_value[:, random.randint(0, latent_value.size(1)), :].squeeze()
                else:
                    raise ValueError("only support last, first or random!")
                output_dict[layer][module] = latent_value.detach().to('cpu')
        else:
            # concatenate context vector for each module
            ensemble_dict = {module:[] for module in config['module']} # {module_name: []}
            for _, latent_dict in enumerate(all_latent_dicts):
                cur_dict = {module:[] for module in config['module']}  # {module_name: []}
                for layer, sub_dict in latent_dict.items():
                    for module in config['module']:
                        latent_value = sub_dict[module]  # (b, seq_len, d)  
                        if config['tok_pos'] == 'last':
                            latent_value = latent_value[:, -1, :].squeeze()
                        elif config['tok_pos'] == 'first':
                            latent_value = latent_value[:, 0, :].squeeze()
                        elif config['tok_pos'] == 'random':
                            latent_value = latent_value[:, random.randint(0, latent_value.size(1)), :].squeeze()
                        else:
                            raise ValueError("only support last, first or random!")
                        cur_dict[module].append(latent_value)

                for module, latent_list in cur_dict.items():
                    cur_latent = torch.stack(latent_list, dim=0) # (layer_num, d)
                    ensemble_dict[module].append(cur_latent)

            for module, latent_list in ensemble_dict.items():
                if config['post_fuse_method'] == 'mean':
                    context_vector = torch.stack(latent_list, dim=0).mean(dim=0)  # (layer_num, d)
                    ensemble_dict[module] = context_vector 
                elif config['post_fuse_method'] == 'pca':
                    latents = torch.stack(latent_list, dim=0)  # (ensemble_num, layer_num, d)
                    ensemble_num, layer_num, d = latents.size()
                    latents = latents.view(ensemble_num, -1)  # (ensemble_num*layer_num, d)
                    # apply pca
                    pca = utils.PCA(n_components=1).to(latents.device).fit(latents.float())
                    context_vector = (pca.components_.sum(dim=0, keepdim=True) + pca.mean_).mean(0)
                    ensemble_dict[module] = context_vector.view(layer_num, d)  # (layer_num, d)
                else:
                    raise ValueError("Unsupported ensemble method!")
            # reorganize ensemble_dict into layers
            layers = list(all_latent_dicts[0].keys())
            output_dict = {layer: {} for layer in layers} 
            for module, context_vector in ensemble_dict.items():
                for layer_idx, layer in enumerate(layers):
                    output_dict[layer][module] = context_vector[layer_idx, :].detach().to('cpu')  # (d)

        return output_dict

    def init_strength(self, config):
        # get linear_coef size
        if type(config['layer']) == str:
            if config['layer'] == 'all':
                layers = list(range(self.num_layers))
                layer_dim = len(layers)
            elif config['layer'] == 'late':
                layers = list(range((self.num_layers*2)//3, self.num_layers))
                layer_dim = len(layers)
            elif config['layer'] == 'early':
                layers = list(range(self.num_layers//3))
                layer_dim = len(layers)
            elif config['layer'] == 'mid':
                layers = list(range(self.num_layers//3, (self.num_layers*2)//3))
                layer_dim = len(layers)
        elif type(config['layer']) == list:
            layers = config['layer']
            layer_dim = len(layers)
        else:
            raise ValueError("layer must be all, late, early, mid or a list of layer index!")

        if config['inject_method'] == 'add':
            param_size = (layer_dim, len(config['module']), 1)  # (layer_num, module_num, 1)
        elif config['inject_method'] in ['linear', 'balance']:
            param_size = (layer_dim, len(config['module']), 2)  # (layer_num, module_num, 2)
        else:
            raise ValueError("only support add, linear or balance!")
        # set inject_layers
        self.inject_layers = layers
        # init linear_coef
        linear_coef = torch.zeros(param_size, device=self.device) 
        linear_coef += torch.tensor(config['init_value'], device=self.device)
        self.linear_coef = nn.Parameter(linear_coef)
        print(f"linear_coef shape: {self.linear_coef.shape}\n")
        if not self.linear_coef.is_leaf:
            raise ValueError("linear_coef is not a leaf tensor, which is required for optimization.")
        

    def init_noise_context_vector(self, context_vector_dict):
        # init learnable context_vector
        for layer, sub_dict in context_vector_dict.items():
            for module, latent in sub_dict.items():
                noise_vector = torch.randn_like(latent).detach().cpu()
                context_vector_dict[layer][module] = noise_vector
        return context_vector_dict
            
                    
    def _get_nested_attr(self, attr_path):
        """
        Accesses nested attributes of an object based on a dot-separated string path.

        :param obj: The object (e.g., a model).
        :param attr_path: A dot-separated string representing the path to the nested attribute.
                        For example, 'transformer.h' or 'model.layers'.
        :return: The attribute at the specified path.
        """
        try:
            return reduce(getattr, attr_path.split('.'), self.model)
        except AttributeError:
            raise AttributeError(f"Attribute path '{attr_path}' not found.")
        
    def _get_layer_num(self):
        raise NotImplementedError("Please implement get_layer_num function for each model!")
    
    def _get_arribute_path(self, layer_idx, target_module):
        raise NotImplementedError("Please implement get_arribute_path function for each model!")

            
class LlamaWrapper(ModelWrapper):
    def __init__(self, model, tokenizer, model_config, device):
        super().__init__(model, tokenizer, model_config, device)
        self.embed_matrix = self.model.model.embed_tokens.weight.data
        self.embed_dim = self.model_config.hidden_size
        self.last_norm = self.model.model.norm
        
    def _get_layer_num(self):
        return len(self.model.model.layers)
    
    def _get_arribute_path(self, layer_idx, target_module):
        if target_module == "attn":
            return f"model.layers.{layer_idx}.self_attn"
        elif target_module == "mlp":
            return f"model.layers.{layer_idx}.mlp"
        elif target_module == "hidden":
            return f"model.layers.{layer_idx}"
        else:
            raise ValueError("only support att or mlp!")


class QwenWrapper(ModelWrapper):
    def __init__(self, model, tokenizer, model_config, device):
        super().__init__(model, tokenizer, model_config, device)
        self.embed_matrix = self.model.model.embed_tokens.weight.data
        self.embed_dim = self.model_config.hidden_size
        self.last_norm = self.model.model.norm
        
    def _get_layer_num(self):
        return len(self.model.model.layers)
    
    def _get_arribute_path(self, layer_idx, target_module):
        if target_module == "attn":
            return f"model.layers.{layer_idx}.self_attn"
        elif target_module == "mlp":
            return f"model.layers.{layer_idx}.mlp"
        elif target_module == "hidden":
            return f"model.layers.{layer_idx}"
        else:
            raise ValueError("only support att or mlp!")


class GPTWrapper(ModelWrapper):
    def __init__(self, model, tokenizer, model_config, device):
        super().__init__(model, tokenizer, model_config, device)
        self.embed_matrix = self.model.transformer.wte.weight.data
        self.embed_dim = self.embed_matrix.size(-1)
        self.last_norm = self.model.transformer.ln_f
        
    def _get_layer_num(self):
        return len(self.model.transformer.h)
    
    def _get_arribute_path(self, layer_idx, target_module):
        if target_module == "attn":
            return f"transformer.h.{layer_idx}.attn"
        elif target_module == "mlp":
            return f"transformer.h.{layer_idx}.mlp"
        elif target_module == "hidden":
            return f"transformer.h.{layer_idx}"
        else:
            raise ValueError("only support att or mlp!")