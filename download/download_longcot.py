import json
import random
from tqdm import tqdm
import os
import sys
from datasets import load_dataset

RE4R_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, RE4R_ROOT_PATH)

from src.utils.utils import write_jsonl

longcot_dataset_name = 'long_form_thought'
longcot_dataset = load_dataset('RUC-AIBOX/long_form_thought_data_5k')['train']

longcot_data = []

for d in tqdm(longcot_dataset, desc='longcot'):
    data = {}
    data['problem'] = d['question']
    data['solution'] = d['combined_text']
    data['domain'] = d['domain']
    longcot_data.append(data)

save_dir = "data/longcot"
os.makedirs(save_dir, exist_ok=True)
write_jsonl(longcot_data, os.path.join(save_dir, 'long_form_thought.jsonl'))