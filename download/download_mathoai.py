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

mathoai_dataset_name = 'mathoai'
mathoai_dataset = load_dataset('HuggingFaceH4/MATH-500')['test']

mathoai_data = []

for d in tqdm(mathoai_dataset, desc='mathoai'):
    data = {}
    data['problem'] = d['problem']
    data['solution'] = d['solution']
    data['answer'] = d['answer']
    data['subject'] = d['subject']
    data['level'] = d['level']
    data['unique_id'] = d['unique_id']
    mathoai_data.append(data)

save_dir = "data/math"
os.makedirs(save_dir, exist_ok=True)
write_jsonl(mathoai_data, os.path.join(save_dir, 'mathoai.jsonl'))