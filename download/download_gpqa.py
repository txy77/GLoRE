import json
import random
from tqdm import tqdm
import os
import sys
from collections import defaultdict
from datasets import load_dataset
from huggingface_hub import login

RE4R_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, RE4R_ROOT_PATH)

from src.utils.utils import write_jsonl

# ===== gpqa =====

random.seed(0)

domains = ['Physics', "Chemistry", "Biology"]

GPQA_dataset = load_dataset('Idavidrein/gpqa', 'gpqa_diamond')['train']

gpqa_data = defaultdict(list)
choices = ['A', 'B', 'C', 'D']

for idx, d in tqdm(enumerate(GPQA_dataset), desc='GPQA'):
    data = {}
    data_domain = d['High-level domain']
    problem = d['Pre-Revision Question']
    correct_answer = d['Pre-Revision Correct Answer']
    incorrect_answer1 = d['Pre-Revision Incorrect Answer 1']
    incorrect_answer2 = d['Pre-Revision Incorrect Answer 2']
    incorrect_answer3 = d['Pre-Revision Incorrect Answer 3']
    answers = [
        ('Correct_answer', correct_answer),
        ('Incorrect_answer1', incorrect_answer1),
        ('Incorrect_answer2', incorrect_answer2),
        ('Incorrect_answer3', incorrect_answer3),
    ]
    random.shuffle(answers)
    
    solution = None
    formatted_answers = []
    for i, (label, answer) in enumerate(answers):
        choice = choices[i]
        formatted_answers.append((choice, answer))
        if label == 'Correct_answer':
            solution = f"({choice}) {answer}" 

    formatted_choices = "\n".join([f"({choice}) {answer}" for choice, answer in formatted_answers])
    problem = f"{problem} Choices:\n{formatted_choices}\n"
    data['unique_id'] = idx
    data['problem'] = problem
    data['solution'] = solution
    gpqa_data[data_domain].append(data)

for domain in domains:
    save_dir = f"data/GPQA/{domain}"
    os.makedirs(save_dir, exist_ok=True)
    write_jsonl(gpqa_data[domain], os.path.join(save_dir, f'{domain}.jsonl'))