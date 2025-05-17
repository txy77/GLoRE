import os
import sys

RE4R_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname((os.path.realpath(__file__))))
)
sys.path.insert(0, RE4R_ROOT_PATH)

RE4R_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)
sys.path.insert(0, RE4R_ROOT_PATH)

from src.dataset.base_dataset import BaseDataset
from src.utils.utils import read_jsonl

class GPQA_physics(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_data(self):
        instruction = "Answer the following question step by step and put the final answer in \\boxed{}:\nYour final choice should be one of the letters A, B, C, or D, DO NOT include any answer content.\n"
        path = "data/GPQA/Physics/Physics.jsonl"
        self.all_data = read_jsonl(path)
        self.all_problems, self.all_inputs, self.all_labels = [], [], []
        for d in self.all_data:
            problem = d['problem']
            answer = d["solution"]
            input_str = instruction + problem
            input_str = input_str.strip()
            self.all_problems.append(problem)
            self.all_inputs.append(input_str)
            self.all_labels.append(answer)