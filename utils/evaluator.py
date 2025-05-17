import json
import re
import sys
import os

RE4R_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, RE4R_ROOT_PATH)

from Qwen_Math.evaluation.parser import parse_question, parse_ground_truth, run_execute
from Qwen_Math.evaluation.python_executor import PythonExecutor
from Qwen_Math.evaluation.grader import math_equal_process

class MATHEvaluator:

    def __init__(self, data_name='math_oai'):
        self.data_name = data_name
        self.executor = None

    def get_pred(self, generated_text):
        result = run_execute(self.executor, generated_text, '', self.data_name)
        return result[0]
    
    def score(self, prediction, sample):
        if isinstance(sample, dict):
            pred = self.get_pred(prediction)
            if 'gt_ans' not in sample:
                gt_cot, gt_ans = parse_ground_truth(sample, self.data_name)
                sample['gt_ans'] = gt_ans
            return math_equal_process((0, pred, sample['gt_ans']))
        elif isinstance(sample, str):
            return math_equal_process((0, self.get_pred(prediction), self.get_pred(sample)))