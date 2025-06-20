import json
import os
import sys
import argparse
from tqdm import tqdm
import torch
import pdb
import random
import numpy as np
from accelerate import Accelerator
from sentence_transformers import SentenceTransformer

RE4R_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname((os.path.realpath(__file__))))
)
sys.path.insert(0, RE4R_ROOT_PATH)

RE4R_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)
sys.path.insert(0, RE4R_ROOT_PATH)

from dataset.base_dataset import BaseDataset
from dataset.mathoai import MATHOAI
from dataset.aime import AIME
from dataset.amc import AMC
from dataset.GPQA_physics import GPQA_physics
from dataset.GPQA_chemistry import GPQA_chemistry
from dataset.GPQA_biology import GPQA_biology

target_datasets = {
    'mathoai': MATHOAI,
    'aime': AIME,
    'amc': AMC,
    'GPQA-physics': GPQA_physics,
    'GPQA-chemistry': GPQA_chemistry,
    'GPQA-biology': GPQA_biology,
}

dataset_dict = {}
dataset_dict.update(target_datasets)

def get_dataset(dataset_name, *args, **kwargs) -> BaseDataset:
    return dataset_dict[dataset_name](dataset_name=dataset_name, *args, **kwargs)