#!/usr/bin/env/python3
"""
Text-only NLU recipe.
Authors
 * Sylvain Verdy, December 2021
"""


import argparse
import os
from pickletools import optimize
import sys
import time

import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import random
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.modeling_jointbert import JointBERT
from models.TFToIntent import Model

#from torchsampler import ImbalancedDatasetSampler
from transformers import AutoModelForSequenceClassification, BertTokenizer, BertModel, BertConfig, AdamW, get_linear_schedule_with_warmup, BertConfig

from utils.data_loader_ATIS_SNIPS import CustomDataset

from train import Trainer
from evaluation import Evaluation

# Get args
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=False, help='run NLU training')
parser.add_argument('--eval', action='store_true', default=False, help='run NLU evaluation')
parser.add_argument('--data_dir', type=str, help='load data dir')
parser.add_argument('--model', type=str, help='path to model.')
parser.add_argument('--model_name', type=str, help='Class model JointBert Use Transformers or Bert(False/True)')
parser.add_argument('--batch_size', type=int, help='batch_size')
parser.add_argument('--optimizer_name', type=str, default='Adam', help='optimizer default adam')
parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate')
parser.add_argument('--loss', type=str, default='CrossEntropyLoss', help='Loss function')
parser.add_argument('--task', type=str, default='slurp', help='get dataset name')
parser.add_argument('--use_crf', type=bool, default=False, help='Use CRF method')
parser.add_argument('--dropout_rate', type=float, default=0.1, help='Use dropout')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--ignore_index', type=int, default=0, help='ignore pad or unk labels')


args = parser.parse_args()
eval = args.eval
train = args.train
model_name = args.model_name
task = args.task


torch.manual_seed(1234); np.random.seed(1234)

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained('bert-base-uncased', finetuning_task='atis')

if model_name == 'bert':
    model = JointBERT.from_pretrained('bert-base-uncased', config=config, args=args).to(device)
    model.train()
else: 
    model = Model(classes=22, classes_slots=122).to(device)
    model.train()

trainer = Trainer(args, model)
evaluation = Evaluation(args)
if train:
    print('start training... \n')
    trainer.train(args, model)
    print('Train done!')

if eval:
    print('start evaluation... \n')
    evaluation.evaluate(args)
    print('eval done!')
