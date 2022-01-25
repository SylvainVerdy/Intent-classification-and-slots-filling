#!/usr/bin/env/python3
"""
Text-only NLU recipe.
Authors
 * Sylvain Verdy, December 2021
"""


import os
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
from transformers import AutoModelForSequenceClassification, BertTokenizer, BertModel, BertConfig, AdamW, get_linear_schedule_with_warmup, BertConfig

from utils.data_loader_ATIS_SNIPS import CustomDataset


CUDA_LAUNCH_BLOCKING=1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Evaluation:
    def __init__(self, args) -> None:
        self.args = args
        self.train_dir = args.data_dir + 'train/NLU_prepared.pkl'
        self.valid_dir = args.data_dir + 'dev/NLU_prepared.pkl'
        self.test_dir = args.data_dir + 'test/NLU_prepared.pkl'

        
    def evaluate(self, args):
        model = self.load_model(args)
        val_loss = 0
        model.eval()
        val_dataset = CustomDataset(csv_file=self.valid_dir)
        dataset = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)
        accuracy_all, loss_tr = [], []
        global_step = 0
        valid_iterator = trange(args.epochs, desc='Epoch')
        with torch.no_grad():
            total, correct, accuracy = 0, 0 ,0
            epoch_iterator = tqdm(dataset, desc='Iteration')
            all_loss = 0
            val_loss = 0
            for step, batch in enumerate(epoch_iterator):
                inputs = {
                        'input_ids':batch[0].to(device),
                        'attention_mask' :batch[1].to(device),
                        'token_type_ids': batch[2].to(device), 
                        'intent_label_ids': batch[3].to(device), 
                        'slot_labels_ids': batch[4].to(device), 
                        }
                outputs = model(**inputs)
                loss = outputs[0]
                tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]
                val_loss += loss.item()

                total +=batch[3].size(0)
                _ , predicted = intent_logits.max(1)
                correct += torch.sum(predicted == inputs['intent_label_ids'])
            
            all_loss = val_loss / (step+1)
            acc = 100 * (correct / total)
            print(f'Val loss : {all_loss}')
            print(f'Val Accuracy : {acc} %')
        return all_loss

    def load_model(self, args):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        config = BertConfig.from_pretrained('bert-base-uncased', finetuning_task='atis')

        model = None
        if not os.path.exists(args.model):
            raise Exception("Model doesn't exists! Train first!")

        try:
            if args.model_name == 'bert':
                model = JointBERT.from_pretrained('bert-base-uncased', config=config, args=args).to(device)
            else: 
                model = Model(classes=22, classes_slots=122).to(device)
            model.load_state_dict(torch.load(args.model))
            model.eval()
            print("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
        return model
            

