#!/usr/bin/env/python3
"""
Text-only NLU recipe.
Authors
 * Sylvain Verdy, December 2021
"""


from ast import Mod
from asyncio.events import BaseDefaultEventLoopPolicy
from calendar import day_abbr
import os
from re import T
from socketserver import DatagramRequestHandler
import sys
import time
from traceback import print_tb

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

from utils.metrics import Metrics
from utils.prepare_dataset import load_and_cache_examples
from utils.data_loader_ATIS_SNIPS import CustomDataset

#from utils.prepare import DataTransform
from utils.EarlyStopping import EarlyStopping
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb

CUDA_LAUNCH_BLOCKING=1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(self, args, model) -> None:
        self.args = args
        self.model = model
        self.train_dir = args.data_dir + 'train/NLU_prepared.pkl'
        self.valid_dir = args.data_dir + 'dev/NLU_prepared.pkl'
        self.test_dir = args.data_dir + 'test/NLU_prepared.pkl'

        
    
    def train(self, args, model):

        if args.optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=0.0000009)
        tr_loss = 0
        model.train()
        train_dataset = CustomDataset(csv_file=self.train_dir)
        dataset = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0)
        accuracy_all, loss_tr = [], []
        global_step = 0
        train_iterator = trange(args.epochs, desc='Epoch')
        better_loss = 0
        for epoch  in train_iterator:
            total, correct, accuracy = 0, 0 ,0
            epoch_iterator = tqdm(dataset, desc='Iteration')
            all_loss = 0
            tr_loss = 0
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
                loss.backward()
                tr_loss += loss.item()
                
                if (step + 1) % 1 == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    model.zero_grad()
                    global_step +=1
                total +=batch[3].size(0)
                _ , predicted = intent_logits.max(1)
                correct += torch.sum(predicted == inputs['intent_label_ids'])
            
            all_loss = tr_loss / (step+1)
            acc = 100 * (correct / total)
            print(f'Train loss : {all_loss}, epoch : {epoch}')
            print(f'Train Accuracy : {acc} %, epoch : {epoch}')
            if epoch == 0:
                self.save_model(args, model)
                better_loss = self.evaluate(args, model)

            if epoch % 4 == 0 :
                print('evaluation starting ...')
                loss = self.evaluate(args, model)
                if loss < better_loss:
                    self.save_model(args, model)
                    better_loss = loss

    def evaluate(self, args, model):
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
            
    def save_model(self, args, model):
        torch.save(model.state_dict(), args.model)
        print("Saving model checkpoint to %s", args.model)
