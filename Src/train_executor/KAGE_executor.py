"""
KAGE_executer.py: Training code for the paper:
Lin, W., Tseng, B. H., & Byrne, B. (2021). Knowledge-Aware Graph-Enhanced GPT-2 for Dialogue State Tracking. EMNLP 2021.
https://arxiv.org/abs/2104.04466v3
"""

__author__ = "Weizhe Lin"
__copyright__ = "Copyright 2021, Weizhe Lin"
__version__ = "1.0.0"
__email__ = "wl356@cam.ac.uk"
__status__ = "Published for Github"


import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import scipy
import datetime
import json
import random
import pickle
import shutil
from pathlib import Path
from functools import partial
from easydict import EasyDict
from utils.dirs import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import math
from train_executor.base_executor import BaseExecutor
from utils.log_system import logger

# Customize
from transformers import GPT2TokenizerFast, GPT2Config, get_linear_schedule_with_warmup

from torch.nn.modules.loss import CrossEntropyLoss
from utils.metrics_manager import MetricsManager
from utils.util_dst import *
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, GPT2LMHeadModel

class KAGEExecutor(BaseExecutor):
    def __init__(self, config, data_loader):
        BaseExecutor.__init__(self, config, data_loader)
        self.data_loader = data_loader
        # Init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_config.base_model)  # NOTE: by default, unk_token sets to <|endoftext|>

        # Add special tokens
        self.SPECIAL_TOKENS = data_loader.SPECIAL_TOKENS
        print(self.SPECIAL_TOKENS)
        self.tokenizer.add_special_tokens(self.SPECIAL_TOKENS)

        # Domain slot information from data loader wrapper
        self.value_id2text = data_loader.value_id2text
        self.value_text2id = data_loader.value_text2id
        self.ds_list = data_loader.ds_list
        self.ds_text2id = data_loader.ds_text2id

        # Create data loaders
        if self.config.data_loader.dummy_dataloader:
            load_num = 10
        else:
            load_num = -1
        self.train_data_loader, train_dataset = data_loader.set_dataloader(config, self.tokenizer, 'train',
                                                                           'teacher_force',
                                                                           self.value_id2text,
                                                                           self.value_text2id,
                                                                           self.ds_list,
                                                                           self.ds_text2id,
                                                                           data_size=load_num)
        self.valid_data_loader, valid_dataset = data_loader.set_dataloader(config, self.tokenizer, 'dev',
                                                                           'teacher_force',
                                                                           self.value_id2text,
                                                                           self.value_text2id,
                                                                           self.ds_list,
                                                                           self.ds_text2id,
                                                                           data_size=load_num)
        self.valid_gen_data_loader, valid_gen_dataset = data_loader.set_dataloader(config, self.tokenizer, 'dev',
                                                                           'generation',
                                                                           self.value_id2text,
                                                                           self.value_text2id,
                                                                           self.ds_list,
                                                                           self.ds_text2id,
                                                                           data_size=load_num)
        # self.train_loss_ratio_dict = train_dataset.loss_ratio_dict

        logger.print("Finished initialization, loading model....")
        # input('stop')
        self.value_id2text = data_loader.value_id2text
        self.value_text2id = data_loader.value_text2id
        self.ontology_value_list = data_loader.ontology_value_list
        self.ontology_value_text2id = data_loader.ontology_value_text2id
        self.ontology_value_id2text = data_loader.ontology_value_id2text

        # self.batch_classifier_id2ds = {}
        # self.batch_classifier_ds2id = {}
        # list_num_classifiers = []
        # for index, str_ds_pair in enumerate(data_loader._ds_list):
        #     num_classifier = len(self.value_text2id[str_ds_pair])
        #     list_num_classifiers.append(num_classifier)
        #     self.batch_classifier_id2ds[index] = str_ds_pair
        #     self.batch_classifier_ds2id[str_ds_pair] = index

        # Initialize models
        self.model_config = AutoConfig.from_pretrained(config.model_config.base_model)

        # self.model_config.batch_classifier = list_num_classifiers
        # self.list_num_classifiers = list_num_classifiers
        # self.model_config.pad_token_id = self.tokenizer.convert_tokens_to_ids('<PAD>')
        # self.model_config.vocab_size = len(self.tokenizer)
        from models.KAGE_GPT2.KAGE_GPT2 import KAGEModel

        self.model = KAGEModel.from_pretrained(config.model_config.base_model,
                                                    config=self.model_config,
                                                    sys_config=self.config)  # GPT2LMHeadModel
        # print(self.model_config)
        # self.model_config.vocab_size = len(self.tokenizer)
        # self.model = DSTQAPlusModel(self.model_config)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Save tokenizer for later use
        print('Tokenizer length:', len(self.tokenizer))
        tokenizer_path = os.path.join(self.config.saved_model_path, 'tokenizer')
        self.tokenizer.save_pretrained(tokenizer_path)
        logger.print('saved tokenizer to', tokenizer_path)

        self.model.to(self.config.device)

        # self.num_labels = self.model.num_labels
        if self.config.freeze_transformer:
            for name, param in self.model.named_parameters():
                if 'transformer' in name:
                    param.requires_grad = False
                print(name, param.requires_grad)
        # input()
        # print(self.model)

        graph_parameters = filter(lambda p: p[1].requires_grad and 'graph' in p[0], self.model.named_parameters())
        transformer_parameters = filter(lambda p: p[1].requires_grad and 'graph' not in p[0], self.model.named_parameters())

        # Create optimizer
        self.optimizer = optim.AdamW(
            [
                dict(params=[param for name, param in graph_parameters],
                lr=self.config.train.graph_lr,
                eps=self.config.train.adam_epsilon),
                dict(params=[param for name, param in transformer_parameters],
                lr=self.config.train.lr,
                eps=self.config.train.adam_epsilon),
            ]
        )

        # Calculate total iterations to execute, apply linear schedule
        t_total = len(
            self.train_data_loader) // self.config.train.additional.gradient_accumulation_steps * (
                              self.config.train.epochs)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.train.additional.warmup_steps,
            num_training_steps=t_total
        )

        # Load checkpoints
        self.load_checkpoint_model(load_epoch=self.config.train.load_epoch,
                                   load_best_model=self.config.train.load_best_model,
                                   load_model_path=self.config.train.load_model_path)

        logger.print("finished initialization...starting training.")

        # Create Metrics Manager
        self.train_metrics = MetricsManager(self.config, data_loader.ds_list, self.value_text2id, self.value_id2text,
                                            self.tokenizer)
        self.valid_metrics = MetricsManager(self.config, data_loader.ds_list, self.value_text2id, self.value_id2text,
                                            self.tokenizer)

        if self.config.model_config.graph_mode != 'none':
            # Add KB data into model
            value_id2tokenized_text = {}
            ontology_value_id2tokenized_text = {}
            for str_ds_pair in self.ds_list:
                value_id2tokenized_text[str_ds_pair] = {}
                value_dict = self.value_id2text[str_ds_pair]
                print(str_ds_pair, value_dict)
                for i in range(len(value_dict)):
                    text = value_dict[i]
                    assert text != ''
                    value_id2tokenized_text[str_ds_pair][i] = self.tokenizer(text)['input_ids']
                    # print(text, self.tokenizer(text)['input_ids'])
            self.value_id2tokenized_text = value_id2tokenized_text

            for value in self.ontology_value_list:
                assert value != ''
                ontology_value_id2tokenized_text[self.ontology_value_text2id[value]] = self.tokenizer(value)['input_ids']
            self.model.add_KB(
                value_id2tokenized_text,
                self.value_id2text,
                self.ds_list,
                self.ontology_value_list,
                self.ontology_value_text2id,
                self.ontology_value_id2text,
                ontology_value_id2tokenized_text,
            )

    def train(self):
        #############################################
        #
        #                load setup
        #
        #############################################

        batch_size = self.config.train.batch_size
        save_interval = self.config.train.save_interval
        device = self.config.device
        gradient_accumulation_steps = self.config.train.additional.gradient_accumulation_steps
        start_time = datetime.datetime.now()
        logdir = os.path.join(self.config.tensorboard_path)  # datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        logger.print(logdir)

        if self.config.fp16:
            # Creates once at the beginning of training
            scaler = torch.cuda.amp.GradScaler()

        ADDED_GRAPH = True
        if self.config.reset:
            ADDED_GRAPH = False

        if self.loaded_epoch == 0:
            ADDED_GRAPH = False

        writer = SummaryWriter(logdir)

        bos_id, eos_id, pad_id, sep_id = self.tokenizer.convert_tokens_to_ids(
            ['<BOS>', '<EOS>', '<PAD>', '<SEP>'])

        for epoch in range(int(self.config.train.epochs)):
            current_epoch = epoch + self.loaded_epoch + 1
            if current_epoch > int(self.config.train.epochs):
                logger.print('Training completed.')
                break
            #############################################
            #
            #                Train
            #
            #############################################
            # zero the parameter gradients
            self.model.train()
            # self.model.set_module_to_train()
            total_loss_list = []
            cls_loss_list = []
            gen_loss_list = []
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                if i_batch == 0:
                    self.optimizer.zero_grad()
                # print(sample_batched)
                # self.optimizer.zero_grad()
                # At each batch
                input_ids = sample_batched['input_ids'].to(self.config.device)
                attention_mask = sample_batched['attention_mask'].to(self.config.device)
                token_type_ids = sample_batched['token_type_ids']
                if token_type_ids:
                    token_type_ids = token_type_ids.to(self.config.device)
                pre_input_ids = sample_batched['pre_input_ids'].to(self.config.device)
                pre_attention_mask = sample_batched['pre_attention_mask'].to(self.config.device)
                pre_ignore_len = sample_batched['pre_ignore_len']
                pre_ds_indice = sample_batched['pre_ds_indice']
                ds_indice = sample_batched['ds_indice']
                # ignore_len = sample_batched['ignore_len']

                cls_labels = sample_batched['cls_labels'] # B x 30
                gen_labels = sample_batched['label_ids'].to(self.config.device)
                cls_labels = torch.LongTensor(cls_labels).to(self.config.device)#.int()
                if self.config.fp16:
                    # Casts operations to mixed precision
                    with torch.cuda.amp.autocast():
                        pre_forward_results = self.model.pre_forward(
                                                pre_input_ids=pre_input_ids,
                                                pre_attention_mask=pre_attention_mask,
                                                pre_ignore_len=None,
                                                pre_ds_indice=pre_ds_indice,
                                            )
                        ds_embeddings = pre_forward_results['ds_embeddings']
                        graph_forward_results = self.model.graph_forward(
                            ds_embeddings=ds_embeddings
                        )
                        ds_embeddings = graph_forward_results['ds_embeddings']

                        forward_results = self.model(input_ids=input_ids,
                                                     attention_mask=attention_mask,
                                                     token_type_ids=token_type_ids,
                                                     labels=gen_labels,
                                                     # list_cls_index=list_cls_index,
                                                     return_dict=True,
                                                     ds_indice=ds_indice,
                                                     ds_embeddings=ds_embeddings,
                                                     )
                else:
                    pre_forward_results = self.model.pre_forward(
                        pre_input_ids=pre_input_ids,
                        pre_attention_mask=pre_attention_mask,
                        pre_ignore_len=None,
                        pre_ds_indice=pre_ds_indice,
                    )
                    ds_embeddings = pre_forward_results['ds_embeddings']
                    graph_forward_results = self.model.graph_forward(
                        ds_embeddings=ds_embeddings,
                        cls_labels=cls_labels,
                    )
                    ds_embeddings = graph_forward_results['ds_embeddings']
                    cls_loss = graph_forward_results['loss']

                    forward_results = self.model(input_ids=input_ids,
                                                 attention_mask=attention_mask,
                                                 token_type_ids=token_type_ids,
                                                 labels=gen_labels,
                                                 # list_cls_index=list_cls_index,
                                                 return_dict=True,
                                                 ds_indice=ds_indice,
                                                 ds_embeddings=ds_embeddings,
                                                 )

                logits = forward_results['logits']

                # cls_logits = forward_results['cls_logits']
                # cls_loss = self.compute_cls_loss(cls_logits, cls_labels, list_cls_index)
                # gen_loss = self.compute_gen_loss(logits, gen_labels)
                gen_loss = forward_results['loss']
                total_loss = gen_loss + cls_loss * 0.001
                total_loss = total_loss / gradient_accumulation_steps

                logger.print("epoch {} - batch {} - current loss {} ({} + {}) - {}/{}".format(current_epoch,
                                                                                          i_batch,
                                                                                          total_loss,#.detach().cpu(),
                                                                                        gen_loss,#.detach().cpu(),
                                                                                        cls_loss,#.detach().cpu(),
                                                                                          i_batch,
                                                                                          len(self.train_data_loader)))


                if self.config.fp16:
                    # Scales the loss, and calls backward()
                    # to create scaled gradients
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward(retain_graph=False)

                # self.optimizer.step()
                if i_batch % gradient_accumulation_steps == 0 and i_batch != 0:

                    if self.config.fp16:
                        # Unscales the gradients of optimizer's assigned params in-place
                        scaler.unscale_(self.optimizer)
                        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       self.config.train.additional.gradient_clipping)
                        # Unscales gradients and calls
                        # or skips optimizer.step()
                        scaler.step(self.optimizer)
                        # Updates the scale for next iteration
                        scaler.update()
                    else:
                        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                              self.config.train.additional.gradient_clipping)
                        print('optimizer step!')
                        self.optimizer.step()

                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
                        print('scheduler step! LR:', self.scheduler.get_last_lr())
                        print([group['lr'] for group in self.optimizer.param_groups])

                    if self.config.model_config.graph_mode != 'none':
                        # refresh value node embeddings after back propagation
                        self.model.refresh_embeddings()


                gen_predictions = torch.argmax(logits.detach().cpu(), dim=-1)

                cls_logits = graph_forward_results['logits']
                cls_predictions = []
                for cls_logit in cls_logits:
                    cls_predictions.append(torch.argmax(cls_logit, dim=-1).detach().cpu())

                if i_batch % 50 == 0:
                    self.train_metrics.add_turn_results_cls(None, cls_predictions, None, sample_batched, verbose=True)
                    self.train_metrics.add_turn_results_gen(gen_predictions, sample_batched, verbose=True)
                else:
                    self.train_metrics.add_turn_results_cls(None, cls_predictions, None, sample_batched)
                    self.train_metrics.add_turn_results_gen(gen_predictions, sample_batched)

                # print(metric_str)
                if i_batch % 50 == 0:
                    metrics = self.train_metrics.get_metrics()
                    # print(metrics)
                    metric_str = ''
                    for str_ds_pair in metrics.keys():
                        metric_str += str_ds_pair
                        metric_str += ':'
                        for metric_name in metrics[str_ds_pair].keys():
                            metric_str += str(metrics[str_ds_pair][metric_name])
                            metric_str += ' '
                    print(metric_str)

                if i_batch % save_interval == 0 and i_batch != 0:
                    if self.config.valid.step_size > 1:
                        if current_epoch % self.config.valid.step_size == 0:
                            # Save a checkpoint when doing validation
                            self.save_checkpoint(current_epoch, i_batch, record_best_model=False)
                    else:
                        # Save a checkpoint
                        self.save_checkpoint(current_epoch, i_batch, record_best_model=False)

                total_loss_list.append(total_loss.detach().cpu().numpy())
                gen_loss_list.append(gen_loss.detach().cpu().numpy())
                if self.config.model_config.cls_loss:
                    cls_loss_list.append(cls_loss.detach().cpu().numpy())

            self.optimizer.step()
            self.optimizer.zero_grad()

            # Add to tensorboard
            writer.add_scalar('train/loss', np.mean(np.array(total_loss_list)), current_epoch)
            writer.add_scalar('train/gen_loss', np.mean(np.array(gen_loss_list)), current_epoch)
            if self.config.model_config.cls_loss:
                writer.add_scalar('train/cls_loss', np.mean(np.array(cls_loss_list)), current_epoch)

            # metrics = self.train_metrics.get_metrics()
            # for str_ds_pair in metrics.keys():
            #     for metric_name in metrics[str_ds_pair].keys():
            #         writer.add_scalar('train_metrics_{}/{}'.format(metric_name, str_ds_pair),
            #                           metrics[str_ds_pair][metric_name], current_epoch)
            # self.train_metrics.init_session()
            writer.flush()
            print('Results added!')

            if self.config.valid.step_size > 1:
                if current_epoch % self.config.valid.step_size == 0:
                    self.save_checkpoint('{}'.format(current_epoch), record_best_model=False)

            # Skip validation with valid.set_size
            if self.config.valid.step_size > 1:
                if current_epoch % self.config.valid.step_size != 0:
                    print('skip validation...')
                    continue

            #############################################
            #
            #         Generation Validation
            #
            #############################################


            with torch.no_grad():
                self.model.eval()
                # self.model.set_module_to_eval()
                self.valid_metrics.init_session()

                if self.config.model_config.graph_mode != 'none':
                    # refresh value node embeddings
                    self.model.refresh_embeddings()

                for i_batch, sample_batched in enumerate(tqdm(self.valid_gen_data_loader)):
                    if i_batch >= self.config.valid.num_valid_generation:
                        break
                    input_ids = sample_batched['input_ids'].to(self.config.device)
                    attention_mask = sample_batched['attention_mask'].to(self.config.device)
                    token_type_ids = sample_batched['token_type_ids']
                    if token_type_ids:
                        token_type_ids = token_type_ids.to(self.config.device)
                    pre_input_ids = sample_batched['pre_input_ids'].to(self.config.device)
                    pre_attention_mask = sample_batched['pre_attention_mask'].to(self.config.device)
                    pre_ds_indice = sample_batched['pre_ds_indice']
                    ds_ids = sample_batched['ds_ids']
                    batch_size, ctx_len = input_ids.size()
                    assert batch_size == 1

                    output = self.model.generate(input_ids,
                                                 max_length=(ctx_len + 300),
                                                 do_sample=False,
                                                 temperature=1.0, use_cache=True,
                                                 num_beams=1,
                                                 pre_input_ids=pre_input_ids,
                                                 pre_attention_mask=pre_attention_mask,
                                                 pre_ds_indice=pre_ds_indice,
                                                 bos_id=bos_id,
                                                 eos_token_id=eos_id,
                                                 pad_token_id=pad_id,
                                                 sep_token_id=sep_id,
                                                 ds_ids=ds_ids,
                                                 early_stopping=True)
                    # output = self.model.run_generation(input_ids,
                    #                                    attention_mask=attention_mask,
                    #                                    token_type_ids=token_type_ids,
                    #                                    pre_input_ids=pre_input_ids,
                    #                                    pre_attention_mask=pre_attention_mask,
                    #                                    pre_ds_indice=pre_ds_indice,
                    #                                    max_length=(ctx_len + 300),
                    #                                    ds_ids=ds_ids,
                    #                                     bos_token_id=bos_id,
                    #                                     eos_token_id=eos_id,
                    #                                     pad_token_id=pad_id,
                    #                                    sep_token_id=sep_id,
                    #                                    )
                    #	output = generation(model, batch) # same speed as .generate() api above
                    ootput_ids = output[0].cpu().numpy().tolist()
                    self.valid_metrics.add_turn_results_gen_test(ootput_ids, sample_batched)

                # After validation, print results
                metrics = self.valid_metrics.get_metrics()
                # print(metrics)
                metric_str = ''
                for str_ds_pair in metrics.keys():
                    metric_str += str_ds_pair
                    metric_str += ':'
                    for metric_name in metrics[str_ds_pair].keys():
                        metric_str += str(metrics[str_ds_pair][metric_name])
                        metric_str += ' '
                # # print(metric_str)
                print(metric_str)
                # Add to tensorboard
                for str_ds_pair in metrics.keys():
                    for metric_name in metrics[str_ds_pair].keys():
                        writer.add_scalar('valid_metrics_{}/{}'.format(metric_name, str_ds_pair),
                                          metrics[str_ds_pair][metric_name], current_epoch)
                self.valid_metrics.init_session()
                writer.flush()
                print('Results added!')

            #############################################
            #
            #                Run Validation
            #
            #############################################
            total_loss_list = []
            gen_loss_list = []
            cls_loss_list = []
            with torch.no_grad():
                self.model.eval()
                # self.model.set_module_to_eval()
                self.valid_metrics.init_session()
                if self.config.model_config.graph_mode != 'none':
                    self.model.refresh_embeddings()
                for i_batch, sample_batched in enumerate(self.valid_data_loader):
                    # At each batch
                    input_ids = sample_batched['input_ids'].to(self.config.device)
                    attention_mask = sample_batched['attention_mask'].to(self.config.device)
                    token_type_ids = sample_batched['token_type_ids']
                    if token_type_ids:
                        token_type_ids = token_type_ids.to(self.config.device)
                    pre_input_ids = sample_batched['pre_input_ids'].to(self.config.device)
                    pre_attention_mask = sample_batched['pre_attention_mask'].to(self.config.device)
                    pre_ignore_len = sample_batched['pre_ignore_len']
                    pre_ds_indice = sample_batched['pre_ds_indice']
                    ds_indice = sample_batched['ds_indice']
                    # ignore_len = sample_batched['ignore_len']

                    cls_labels = sample_batched['cls_labels']  # B x 30
                    gen_labels = sample_batched['label_ids'].to(self.config.device)
                    cls_labels = torch.LongTensor(cls_labels).to(self.config.device)  # .int()

                    pre_forward_results = self.model.pre_forward(
                        pre_input_ids=pre_input_ids,
                        pre_attention_mask=pre_attention_mask,
                        pre_ignore_len=None,
                        pre_ds_indice=pre_ds_indice,
                    )
                    ds_embeddings = pre_forward_results['ds_embeddings']
                    graph_forward_results = self.model.graph_forward(
                        ds_embeddings=ds_embeddings,
                        cls_labels=cls_labels,
                    )
                    ds_embeddings = graph_forward_results['ds_embeddings']
                    cls_loss = graph_forward_results['loss']

                    forward_results = self.model(input_ids=input_ids,
                                                 attention_mask=attention_mask,
                                                 token_type_ids=token_type_ids,
                                                 labels=gen_labels,
                                                 # list_cls_index=list_cls_index,
                                                 return_dict=True,
                                                 ds_indice=ds_indice,
                                                 ds_embeddings=ds_embeddings,
                                                 )
                    logits = forward_results['logits']
                    gen_loss = forward_results['loss']

                    total_loss = gen_loss + cls_loss * 0.001
                    # total_loss = total_loss / gradient_accumulation_steps
                    logger.print("epoch {} - batch {} - current loss {} ({} + {}) - {}/{}".format(current_epoch,
                                                                                                  i_batch,
                                                                                                  total_loss,
                                                                                                  gen_loss,
                                                                                                  cls_loss,
                                                                                                  i_batch,
                                                                                              len(
                                                                                                  self.valid_data_loader)))

                    cls_logits = graph_forward_results['logits']
                    cls_predictions = []
                    for cls_logit in cls_logits:
                        cls_predictions.append(torch.argmax(cls_logit, dim=-1).detach().cpu())

                    if i_batch % 50 == 0:
                        self.valid_metrics.add_turn_results_cls(None, cls_predictions, None, sample_batched,
                                                                verbose=True)
                    else:
                        self.valid_metrics.add_turn_results_cls(None, cls_predictions, None, sample_batched)

                    # gen_predictions = torch.argmax(logits.detach().cpu(), dim=-1)
                    # cls_predictions = []
                    # for logits in cls_logits:
                    #     cls_predictions.append(torch.argmax(logits, dim=-1).detach().cpu())
                    #
                    # if i_batch % 50 == 0:
                    #     self.valid_metrics.add_turn_results_gen(gen_predictions, sample_batched, verbose=True)
                    # else:
                    #     self.valid_metrics.add_turn_results_gen(gen_predictions, sample_batched)

                    total_loss_list.append(total_loss.detach().cpu().numpy())
                    gen_loss_list.append(gen_loss.detach().cpu().numpy())
                    if self.config.model_config.cls_loss:
                        cls_loss_list.append(cls_loss.detach().cpu().numpy())
                # After validation, print results
                metrics = self.valid_metrics.get_metrics()
                print(metrics)
                metric_str = ''
                for str_ds_pair in metrics.keys():
                    metric_str += str_ds_pair
                    metric_str += ':'
                    for metric_name in metrics[str_ds_pair].keys():
                        metric_str += str(metrics[str_ds_pair][metric_name])
                        metric_str += ' '
                # # print(metric_str)
                print(metric_str)

                # Add to tensorboard
                writer.add_scalar('valid/loss', np.mean(np.array(total_loss_list)), current_epoch)
                writer.add_scalar('valid/gen_loss', np.mean(np.array(gen_loss_list)), current_epoch)
                if self.config.model_config.cls_loss:
                    writer.add_scalar('valid/cls_loss', np.mean(np.array(cls_loss_list)), current_epoch)
                for str_ds_pair in metrics.keys():
                    for metric_name in metrics[str_ds_pair].keys():
                        writer.add_scalar('valid_metrics_CLS_{}/{}'.format(metric_name, str_ds_pair),
                                          metrics[str_ds_pair][metric_name], current_epoch)
                self.valid_metrics.init_session()
                writer.flush()
                print('Results added!')
