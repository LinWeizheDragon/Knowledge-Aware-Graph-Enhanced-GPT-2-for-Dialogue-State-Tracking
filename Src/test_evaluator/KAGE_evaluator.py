"""
KAGE_evaluator.py: Test code for the paper:
Lin, W., Tseng, B. H., & Byrne, B. (2021). Knowledge-Aware Graph-Enhanced GPT-2 for Dialogue State Tracking. EMNLP 2021.
https://arxiv.org/abs/2104.04466v3
"""

__author__ = "Weizhe Lin"
__copyright__ = "Copyright 2021, Weizhe Lin"
__version__ = "1.0.0"
__email__ = "wl356@cam.ac.uk"
__status__ = "Published for Github"

import scipy
import os
from pathlib import Path
import numpy as np
import os
import sys
import scipy
import datetime
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.log_system import logger
import sys
import json
from test_evaluator.base_evaluator import BaseEvaluator

# Customize
from transformers import GPT2TokenizerFast, GPT2Config, AutoTokenizer, AutoConfig, GPT2LMHeadModel, GPT2Tokenizer
from torch.nn.modules.loss import CrossEntropyLoss
from utils.metrics_manager import MetricsManager
from tqdm import tqdm
import time
from torch.multiprocessing import Manager, spawn, Process
import pickle
import shutil
torch.multiprocessing.set_sharing_strategy('file_system')
from utils.dirs import create_dirs, delete_dir

class KAGEEvaluator(BaseEvaluator):
    def __init__(self, config, data_loader):
        BaseEvaluator.__init__(self, config, data_loader)
        self.config = config

        # Set seed
        if config.test.seed:
            set_seed(config.test.seed)
            logger.print("SEED is set to:", config.test.seed)

        # Load tokenizer from training
        tokenizer_path = os.path.join(self.config.saved_model_path, 'tokenizer')
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        logger.print('loaded tokenizer from', tokenizer_path)

        # self.SPECIAL_TOKENS = data_loader.SPECIAL_TOKENS
        # print(self.SPECIAL_TOKENS)
        # self.tokenizer.add_special_tokens(self.SPECIAL_TOKENS)

        # Domain slot information from data loader
        self.value_id2text = data_loader.value_id2text
        self.value_text2id = data_loader.value_text2id
        self.ds_list = data_loader.ds_list
        self.ds_text2id = data_loader.ds_text2id
        self.ontology_value_list = data_loader.ontology_value_list
        self.ontology_value_text2id = data_loader.ontology_value_text2id
        self.ontology_value_id2text = data_loader.ontology_value_id2text

        # Create test data loader
        if self.config.data_loader.dummy_dataloader:
            load_num = 100
        else:
            load_num = -1

        if self.config.test.additional.generate_data:
            config.valid.batch_size = config.test.batch_size
            config.train.batch_size = config.test.batch_size
            self.train_data_loader, train_dataset = data_loader.set_dataloader(config, self.tokenizer, 'train',
                                                                               'generation',
                                                                               self.value_id2text,
                                                                               self.value_text2id,
                                                                               self.ds_list,
                                                                               self.ds_text2id,
                                                                               data_size=load_num)
            self.valid_data_loader, valid_dataset = data_loader.set_dataloader(config, self.tokenizer, 'dev',
                                                                               'generation',
                                                                               self.value_id2text,
                                                                               self.value_text2id,
                                                                               self.ds_list,
                                                                               self.ds_text2id,
                                                                               data_size=load_num)
        self.test_data_loader, test_dataset = data_loader.set_dataloader(config, self.tokenizer, 'test',
                                                                                   'generation',
                                                                                   self.value_id2text,
                                                                                   self.value_text2id,
                                                                                   self.ds_list,
                                                                                   self.ds_text2id,
                                                                                   data_size=load_num)

        logger.print('Test samples', len(self.test_data_loader))
        logger.print("Finished initialization, loading model....")

        # Initialize models
        from models.KAGE_GPT2.KAGE_GPT2 import KAGEModel

        self.model_config = AutoConfig.from_pretrained('gpt2')
        self.model = KAGEModel.from_pretrained('gpt2',
                                                    config=self.model_config,
                                                    sys_config=self.config)  # GPT2LMHeadModel
        self.model.resize_token_embeddings(len(self.tokenizer))

        # self.model.init_batch_classifiers(len(list_num_classifiers), list_num_classifiers, self.config.device)

        self.model.to(self.config.device)
        print(self.model)

        # Load checkpoints
        self.load_checkpoint_model(load_epoch=self.config.test.load_epoch,
                                   load_best_model=self.config.test.load_best_model,
                                   load_model_path=self.config.test.load_model_path)

        logger.print("finished initialization...starting evaluation.")

        self.test_metrics = MetricsManager(self.config, data_loader.ds_list, self.value_text2id, self.value_id2text,
                                            self.tokenizer)

        if self.config.test_output_attention:
            attention_info_data = {
                'ds_list': self.ds_list,
                'ontology_value_list': self.ontology_value_list,
                'ontology_value_text2id': self.ontology_value_text2id,
                'ontology_value_id2text': self.ontology_value_id2text,
            }
            att_info_data_path = os.path.join(self.config.results_path, 'att_info.json')
            with open(att_info_data_path, 'w') as f:
                json.dump(attention_info_data, f)
                print("attention info data saved at:", att_info_data_path)


        if self.config.model_config.graph_mode != 'none':
            # Add KB data into model
            value_id2tokenized_text = {}
            ontology_value_id2tokenized_text = {}
            for str_ds_pair in self.ds_list:
                value_id2tokenized_text[str_ds_pair] = {}
                value_dict = self.value_id2text[str_ds_pair]
                # print(str_ds_pair, value_dict)
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


    def evaluate_multi(self, mode, data_loader):
        self.test_metrics.init_session()
        manager = Manager()
        recorder_queue = manager.Queue()
        task_queue = manager.Queue(20)
        NUM_PROCESSES = self.config.test.additional.multiprocessing
        ps = []
        bos_id, eos_id, pad_id, sep_id = self.tokenizer.convert_tokens_to_ids(
            ['<BOS>', '<EOS>', '<PAD>', '<SEP>'])

        with torch.no_grad():
            for i in range(NUM_PROCESSES):
                # p = Process(target=test_thread, args=(i,
                #                              self.config,
                #                              self.model,
                #                              task_queue,
                #                              recorder_queue,
                #                              bos_id, eos_id, pad_id, sep_id, ))
                p = spawn(test_thread, args=(i,
                                             self.config,
                                             self.model,
                                             task_queue,
                                             recorder_queue,
                                             bos_id, eos_id, pad_id, sep_id), join=False)
                # print(p)
                # p = spawn(test_thread, args=(i,), join=True)
                ps.append(p)
            print('waiting for subprocesses to finish...')

            # Create Temp folder
            tmp_folder = os.path.join(
                        self.config.results_path, 'tmp', mode)
            if not os.path.exists(tmp_folder):
                create_dirs([tmp_folder])

            i = 1  # dataset index
            for i_batch, sample_batched in enumerate(tqdm(data_loader)):
                try:
                    tmp_path = os.path.join(
                        self.config.results_path, 'tmp', mode, 'tmp{}.pkl'.format(i_batch))
                    task_queue.put((i_batch, sample_batched, tmp_path), block=True)
                    print('new task {} has been initialized'.format(i))
                    i = i + 1
                except Exception as e:
                    print(e)

            # Wait for all processes done
            for p in ps:
                p.join()

            # Read recorder queue until finish all
            count_task = 0
            while count_task < len(data_loader):
                tmp_path = recorder_queue.get(block=True)
                with open(tmp_path, 'rb') as f:
                    log_result = pickle.load(f)
                ootput_ids, sample_batched, attentions = log_result
                print('loaded temp file:', tmp_path)
                # ootput_ids, sample_batched = recorder_queue.get(block=True)

                # Save attentions
                attentions_data = None
                if self.config.test_output_attention:
                    attentions_data = {
                        'attentions': [],
                    }
                    for attention in attentions:
                        att_list = []
                        name_list = self.ds_list + self.ontology_value_list
                        att = attention[0, 0, :, :]
                        for i in range(att.shape[0]):
                            for j in range(att.shape[1]):
                                att_value = att[i, j]
                                if att_value != 0:
                                    # print(name_list[i], att_value, name_list[j])
                                    att_list.append([name_list[i], str(round(att_value, 5)), name_list[j]])
                        attentions_data['attentions'].append(att_list)

                self.test_metrics.add_turn_results_gen_test(ootput_ids, sample_batched, attentions_data)
                count_task += 1

            print('all tasks should have been done...Waiting for subprocesses to finish')
            print('all subprocesses finished.')
            # After validation, print results
            metrics = self.test_metrics.get_metrics()
            # print(metrics)
            metric_str = ''
            for str_ds_pair in metrics.keys():
                metric_str += str_ds_pair
                metric_str += ':'
                for metric_name in metrics[str_ds_pair].keys():
                    metric_str += str(metrics[str_ds_pair][metric_name])
                    metric_str += ' '
            print(metric_str)

            # Save metrics
            metrics_path = os.path.join(self.config.results_path, 'metrics.json')
            with open(metrics_path, 'w') as result_file:
                json.dump(metrics, result_file, indent=4)
                print('metrics has been saved to', metrics_path)

            # Save Results
            self.test_metrics.save_results('{}_results.json'.format(mode))

            # Remove tmp files
            print('Removing tmp files...')
            delete_dir(tmp_folder)
            print('Done.')

    def evaluate(self):
        print('start evaluation')
        start_time = datetime.datetime.now()
        logdir = os.path.join(self.config.tensorboard_path)  # datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        logger.print(logdir)

        writer = SummaryWriter(logdir)

        data_loaders = [('test', self.test_data_loader)]
        if self.config.test.additional.generate_data:
            data_loaders = [
                ('train', self.train_data_loader),
                ('valid', self.valid_data_loader)
                           ] + data_loaders

        ################ GENERATION ################

        bos_id, eos_id, pad_id, sep_id = self.tokenizer.convert_tokens_to_ids(
            ['<BOS>', '<EOS>', '<PAD>', '<SEP>'])
        for mode, data_loader in data_loaders:
            if self.config.test.additional.multiprocessing != 0:
                # use multiprocess
                self.evaluate_multi(mode, data_loader)
            else:
                with torch.no_grad():
                    self.model.eval()
                    # self.model.set_module_to_eval()
                    self.test_metrics.init_session()
                    for i_batch, sample_batched in enumerate(tqdm(data_loader)):
                        tmp_path = os.path.join(
                            self.config.results_path, 'tmp', mode, 'tmp{}.pkl'.format(i_batch))
                        if os.path.exists(tmp_path):
                            print(tmp_path, 'already calculated.. skipped')
                            continue

                        if self.config.test.num_evaluation != 0:
                            if i_batch >= self.config.test.num_evaluation:
                                # early stop
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
                        max_len = min(ctx_len + 300, 1024)
                        output = self.model.generate(input_ids,
                                                     max_length=max_len,
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

                        #	output = generation(model, batch) # same speed as .generate() api above
                        output_ids = output[0].cpu().numpy().tolist()

                        # Save attentions
                        attentions_data = None
                        if self.config.test_output_attention:
                            attentions_data = {
                                'attentions': [],
                            }
                            attentions = self.model.graph_attentions
                            # attentions = [self.model.graph_signal_level]
                            for attention in attentions:
                                att_list = []
                                name_list = self.ds_list + self.ontology_value_list
                                att = attention[0, 0, :, :]
                                for i in range(att.shape[0]):
                                    for j in range(att.shape[1]):
                                        att_value = att[i, j]
                                        if att_value != 0:
                                            # print(name_list[i], att_value, name_list[j])
                                            att_list.append([name_list[i], str(round(att_value, 5)), name_list[j]])
                                attentions_data['attentions'].append(att_list)
                                # print(attention)
                                # att = attention[0, 0].tolist()
                                # attentions_data['attentions'].append(att)
                                # att = attention[0, 0, :N_ds, N_ds:] # N_ds x N_v
                                # print(att)
                                # for ds_id in range(att.shape[0]):
                                #     if 'hotel' not in self.ds_list[ds_id] and 'restaurant' not in self.ds_list[ds_id]:
                                #         continue
                                #     value_attentions = att[ds_id] # N_v
                                #     for value_id in range(value_attentions.shape[0]):
                                #         if value_attentions[value_id] != 0:
                                #             print(self.ds_list[ds_id], '--<{}>--'.format(value_attentions[value_id]), self.ontology_value_id2text[value_id]
                                #                   )

                        self.test_metrics.add_turn_results_gen_test(output_ids, sample_batched, attentions_data)
                        # input()
                        # log_result = (output_ids, sample_batched)
                        # with open(tmp_path, 'wb') as f:
                        #     pickle.dump(log_result, f)
                        # print("temp result cached at:", tmp_path)

                    # After validation, print results
                    metrics = self.test_metrics.get_metrics()
                    # print(metrics)
                    metric_str = ''
                    for str_ds_pair in metrics.keys():
                        metric_str += str_ds_pair
                        metric_str += ':'
                        for metric_name in metrics[str_ds_pair].keys():
                            metric_str += str(metrics[str_ds_pair][metric_name])
                            metric_str += ' '
                    print(metric_str)
                    # Save to result path
                    self.test_metrics.save_results('{}_results.json'.format(mode))

                    # Add to tensorboard
                    # for str_ds_pair in metrics.keys():
                    #     for metric_name in metrics[str_ds_pair].keys():
                    #         writer.add_scalar('test_metrics_{}/{}'.format(metric_name, str_ds_pair),
                    #                           metrics[str_ds_pair][metric_name], self.loaded_epoch)
                    # self.test_metrics.init_session()
                    # writer.flush()
                    # print('Results added!')

def set_seed(seed):
    ''' for reproduction '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


def test_thread(thread_subid, thread_index, config, model, task_queue, recorder_queue, bos_id, eos_id, pad_id, sep_id):
    # Set seed
    if config.test.seed:
        set_seed(config.test.seed)
        print("thread SEED is set to:", config.test.seed)
    print('thread {} initiated'.format(thread_index))
    # Delay 20s
    time.sleep(20)
    print('thread {} started'.format(thread_index))
    model.eval()
    with torch.no_grad():
        while task_queue.qsize() > 0:
            try:
                i_batch, sample_batched, tmp_path = task_queue.get(block=False)
                print('thread {} gets task {}'.format(thread_index, i_batch))
            except Exception as e:
                print(e)
                return

            if os.path.exists(tmp_path):
                print(tmp_path, 'already calculated.. skipped')
                recorder_queue.put(tmp_path)
                time.sleep(0.1)
            else:
                input_ids = sample_batched['input_ids'].to(config.device)
                batch_size, ctx_len = input_ids.size()
                pre_input_ids = sample_batched['pre_input_ids'].to(config.device)
                pre_attention_mask = sample_batched['pre_attention_mask'].to(config.device)
                pre_ds_indice = sample_batched['pre_ds_indice']
                ds_ids = sample_batched['ds_ids']
                batch_size, ctx_len = input_ids.size()
                assert batch_size == 1
                max_len = min(ctx_len + 300, 1024)
                output = model.generate(input_ids,
                                             max_length=max_len,
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

                #	output = generation(model, batch) # same speed as .generate() api above
                output_ids = output[0].cpu().numpy().tolist()
                attentions = None
                if config.test_output_attention:
                    attentions = model.graph_attentions
                log_result = (output_ids, sample_batched, attentions)
                with open(tmp_path, 'wb') as f:
                    pickle.dump(log_result, f)
                recorder_queue.put(tmp_path)
                print("temp result cached at:", tmp_path)
                # recorder_queue.put()
