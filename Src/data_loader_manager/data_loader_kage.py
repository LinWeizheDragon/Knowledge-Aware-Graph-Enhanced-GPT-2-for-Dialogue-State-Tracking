"""
data_loader_kage.py: Data Loader for the paper:
Lin, W., Tseng, B. H., & Byrne, B. (2021). Knowledge-Aware Graph-Enhanced GPT-2 for Dialogue State Tracking. EMNLP 2021.
https://arxiv.org/abs/2104.04466v3
"""

__author__ = "Weizhe Lin"
__copyright__ = "Copyright 2021, Weizhe Lin"
__version__ = "1.0.0"
__email__ = "wl356@cam.ac.uk"
__status__ = "Published for Github"


import sys
import time
import json
import copy
from itertools import chain
from tqdm import tqdm, trange

import json
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils.util_dst import iterate_dst_file
from copy import deepcopy
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


class DataLoaderWrapper():
    '''
    Data loader for Knowledge-Aware Graph-Enhanced GPT-2 for Dialogue State Tracking
    '''

    def __init__(self, config):
        self.config = config

        ###############################################
        #           Read in ontology data             #
        ###############################################
        ontology_path = config.data_loader.additional.ontology_path
        with open(ontology_path, "r") as f:
            ontology_data = json.load(f)
        # print(ontology_data)
        self.value_text2id = {}
        self.value_id2text = {}
        self.ontology_value_list = []
        self.ontology_value_text2id = {}
        self.ontology_value_id2text = {}

        self.ds_list = []
        self.ds_text2id = {}
        self.ori_ds_mapping = {}
        origin_str_ds_pairs = []
        for str_ds_pair, ds_value_list in ontology_data.items():
            if str_ds_pair == 'all':
                continue
            origin_str_ds_pair = deepcopy(str_ds_pair)
            origin_str_ds_pairs.append(' '.join(origin_str_ds_pair.split('-')))
            # Reformat to lower case and remove bars
            str_ds_pair = str_ds_pair.lower().replace('-', ' ')

            # Save a mapping
            self.ori_ds_mapping[origin_str_ds_pair] = str_ds_pair
            # if str_ds_pair in ['taxi destination',
            #                    'taxi departure',
            #                    'taxi arriveby',
            #                    'restaurant time',
            #                    'taxi leaveat',
            #                    'train arriveby',
            #                    'train leaveat',
            #                    'hotel name',
            #                    'restaurant name',
            #                    'restaurant food',
            #                    'attraction name']:
            #     ds_value_list = ['', 'dont care', 'span']
            # print(ds_value_list)
            value2id_dict = {
                'none': 0,
                'dont care': 1,
            }
            id2value_dict = {
                0: 'none',
                1: 'dont care',
            }
            for value in ds_value_list:
                if value in ['does not care', 'dontcare', 'none', '']:
                    continue
                if value not in value2id_dict.keys():
                    value2id_dict[value] = len(value2id_dict.keys())
                if value not in id2value_dict.values():
                    id2value_dict[len(id2value_dict.keys())] = value
            self.value_text2id[str_ds_pair] = value2id_dict
            self.value_id2text[str_ds_pair] = id2value_dict
            for value in value2id_dict.keys():
                if value not in self.ontology_value_list:
                    self.ontology_value_list.append(value)
            self.ds_list.append(str_ds_pair)
            self.ds_text2id[str_ds_pair] = len(self.ds_text2id)

        origin_str_ds_pairs = sorted(origin_str_ds_pairs)
        if self.config.data_loader.additional.reverse_slot_order:
            origin_str_ds_pairs.reverse()
        self.gen_ds_indice = origin_str_ds_pairs
        print('Generation domain-slot order:', self.gen_ds_indice)

        print('Ontology: value nodes:', len(self.ontology_value_list))
        for index, value in enumerate(self.ontology_value_list):
            self.ontology_value_text2id[value] = index
            self.ontology_value_id2text[index] = value
        # print(self.ontology_value_id2text)

        self.SPECIAL_TOKENS = {
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
            "pad_token": "<PAD>",
            "sep_token": "<SEP>",
            "additional_special_tokens": ["<USR>", "<SYS>", "<BOC>", "<CLS>"]
            + ['<{}>'.format(str_ds_pair.replace(' ', '_')) for str_ds_pair in self.ds_list]
        }

        self.SPECIAL_TOKENS_VALUES = ["<BOS>", "<EOS>", "<PAD>", "<SEP>", "<USR>",
                                      "<SYS>", "<BOC>", "<CLS>"]  + ['<{}>'.format(str_ds_pair.replace(' ', '_')) for str_ds_pair in self.ds_list]

    def set_dataloader(self, config, tokenizer, data_type, run_type,
                       value_id2text, value_text2id, ds_list, ds_text2id, data_size=-1):
        """This function create PyTorch DataLoader instances

        Args:
            config (EasyDict): global config
            tokenizer (Transformer tokenizer): tokenizer to use
            data_type (String): train/test/valid
            run_type (String): "generation" to run generation data loader
            value_id2text (Dict): id2text mapping of value
            value_text2id (Dict): text2id mapping of value
            ds_list (List): Domain-slot list
            ds_text2id (Dict): text2id mapping of domain-slots
            data_size (int, optional): how many data entries to load. Defaults to -1, unlimited.

        Returns:
            torch.utils.data.DataLoader: Data Loader
            torch.utils.data.Dataset: Dataset for creating data loader
        """
        dataset = Dataset(config, tokenizer, data_type, run_type == 'generation',
                          value_id2text, value_text2id, ds_list, ds_text2id, self.ori_ds_mapping, self.gen_ds_indice,
                          self.SPECIAL_TOKENS, self.SPECIAL_TOKENS_VALUES, data_size)
        if data_type == 'train' and config.mode=='train':
            sampler = RandomSampler(dataset)  # if args.local_rank == -1 else DistributedSampler(train_dataset)
        else:
            sampler = SequentialSampler(dataset)

        batch_size = 1
        if data_type == 'train':
            batch_size = config.train.batch_size
        elif data_type == 'valid':
            if run_type == 'generation':
                batch_size = config.valid.valid_gen_batch_size
            else:
                batch_size = config.valid.batch_size
        elif data_type == 'test':
            batch_size = config.test.batch_size
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            num_workers=4,
        )
        return dataloader, dataset


class Dataset(torch.utils.data.Dataset):
    '''
    Torch Dataset for DST generation/training
    '''
    def __init__(self, config, tokenizer, data_type, generation,
                 value_id2text, value_text2id, ds_list, ds_text2id, ori_ds_mapping, gen_ds_indice,
                 SPECIAL_TOKENS, SPECIAL_TOKENS_VALUES,
                 data_size):
        assert data_type in ['train', 'dev', 'test', 'demo']
        self.config = config
        self.data_size = data_size
        self.tokenizer = tokenizer
        self.data_type = data_type
        self.generation = generation
        self.value_id2text = value_id2text
        self.value_text2id = value_text2id
        self.ds_list = ds_list
        self.ds_text2id = ds_text2id
        self.ori_ds_mapping = ori_ds_mapping
        self.ori_ds_mapping_reverse = {}
        for key, value in ori_ds_mapping.items():
            self.ori_ds_mapping_reverse[value] = key
        self.gen_ds_indice = gen_ds_indice
        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES
        self.domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi']
        self._get_special_token_ids()
        self._process_dst(data_type)
        self.ds_tokenized_ids = []
        self.len_gen_ds_pair = []
        for gen_ds_pair in self.gen_ds_indice:
            # print(gen_ds_pair, self.tokenizer.encode(gen_ds_pair))
            self.len_gen_ds_pair.append(len(self.tokenizer.encode(gen_ds_pair)))
            
        self._create_examples()


    def _process_dst(self, split):
        # Preprocess dst data
        #		with open(self.args.dst_slot_list) as f:
        #			dst_slot_list = json.load(f)
        #		dst_cont = {} # data container
        dst_files = {
            'train': self.config.data_loader.additional.train_data_path,
            'dev': self.config.data_loader.additional.validation_data_path,
            'test': self.config.data_loader.additional.test_data_path
        }
        dst_f = dst_files[split]
        #		with open(dst_f) as f1:
        #			dst_data = json.load(f1)
        #		dial_n, example_n = iter_dst_file(dst_cont, dst_data, None, dst_slot_list, \
        #						remove_dontcare=self.args.remove_dontcare, fix_wrong_domain=self.args.fix_wrong_domain, display=False)
        dst_cont, dial_n, example_n = iterate_dst_file(dst_f)
        print('{} -> # of dialogues: {}, examples: {}'.format(split, dial_n, example_n))
        self.dst_data = dst_cont

    def _get_special_token_ids(self):
        self.SPECIAL_TOKENS = self.SPECIAL_TOKENS
        self.SPECIAL_TOKENS_VALUES = self.SPECIAL_TOKENS_VALUES
        self.bos_id = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["bos_token"])
        self.eos_id = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["eos_token"])
        self.pad_id = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["pad_token"])
        self.sep_id = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["sep_token"])
        self.boc_id = self.tokenizer.convert_tokens_to_ids("<BOC>")
        self.cls_id = self.tokenizer.convert_tokens_to_ids("<CLS>")
        self.ds_special_ids = self.tokenizer.convert_tokens_to_ids(['<{}>'.format(str_ds_pair.replace(' ', '_')) for str_ds_pair in self.ds_list])
        self.ds_special_tokens = ['<{}>'.format(str_ds_pair.replace(' ', '_')) for str_ds_pair in self.ds_list]
        print('SPECIAL TOKEN MAPPING:')
        print('bos:{} | eos:{} | pad:{} | sep:{} | boc:{} | cls:{}'.format(
            self.bos_id, self.eos_id, self.pad_id, self.sep_id, self.boc_id, self.cls_id))
        '''
        if using BPE, no need unk_token
        if using convert_tokens_to_ids, check which is correct way to handle oov:
            a) simply use <endoftext> as unk_token (default setup) or
            b) add unk_token into special tokens
        '''

    def dict2sorted_str(self, bs_dict):
        '''
        convert dict of {d-s: v} into string with sorted order as d1 s1 v1 <SEP> d2 s2 v2 <SEP> d3 s3 v3
        sort by first domain, then slot
        '''
        out = []

        for ds, v in bs_dict.items():
            d, s = ds.split('-')
            out.append("{} {} {}".format(d, s, v))
        out = sorted(out)
        if self.config.data_loader.additional.reverse_slot_order:
            out.reverse()
        out = " {} ".format(self.SPECIAL_TOKENS["sep_token"]).join(out)
        return out

    def dict2sorted_str_without_values(self, bs_dict):
        '''
        convert dict of {d-s: v} into string with sorted order as d1 s1 <SEP> d2 s2 <SEP> d3 s3
        sort by first domain, then slot
        remove values
        '''
        out = []
        for ds, v in bs_dict.items():
            d, s = ds.split('-')
            out.append("{} {}".format(d, s))
        out = sorted(out)
        if self.config.data_loader.additional.reverse_slot_order:
            out.reverse()
        # out = " {} ".format(self.SPECIAL_TOKENS["sep_token"]).join(out)
        # Use cls for separation
        # out = " {} ".format('<CLS>').join(out)
        # return out
        out_str = ''
        for index, bs_str in enumerate(out):
            out_str += bs_str
            out_str += ' {} '.format(self.ds_special_tokens[index])
        out_str = out_str.strip()
        return out_str

    def normalize_value(self, bs_dict):
        '''
        normalize value in belief state using gpt decode function in case any mismatch
        '''
        for ds, v in bs_dict.items():
            ids = self.tokenizer(v)['input_ids']
            v_recons = self.tokenizer.decode(ids)
            # if '|' in v_recons:
            #     v_recons = v_recons.split('|')[0]
            bs_dict[ds] = v_recons  # NOTE: very few (only 5 values in less than 20 examples) are different in train set

    #			if v != v_recons:
    #				print(v, v_recons)

    def fill_empty(self, bs_dict):
        """Fill empty slots in the bs_dict

        Args:
            bs_dict (Dict): slot value dict

        Returns:
            Dict: Updated bs_dict
        """
        for ori_str_ds_pair in self.ori_ds_mapping.keys():
            if ori_str_ds_pair not in bs_dict.keys():
                bs_dict[ori_str_ds_pair] = 'none'
        return bs_dict


    def dict2label(self, bs_dict):
        """Transform values to classification labels

        Args:
            bs_dict (Dict): dict of values

        Returns:
            Dict: cls label ids
            Dict: label values
            Boolean: Whether the CLS label failed to be found from ontology
        """
        out = {}
        values = {}
        fail = False
        for str_ds_pair in self.ds_list:
            out[str_ds_pair] = 0
            values[str_ds_pair] = ''
        # print(out)
        for ds, v in bs_dict.items():
            d, s = ds.lower().split('-')
            str_ds_pair = ' '.join([d, s])
            # print(str_ds_pair)
            if str_ds_pair in out.keys():
                if v == '':
                    v = 'none'
                if v == 'does not care':
                    v = 'dont care'
                if v == 'dontcare':
                    v = 'dont care'
                if v in self.value_text2id[str_ds_pair].keys():
                    # print(v, 'found', str_ds_pair)
                    cls_id = self.value_text2id[str_ds_pair][v]
                else:
                    if 'span' in self.value_text2id[str_ds_pair].keys():
                        cls_id = self.value_text2id[str_ds_pair]['span']
                    else:
                        cls_id = 0
                        fail = True
                        print(v, 'not found in', str_ds_pair)
                        # input('stopped')
                out[str_ds_pair] = cls_id
                values[str_ds_pair] = v
        return out, values, fail

    def _create_examples(self):
        self.examples = []
        load_only_last_turn = False
        if self.config.valid.last_turn_only_generation:
            if self.data_type == 'dev' and self.generation:
                load_only_last_turn = True
                print('Loading only last turn for validation...')

        if self.config.data_loader.additional.only_last_turn:
            if self.data_type == 'dev' and self.generation:
                load_only_last_turn = False
                print('Loading all turns for validation [testing mid turn performance]...')
            else:
                load_only_last_turn = True
                print('Loading only last turn as --only_last_turn flag is set...')

        slot_label_counter = {}

        for example_num, example_id in enumerate(tqdm(sorted(self.dst_data.keys()))):
            if self.data_size != -1 and example_num == self.data_size:
                break
            # print(self.dst_data[example_id]['current_turn_index'], self.dst_data[example_id]['total_num_turn'])
            if load_only_last_turn:
                # Use only last turn for validation
                if self.dst_data[example_id]['current_turn_index'] < self.dst_data[example_id]['total_num_turn'] - 1:
                    continue
            context = self.dst_data[example_id]['context']  # str of word token
            turn_utt = self.dst_data[example_id]['turn_utt']
            bs_ref_dict = self.dst_data[example_id]['belief_state']  # dict

            # TODO fill empty
            bs_dict = self.fill_empty(bs_ref_dict)

            self.normalize_value(bs_ref_dict)
            bs_str = self.dict2sorted_str(bs_ref_dict)
            bs_str_without_values = self.dict2sorted_str_without_values(bs_ref_dict)
            # print(bs_ref_dict)
            cls_label_dict, value_label_dict, fail_flag = self.dict2label(bs_ref_dict)
            # print(cls_label_dict, value_label_dict, fail_flag)
            if fail_flag:
                continue
            for str_ds_pair, value in bs_ref_dict.items():
                slot_label_counter.setdefault(str_ds_pair, 0)
                if value != 'none':
                    slot_label_counter[str_ds_pair] += 1
            
            context_ids = self.tokenizer(context)['input_ids']
            target_ids = self.tokenizer(bs_str)['input_ids']  # TODO: Q3
            pre_target_ids = self.tokenizer(bs_str_without_values)['input_ids']
            
            target_len = len(target_ids)
            if not self.generation:
                # dialogue_context <BOS> d1 s1 v1 <SEP> d2 s2 v2 <SEP> d3 s3 v3 <EOS>
                input_ids = context_ids + [self.bos_id] + target_ids + [self.eos_id]
                ignore_len = len(input_ids) - target_len - 1  # eos_id
                label_ids = [-100] * ignore_len + target_ids + [self.eos_id]
                assert len(input_ids) == len(label_ids)
                if len(input_ids) >= 1024:  # handle over-length example
                    input_ids = input_ids[-1023:]
                    label_ids = label_ids[-1023:]

            else:
                input_ids = context_ids + [self.bos_id]  #  give bos for generate() api
                label_ids = None
                if len(input_ids) >= 1024:
                    input_ids = input_ids[-1023:]

            if self.config.use_graph:
                # dialogue_context <BOS> d1 s1 <SEP> d2 s2 <SEP> d3 s3 <EOS>
                # pre_input_ids = context_ids + [self.bos_id] + pre_target_ids + [self.eos_id]
                # dialogue_context <BOC> d1 s1 <CLS> d2 s2 <CLS> d3 s3 <CLS>
                pre_input_ids = context_ids + [self.boc_id] + pre_target_ids + [self.cls_id]
                if len(pre_input_ids) >= 1024:
                    pre_input_ids = pre_input_ids[-1023:]
                pre_ignore_len = len(pre_input_ids) - len(pre_target_ids) - 1
                pre_ds_indice = []
                ds_indice = []
                # print(self.tokenizer.convert_ids_to_tokens(pre_input_ids))
                for token_index, token_id in enumerate(pre_input_ids):
                    # if token_id == self.sep_id or token_id == self.eos_id:
                    # print(token_id, self.cls_id)
                    # TODO use diff separation
                    # if token_id == self.cls_id:
                    if token_id in self.ds_special_ids:
                        # end of domain-slot pairs in pre-sequence
                        # pre_ds_indice.append(token_index-1) # index of end of ds pairs
                        pre_ds_indice.append(token_index) # TODO: Index of <CLS> token
                        # print(self.tokenizer.convert_ids_to_tokens(pre_input_ids[token_index]))
                    # print(len(pre_ds_indice))
                assert len(pre_ds_indice) == len(self.len_gen_ds_pair)

                if len(self.ds_tokenized_ids) == 0:
                    current_index = 0
                    for token_index, token_id in enumerate(pre_target_ids + [self.cls_id]):
                        # TODO use diff separation
                        # if token_id == self.cls_id:
                        if token_id in self.ds_special_ids:
                        # if token_id == self.sep_id or token_id == self.eos_id:
                            self.ds_tokenized_ids.append(pre_target_ids[current_index:token_index])
                            # print(self.tokenizer.convert_ids_to_tokens(pre_target_ids[current_index:token_index]))
                            current_index = token_index+1
                    assert len(self.ds_tokenized_ids) == len(self.len_gen_ds_pair)
                    

                count = 0
                if not self.generation:
                    # Teacher force, get span indice
                    for token_index, token_id in enumerate(input_ids):
                        # print(self.tokenizer.convert_ids_to_tokens([token_id]))
                        if token_id == self.bos_id or token_id == self.sep_id:
                            # span indice of corresponding domain-slot pairs
                            ds_indice.append([token_index + 1, token_index + 1 + self.len_gen_ds_pair[count]])
                            # print(self.tokenizer.convert_ids_to_tokens(input_ids[token_index + 1:token_index + 1 + self.len_gen_ds_pair[count]]))
                            count += 1
                    assert len(ds_indice) == len(self.len_gen_ds_pair)

            else:
                pre_input_ids = None
                pre_ignore_len = None
                pre_ds_indice = None
                ds_indice = None
                ds_ids = None

            assert len(input_ids) < 1024
            self.examples.append({
                'pre_input_ids': pre_input_ids,
                'pre_ignore_len': pre_ignore_len,
                'pre_ds_indice': pre_ds_indice,
                'input_ids': input_ids,  # list of ids
                'label_ids': label_ids,  # list of ids
                'ds_indice': ds_indice,
                'cls_label_dict': cls_label_dict,
                'cls_label': [cls_label_dict[str_ds_pair] for str_ds_pair in self.ds_list],
                # 'value_label': value_labels,
                # 'ignore_len': ignore_len,
                'context': context,
                'turn_utt': turn_utt,
                'bs_dict': bs_dict,
                'bs_str': bs_str,
                'example_id': example_id,
                # 'str_ds_pair': [str_ds_pair for str_ds_pair, ds_id in self.ds_text2id.items()]
            })

        if self.data_type != 'demo':
            print(slot_label_counter)
            print('Data Statistics: {} -> {} examples'.format(self.data_type, len(self.examples)))
            print('Data Statistics: {} -> {} examples'.format(self.data_type, len(self.examples)), file=sys.stderr)


    def _pad(self, sentences, pad_id):
        '''
            Pad batch of sentences with 0
            sentences: a list of list with ids
        '''
        max_len = max((map(len, sentences)))
        attention_mask = []
        sentences_pad = []
        for sent in sentences:
            pad_len = max_len - len(sent)
            sentences_pad.append(sent + [pad_id] * pad_len)
            attention_mask.append([1] * len(sent) + [0] * pad_len)
        return sentences_pad, attention_mask

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        '''
            index will be ramdomly sampled by the fed sampler, we dont need to worry about index
        '''
        return self.examples[index]

    def collate_fn(self, batch):
        '''
            when collate_fn is given to the torch dataloader, we can do further actions to the batch, e.g., tensor can be formed here
            a batch is formed as a list where each element is a defined data returned by __getitem__, andy
        '''
        input_ids = [example['input_ids'] for example in batch]
        input_ids, attention_mask = self._pad(input_ids, self.pad_id)
        input_ids, attention_mask = torch.tensor(input_ids).long(), torch.tensor(
            attention_mask).long()

        if self.config.use_graph:
            # When use graph, prepare pre sequence
            pre_input_ids = [example['pre_input_ids'] for example in batch]
            pre_input_ids, pre_attention_mask = self._pad(pre_input_ids, self.pad_id)
            pre_input_ids, pre_attention_mask = torch.tensor(pre_input_ids).long(), torch.tensor(
                pre_attention_mask).long()
            pre_ignore_len = [ex['pre_ignore_len'] for ex in batch]
        else:
            pre_input_ids = None
            pre_attention_mask = None
            pre_ignore_len = None

        if not self.generation:
            label_ids = [example['label_ids'] for example in batch]
            label_ids, _ = self._pad(label_ids, -100)
            label_ids = torch.tensor(label_ids).long()
        else:
            label_ids = None
        token_type_ids = None

        # store info for scoring
        cls_labels = [ex['cls_label'] for ex in batch]
        cls_label_dict = [ex['cls_label_dict'] for ex in batch]
        context = [ex['context'] for ex in batch]
        bs_dict = [ex['bs_dict'] for ex in batch]
        bs_str = [ex['bs_str'] for ex in batch]
        example_id = [ex['example_id'] for ex in batch]
        turn_utt = [ex['turn_utt'] for ex in batch]
        ds_indice = [ex['ds_indice'] for ex in batch]
        pre_ds_indice = [ex['pre_ds_indice'] for ex in batch]
        return {
            'pre_input_ids': pre_input_ids,
            'pre_attention_mask': pre_attention_mask,
            'pre_ignore_len': pre_ignore_len,
            'pre_ds_indice': pre_ds_indice,
            'cls_labels': cls_labels,
            'cls_label_dict': cls_label_dict,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'ds_indice': ds_indice,
            'ds_ids': self.ds_tokenized_ids,
            'token_type_ids': token_type_ids,
            'label_ids': label_ids,
            'context': context, 'bs_dict': bs_dict, 'bs_str': bs_str, 'example_id': example_id,
            'turn_utt': turn_utt
        }


if __name__ == '__main__':
    train_data_loader, train_dataset = data_loader.set_dataloader(config, self.tokenizer, 'train',
                                                                       'teacher_force',
                                                                       self.value_id2text,
                                                                       self.value_text2id,
                                                                       self.ds_list,
                                                                       self.ds_text2id,
                                                                       data_size=load_num)
