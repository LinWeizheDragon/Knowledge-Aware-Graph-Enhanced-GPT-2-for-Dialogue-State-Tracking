import sys
import time
import json
import copy
from itertools import chain
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader, RandomSampler
from utils.util_dst import iterate_dst_file

# checklist
# Q1: when scoring, make sure value string is processed using same tokenizer in both gt and pred since it matters for exact match! - Done
# Q2: tune how much different when consider dontcare - Done
# Q3: some key words are separate into several tokens, e.g., restaurant into res + aur + ant, due to BPE, not sure if needs to keep them undivided as special tokens for consistency
# Q4: work on MultiWOZ2.1 - Done

SPECIAL_TOKENS = {
	"bos_token": "<BOS>",
	"eos_token": "<EOS>",
	"pad_token": "<PAD>",
	"sep_token": "<SEP>",
	"additional_special_tokens": ["<USR>", "<SYS>"]
}

SPECIAL_TOKENS_VALUES = ["<BOS>", "<EOS>", "<PAD>", "<SEP>", "<USR>", "<SYS>"]


class Dataset(torch.utils.data.Dataset):
	def __init__(self, config, tokenizer, data_type, generation, data_size):
		assert data_type in ['train', 'dev', 'test', 'demo']
		self.config = config
		self.data_size = data_size
		self.tokenizer = tokenizer
		self.data_type = data_type
		self.generation = generation

		self.domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi']
		self._get_special_token_ids()
		self._process_dst(data_type)
		self._create_examples()


	def _process_dst(self, split):
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
		self.SPECIAL_TOKENS = SPECIAL_TOKENS
		self.SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES
		self.bos_id = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["bos_token"])
		self.eos_id = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["eos_token"])
		self.pad_id = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["pad_token"])
		self.sep_id = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["sep_token"])
		print('SPECIAL TOKEN MAPPING:')
		print('bos:{} | eos:{} | pad:{} | sep:{}'.format(self.bos_id, self.eos_id, self.pad_id, self.sep_id))
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
			out.append("{} {} {}".format(d,s,v))
		out = sorted(out)
		out = " {} ".format(self.SPECIAL_TOKENS["sep_token"]).join(out)
		return out


	def normalize_value(self, bs_dict):
		'''
		normalize value in belief state using gpt decode function in case any mismatch
		'''
		for ds, v in bs_dict.items():
			ids = self.tokenizer(v)['input_ids']
			v_recons = self.tokenizer.decode(ids)
			bs_dict[ds] = v_recons # NOTE: very few (only 5 values in less than 20 examples) are different in train set
#			if v != v_recons:
#				print(v, v_recons)

	def _create_examples(self):
		self.examples = []
		for example_num, example_id in enumerate(tqdm(sorted(self.dst_data.keys()))):
			if self.data_size != -1 and example_num == self.data_size:
				break

			context = self.dst_data[example_id]['context'] # str of word token
			turn_utt = self.dst_data[example_id]['turn_utt']
			bs_dict = self.dst_data[example_id]['belief_state'] # dict
			self.normalize_value(bs_dict)
			bs_str = self.dict2sorted_str(bs_dict)
			# if self.args.remove_dontcare:
			# 	assert 'dontcare' not in bs_str
		
			context_ids = self.tokenizer(context)['input_ids']
			target_ids = self.tokenizer(bs_str)['input_ids'] # TODO: Q3
			target_len = len(target_ids)
			if not self.generation:
				# dialogue_context <BOS> belief_state <EOS>
				input_ids = context_ids + [self.bos_id] + target_ids + [self.eos_id]
				ignore_len = len(input_ids) - target_len - 1 # eos_id
				label_ids = [-100] * ignore_len + target_ids + [self.eos_id]
				assert len(input_ids) == len(label_ids)
				if len(input_ids) >= 1024: # handle over-length example
					input_ids = input_ids[-1023:]
					label_ids = label_ids[-1023:]
			else:
				input_ids = context_ids + [self.bos_id] # give bos for generate() api
				label_ids = None
				if len(input_ids) >= 1024:
					input_ids = input_ids[-1023:]

			assert len(input_ids) < 1024
			self.examples.append({
				'input_ids': input_ids, # list of ids
				'label_ids': label_ids, # list of ids
				'context': context,
				'turn_utt': turn_utt,
				'bs_dict': bs_dict,
				'bs_str': bs_str, 
				'example_id': example_id,
			})

		if self.data_type != 'demo':
			print('Data Statistics: {} -> {} examples'.format(self.data_type, len(self.examples)))
			print('Data Statistics: {} -> {} examples'.format(self.data_type, len(self.examples)), file=sys.stderr)


	def _pad(self, sentences, pad_id):
		'''
			sentences: a list of list with ids
		'''
		max_len = max((map(len, sentences)))
		attention_mask = []
		sentences_pad = []
		for sent in sentences:
			pad_len = max_len - len(sent)
			sentences_pad.append( sent + [pad_id]*pad_len )
			attention_mask.append( [1]*len(sent) + [0]*pad_len)
		return sentences_pad, attention_mask


	def __len__(self): # required
		return len(self.examples)


	def __getitem__(self, index): # required
		'''
			index will be ramdomly sampled by the fed sampler, we dont need to worry about index
		'''
		return self.examples[index]


	def collate_fn(self, batch): # optional but useful
		'''
			when collate_fn is given to the torch dataloader, we can do further actions to the batch, e.g., tensor can be formed here
			a batch is formed as a list where each element is a defined data returned by __getitem__, andy
		'''
		input_ids = [example['input_ids'] for example in batch]
		input_ids, attention_mask = self._pad(input_ids, self.pad_id)
		input_ids, attention_mask = torch.tensor(input_ids).long().to(self.config.device), torch.tensor(attention_mask).long().to(self.config.device)

		if not self.generation:
			label_ids = [example['label_ids'] for example in batch]
			label_ids, _ = self._pad(label_ids, -100)
			label_ids = torch.tensor(label_ids).long().to(self.config.device)
		else:
			label_ids = None
		token_type_ids = None

		# store info for scoring
		context = [ex['context'] for ex in batch]
		bs_dict = [ex['bs_dict'] for ex in batch]
		bs_str = [ex['bs_str'] for ex in batch]
		example_id = [ex['example_id'] for ex in batch]
		turn_utt = [ex['turn_utt'] for ex in batch]

		return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids, 'label_ids': label_ids, \
				'context': context, 'bs_dict': bs_dict, 'bs_str': bs_str, 'example_id': example_id, 'turn_utt': turn_utt}


if __name__ == '__main__':
	pass
