import sys
import pdb
import json
import pickle
from tqdm import tqdm

from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.data.tokenizers import WordTokenizer, Token
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter


options_file = "/home/pc/.allennlp/cache/18f4665df6985a9653629d15c95755f698de850928c9c78dd8039eb74105a9ef.5f853b1c8192b2a4381c281049440b954c1f3af7a87fb2f3575bed58e74a2376"
weight_file = "/home/pc/.allennlp/cache/10976fbaec55ae3ffb6d4e094724e59c05a73bd8c4f370fd4810c060af4fb214.d5198fa44b348ec6b5e1b378f4963acc528d83766fe9b2651650e04294b89f0e"

class ElmoManager():
    def __init__(self):
        #### init the elmo models
        self.tokenizer = WordTokenizer(word_splitter=SpacyWordSplitter())
        self.elmo = [None] * 3
        self.elmo[0] = Elmo(options_file, weight_file, 1, dropout=0, scalar_mix_parameters=[1.0, 0, 0]).cuda()
        self.elmo[1] = Elmo(options_file, weight_file, 1, dropout=0, scalar_mix_parameters=[0, 1.0, 0]).cuda()
        self.elmo[2] = Elmo(options_file, weight_file, 1, dropout=0, scalar_mix_parameters=[0, 0, 1.0]).cuda()
        self._elmo_warm_up(['am', 'looking', 'for', 'a', 'place', 'to', 'to', 'stay', 'that', 'has', 'cheap', 'price', 'range', 'it', 'should', 'be', 'in', 'a', 'type', 'of', 'hotel', '</S>', '<S>', 'okay', ',', 'do', 'you', 'have', 'a', 'specific', 'area', 'you', 'want', 'to', 'stay', 'in', '?', 'no', ',', 'i', 'just', 'need', 'to', 'make', 'sure', 'it', 's', 'cheap', '.', 'oh', ',', 'and', 'i', 'need', 'parking', '</S>', '<S>', 'i', 'found', '1', 'cheap', 'hotel', 'for', 'you', 'that', 'include', '-s', 'parking', '.', 'do', 'you', 'like', 'me', 'to', 'book', 'it', '?', 'yes', ',', 'please', '.', '6', 'people', '3', 'nights', 'starting', 'on', 'tuesday', '.', '</S>', '<S>', 'i', 'am', 'sorry', 'but', 'i', 'was', 'not', 'able', 'to', 'book', 'that', 'for', 'you', 'for', 'tuesday', '.', 'is', 'there', 'another', 'day', 'you', 'would', 'like', 'to', 'stay', 'or', 'perhaps', 'a', 'shorter', 'stay', '?', 'how', 'about', 'only', '2', 'nights', '.', '</S>', '<S>', 'booking', 'was', 'successful', '.', 'reference', 'number', 'is', ':', '7gawk763', '.', 'anything', 'else', 'i', 'can', 'do', 'for', 'you', '?', 'no', ',', 'that', 'will', 'be', 'all', '.', 'goodbye', '.'])

    def calc_elmo_embeddings(self, dialog):
        # Compute two different representation for each token.
        # Each representation is a linear weighted combination for the
        # 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))

        # use batch_to_ids to convert sentences to character ids
        dialog_content = []
        for item in dialog:
            dialog_content.append([t.text for t in item])
        print(dialog_content)
        character_ids = batch_to_ids(dialog_content).cuda()
        dialog_embeddings = []
        for i in range(3):
            embeddings = self.elmo[i](character_ids)
            batch_embeddings = embeddings['elmo_representations'][0]
            batch_embeddings = batch_embeddings.squeeze(0)
            dialog_embeddings.append(batch_embeddings.cpu())

        return dialog_embeddings

    # https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md
    # After loading the pre-trained model, the first few batches will be negatively impacted until the biLM can reset its internal states. You may want to run a few batches through the model to warm up the states before making predictions (although we have not worried about this issue in practice).
    def _elmo_warm_up(self, dialog):
        character_ids = batch_to_ids(dialog).cuda()
        for i in range(3):
            for _ in range(20):
                res = self.elmo[i](character_ids)


    # def calc_elmo_embeddings(self, dialog):
    #     dialog_embedding = self._calc_elmo_embeddings(dialog)
    #     return dialog_embedding
