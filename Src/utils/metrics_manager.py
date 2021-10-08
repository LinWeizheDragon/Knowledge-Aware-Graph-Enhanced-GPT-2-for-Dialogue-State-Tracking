import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.util_dst import *

class MetricsManager():
    def __init__(self, config, ds_list, value_text2id,  value_id2text, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.ds_list = ds_list
        self.value_text2id = value_text2id
        self.value_id2text = value_id2text
        self.init_session()
        print('Metrics Manager Initialized.')

    def init_session(self):
        self.results = []
        self.turn_results = []
        self.list_cls_record = {}
        self.list_notnone_cls_record = {}
        self.list_token_record = {}
        self.list_span_record = {}
        for str_ds_pair in self.ds_list:
            self.list_cls_record[str_ds_pair] = []
            self.list_notnone_cls_record[str_ds_pair] = []
            self.list_token_record[str_ds_pair] = []
            self.list_span_record[str_ds_pair] = []

    def check_token_result(self, input_ids, token_prediction, value_label, verbose=False):
        decoded_value_label = self.tokenizer.decode(self.tokenizer.encode(value_label)[1:-1])
        decoded_predicted_label = self.tokenizer.decode(input_ids[token_prediction == 1])
        if verbose:
            print('check predicted:', decoded_predicted_label, 'value label', decoded_value_label)

        return self.tokenizer.encode(value_label)[1:-1], list(input_ids[token_prediction == 1].numpy())

    def add_results(self, token_predictions, cls_predictions, span_predictions, sampled_batch, verbose=False):
        for index, cls_prediction in enumerate(cls_predictions):
            # print(index)
            # print(sampled_batch['dialog_content'][index])
            # print(sampled_batch['replaced_dialog_content'][index])
            # print(sampled_batch['question'][index])
            # print(sampled_batch['label'][index])
            str_ds_pair = sampled_batch['str_ds_pair'][index]
            input_ids = sampled_batch['input']['input_ids'][index][0]
            value_label = sampled_batch['label'][index]
            cls_label = sampled_batch['cls_label'][index]

            # print(sampled_batch['dialog_content'][index])
            # print('value label:', value_label)
            # print('cls label:', cls_label)
            # input()
            start_span_label = sampled_batch['span_label'][0][index]
            end_span_label = sampled_batch['span_label'][1][index]

            '''
            Token Classification
            '''
            if token_predictions is not None:
                token_prediction = token_predictions[index]
                # For debug only
                # token_prediction = token_label_array

                # print(input_ids[token_label_array==1])
                # print(self.tokenizer.decode(self.tokenizer.encode()))
                # print(self.tokenizer.decode(input_ids[token_label_array==1]))

                if value_label is not None and value_label not in ['yes', 'no', 'none', 'dont care']:
                    label_ids, predicted_ids = self.check_token_result(input_ids, token_prediction, value_label, verbose=verbose)
                    # print(label_ids, predicted_ids)
                    if predicted_ids[:len(label_ids)] == label_ids:
                        # print('token result correct')
                        self.list_token_record[str_ds_pair].append(1)
                    else:
                        # print('token result wrong')
                        self.list_token_record[str_ds_pair].append(0)
                else:
                    if np.count_nonzero(token_prediction) > 0:
                        # print('tokens wrongly classified')
                        self.list_token_record[str_ds_pair].append(0)
                    else:
                        # print('not making predictions, correct')
                        self.list_token_record[str_ds_pair].append(1)
            '''
            Classification
            '''
            # if verbose:
            #     print('label {} predict: {}'.format(cls_label, cls_prediction))
            if cls_prediction == cls_label:
                self.list_cls_record[str_ds_pair].append(1)
                if cls_label != 0: # Has value
                    # print('cls_label is', cls_label)
                    self.list_notnone_cls_record[str_ds_pair].append(1)
            else:
                self.list_cls_record[str_ds_pair].append(0)
                if cls_label != 0: # Has value
                    # print('cls_label is', cls_label)
                    self.list_notnone_cls_record[str_ds_pair].append(0)
            '''
            Span Prediction
            '''
            if span_predictions is not None:
                start_span_prediction = span_predictions[0][index]
                end_span_prediction = span_predictions[1][index]
                if verbose:
                    print(start_span_label, end_span_label, 'predict:', start_span_prediction, end_span_prediction)
                if start_span_label == start_span_prediction and end_span_label == end_span_prediction:
                    self.list_span_record[str_ds_pair].append(1)
                else:
                    self.list_span_record[str_ds_pair].append(0)

        if verbose:
            print('finished adding results')
            print(self.list_token_record)
            print(self.list_cls_record)
            print(self.list_notnone_cls_record)
            print(self.list_span_record)

    def add_turn_results(self, token_predictions, cls_predictions, span_predictions, sampled_batch, verbose=False):
        # print(sampled_batch)
        turn_list_result = []
        for index, batch_cls_predictions in enumerate(cls_predictions):
            dialog_contents = sampled_batch['dialog_content'] # length B
            str_ds_pairs = sampled_batch['str_ds_pair'][index] # length B
            input_ids = sampled_batch['input']['input_ids'] # length B
            value_labels = sampled_batch['labels'][index] # length B
            cls_labels = sampled_batch['cls_labels'][:, index] # B x 1
            span_labels = sampled_batch['span_labels'][:, index] # B x 2
            batch_span_predictions = span_predictions[index] # 2 * B

            if index == 0:
                turn_list_result = [True] * cls_labels.shape[0] # length B

            for i in range(len(dialog_contents)):
                # each item in the batch
                dialog_content = dialog_contents[i]
                str_ds_pair = str_ds_pairs[i]
                input_id_sequence = input_ids[i][0]
                value_label = value_labels[i]
                cls_label = int(cls_labels[i].numpy())
                cls_prediction = batch_cls_predictions[i]
                span_label_start = span_labels[i, 0]
                span_label_end = span_labels[i, 1]
                span_prediction_start = batch_span_predictions[0][i]
                span_prediction_end = batch_span_predictions[1][i]

                # print(dialog_content)
                # print('current ds', str_ds_pair)
                # print(span_label_start, span_label_end, span_prediction_start, span_prediction_end)
                # str_span_label = self.tokenizer.decode(input_id_sequence[span_label_start:span_label_end])
                # str_span_prediction = self.tokenizer.decode(input_id_sequence[span_prediction_start:span_prediction_end])
                # print(str_span_label, '->', str_span_prediction)
                # str_cls_label = self.value_id2text[str_ds_pair][cls_label]
                # str_cls_prediction = self.value_id2text[str_ds_pair][cls_prediction]
                # print(str_cls_label, str_cls_prediction)
                # input()
                flag = True

                '''
                Classification
                '''
                if verbose:
                    print(index, i, 'label {} predict: {}'.format(cls_label, cls_prediction))
                if cls_prediction == cls_label:
                    self.list_cls_record[str_ds_pair].append(1)
                    if cls_label != 0: # Has value
                        # print('cls_label is', cls_label)
                        self.list_notnone_cls_record[str_ds_pair].append(1)
                else:
                    flag = False
                    self.list_cls_record[str_ds_pair].append(0)
                    if cls_label != 0: # Has value
                        # print('cls_label is', cls_label)
                        self.list_notnone_cls_record[str_ds_pair].append(0)

                '''
                Span Prediction
                '''
                # if verbose:
                #     print(span_label_start, span_label_end, 'predict:', span_prediction_start, span_prediction_end)
                if span_label_start == 0 and span_label_end == 0:
                    pass
                else:
                    if span_label_start == span_prediction_start and span_label_end == span_prediction_end:
                        self.list_span_record[str_ds_pair].append(1)
                    else:
                        # flag = False
                        self.list_span_record[str_ds_pair].append(0)

                if turn_list_result[i] == True and flag == False:
                    turn_list_result[i] = False

                # print(turn_list_result)
                # print('...')
        self.turn_results += turn_list_result
        # if verbose:
        #     print(self.turn_results)
        #     print('finished adding results')
        #     print(self.list_token_record)
        #     print(self.list_cls_record)
        #     print(self.list_notnone_cls_record)
        #     print(self.list_span_record)

    def add_turn_results_graph(self, cls_predictions, sampled_batch, verbose=False):
        for index, batch_cls_predictions in enumerate(cls_predictions):
            cls_labels = torch.LongTensor(sampled_batch['cls_labels'])[:, index]  # B x 1
            dialog_contents = sampled_batch['turn_utt']
            if index == 0:
                turn_list_result = [1] * cls_labels.shape[0]  # length B
            str_ds_pair = self.ds_list[index]
            for i in range(len(dialog_contents)):
                # each item in the batch
                dialog_content = dialog_contents[i]

                cls_label = int(cls_labels[i].cpu().numpy())
                cls_prediction = batch_cls_predictions[i]

                flag = True

                '''
                Classification
                '''
                if verbose:
                    print(index, i, 'label {} predict: {}'.format(cls_label, cls_prediction))
                if cls_prediction == cls_label:
                    self.list_cls_record[str_ds_pair].append(1)
                    if cls_label != 0:  # Has value
                        # print('cls_label is', cls_label)
                        self.list_notnone_cls_record[str_ds_pair].append(1)
                else:
                    flag = False
                    self.list_cls_record[str_ds_pair].append(0)
                    if cls_label != 0:  # Has value
                        # print('cls_label is', cls_label)
                        self.list_notnone_cls_record[str_ds_pair].append(0)

                if turn_list_result[i] == 1 and flag == False:
                    turn_list_result[i] = 0
        # print(self.list_cls_record)
        # input()
        self.turn_results += turn_list_result

    # def add_turn_results_cls(self, token_predictions, cls_predictions, span_predictions, sampled_batch, verbose=False):
    #     # print(sampled_batch)
    #     turn_list_result = []
    #     # print(len(cls_predictions))
    #     for index, batch_cls_predictions in enumerate(cls_predictions):
    #         dialog_contents = sampled_batch['dialog_content'] # length B
    #         # print(sampled_batch['str_ds_pair'])
    #         str_ds_pairs = sampled_batch['str_ds_pair'][index] # length B
    #         input_ids = sampled_batch['input']['input_ids'] # length B
    #         value_labels = sampled_batch['labels'][index] # length B
    #         cls_labels = sampled_batch['cls_labels'][:, index] # B x 1
    #         # span_labels = sampled_batch['span_labels'][:, index] # B x 2
    #
    #         if index == 0:
    #             turn_list_result = [1] * cls_labels.shape[0] # length B
    #
    #         for i in range(len(dialog_contents)):
    #             # each item in the batch
    #             dialog_content = dialog_contents[i]
    #             str_ds_pair = str_ds_pairs[i]
    #             input_id_sequence = input_ids[i][0]
    #             value_label = value_labels[i]
    #             cls_label = int(cls_labels[i].cpu().numpy())
    #             cls_prediction = batch_cls_predictions[i]
    #
    #             flag = True
    #
    #             '''
    #             Classification
    #             '''
    #             if verbose:
    #                 print(index, i, 'label {} predict: {}'.format(cls_label, cls_prediction))
    #             if cls_prediction == cls_label:
    #                 self.list_cls_record[str_ds_pair].append(1)
    #                 if cls_label != 0: # Has value
    #                     # print('cls_label is', cls_label)
    #                     self.list_notnone_cls_record[str_ds_pair].append(1)
    #             else:
    #                 flag = False
    #                 self.list_cls_record[str_ds_pair].append(0)
    #                 if cls_label != 0: # Has value
    #                     # print('cls_label is', cls_label)
    #                     self.list_notnone_cls_record[str_ds_pair].append(0)
    #
    #             if turn_list_result[i] == 1 and flag == False:
    #                 turn_list_result[i] = 0
    #
    #     self.turn_results += turn_list_result

    def add_turn_results_cls(self, token_predictions, cls_predictions, span_predictions, sampled_batch, verbose=False):
        # print(sampled_batch)
        turn_list_result = []
        # print(len(cls_predictions))
        for index, batch_cls_predictions in enumerate(cls_predictions):
            dialog_contents = sampled_batch['context'] # length B
            # print(sampled_batch['str_ds_pair'])
            # str_ds_pairs = sampled_batch['str_ds_pair'][index] # length B
            input_ids = sampled_batch['input_ids'] # length B
            # value_labels = sampled_batch['labels'][index] # length B
            cls_labels = torch.LongTensor(sampled_batch['cls_labels'])[:, index] # B x 1
            # span_labels = sampled_batch['span_labels'][:, index] # B x 2
            # print(batch_cls_predictions)
            if index == 0:
                turn_list_result = [1] * cls_labels.shape[0] # length B

            for i in range(len(turn_list_result)):
                # each item in the batch
                dialog_content = dialog_contents[i]
                # str_ds_pair = str_ds_pairs[i]
                # input_id_sequence = input_ids[i][0]
                # value_label = value_labels[i]
                cls_label = int(cls_labels[i].cpu().numpy())
                cls_prediction = batch_cls_predictions[i]
                str_ds_pair = self.ds_list[index]
                flag = True

                '''
                Classification
                '''
                if verbose:
                    print(index, i, 'label {} predict: {}'.format(cls_label, cls_prediction))
                if cls_prediction == cls_label:
                    self.list_cls_record[str_ds_pair].append(1)
                    if cls_label != 0: # Has value
                        # print('cls_label is', cls_label)
                        self.list_notnone_cls_record[str_ds_pair].append(1)
                else:
                    flag = False
                    self.list_cls_record[str_ds_pair].append(0)
                    if cls_label != 0: # Has value
                        # print('cls_label is', cls_label)
                        self.list_notnone_cls_record[str_ds_pair].append(0)

                if turn_list_result[i] == 1 and flag == False:
                    turn_list_result[i] = 0

        self.turn_results += turn_list_result


    def add_turn_results_gen(self, gen_predictions, sampled_batch, verbose=False):
        # print(gen_predictions.shape)
        for index, gen_prediction in enumerate(gen_predictions):
            # ignore_len = sampled_batch['ignore_len'][index]
            decoded_str = self.tokenizer.decode(gen_prediction)

            label_ids = sampled_batch['label_ids'][index].detach().cpu().clone()
            pad_id = self.tokenizer.convert_tokens_to_ids('<PAD>')
            label_ids[label_ids==-100] = pad_id
            if verbose:
                print('predicted:', decoded_str)
                print('label:', self.tokenizer.decode(label_ids))


    def add_turn_results_gen_test(self, ootput_ids, sample_batched, attentions=None):
        bos_id, eos_id, pad_id, sep_id = self.tokenizer.convert_tokens_to_ids(
            ['<BOS>', '<EOS>', '<PAD>', '<SEP>'])
        gen = self.tokenizer.decode(ootput_ids)  # include context fed into model
        # print(gen)
        bs_ref = sample_batched['bs_dict']
        assert len(bs_ref) == 1  # batch_size==1
        bs_ref = bs_ref[0]
        bs_ref = self.convert_bar_to_space(bs_ref)

        try:
            bs_gen = self.tokenizer.decode(
                ootput_ids[ootput_ids.index(bos_id) + 1:ootput_ids.index(eos_id)]
            )
            # print(bs_gen)
        except Exception as e:
            # can not find proper <BOS> <EOS>
            # print(gen)
            print('======= NOT FOUND =======')
            self.turn_results.append(0)
            for str_ds_pair in self.ds_list:
                self.list_cls_record[str_ds_pair].append(0)
                if str_ds_pair in bs_ref.keys():
                    if bs_ref[str_ds_pair] not in ['', 'none']:
                        self.list_notnone_cls_record[str_ds_pair].append(0)
            return
        # print(gen)
        print(bs_gen)
        # input()
        bs_dict, bs_str = self.gen2bs_dict(bs_gen, sample_batched['example_id'][0])
        scores = self.score_fn(sample_batched, [bs_dict], [bs_str])
        print('=============')
        print(bs_ref)
        bs_dict = self.convert_bar_to_space(bs_dict)
        print(bs_dict)
        for str_ds_pair in self.ds_list:
            if str_ds_pair in bs_ref.keys():
                # label is not None
                label = bs_ref[str_ds_pair]
                if str_ds_pair in bs_dict.keys():
                    # Predicted this ds
                    pred_value = bs_dict[str_ds_pair]
                    if pred_value == label:
                        print(str_ds_pair, '->', pred_value, '==', label)
                        self.list_cls_record[str_ds_pair].append(1)
                        self.list_notnone_cls_record[str_ds_pair].append(1)
                    else:
                        print(str_ds_pair, '->', pred_value, '!=', label)
                        self.list_cls_record[str_ds_pair].append(0)
                        self.list_notnone_cls_record[str_ds_pair].append(0)
                else:
                    print(str_ds_pair, '->', 'not found')
                    # Did not find this ds
                    self.list_cls_record[str_ds_pair].append(0)
                    self.list_notnone_cls_record[str_ds_pair].append(0)
            else:
                # label is none
                if str_ds_pair in bs_dict.keys():
                    # wrongly find this ds
                    self.list_cls_record[str_ds_pair].append(0)
                    print(str_ds_pair, '->', 'wrongly foound')
                else:
                    # pred value is none as well
                    self.list_cls_record[str_ds_pair].append(1)
                    print(str_ds_pair, '->', 'both none')

        print(scores)
        scores['attentions'] = attentions if attentions else []
        self.results.append(scores)
        self.turn_results.append(scores['acc-joint'])


    def get_metrics(self):
        metrics = {}
        for str_ds_pair in self.ds_list:
            list_token_record = np.array(self.list_token_record[str_ds_pair])
            list_cls_record = np.array(self.list_cls_record[str_ds_pair])
            list_notnone_cls_record = np.array(self.list_notnone_cls_record[str_ds_pair])
            list_span_record = np.array(self.list_span_record[str_ds_pair])
            # try:
            #     token_acc = np.count_nonzero(list_token_record) / len(list_token_record)
            # except:
            #     token_acc = 0
            try:
                cls_acc = np.count_nonzero(list_cls_record) / len(list_cls_record)
            except:
                cls_acc = 0
            try:
                notnone_cls_acc = np.count_nonzero(list_notnone_cls_record) / len(list_notnone_cls_record)
            except:
                notnone_cls_acc = 0
            try:
                span_acc = np.count_nonzero(list_span_record) / len(list_span_record)
            except:
                span_acc = 0
            entry = {
                # 'token_acc': round(token_acc, 3),
                'cls_acc': round(cls_acc, 6),
                'notnone_cls_acc': round(notnone_cls_acc, 6),
                'span_acc': round(span_acc, 6),
            }
            metrics[str_ds_pair] = entry
        # print(self.turn_results)
        turn_results = np.array(self.turn_results)
        joint_acc = np.mean(turn_results)
        # Get slot acc
        cls_record = np.array([self.list_cls_record[str_ds_pair] for str_ds_pair in self.ds_list])
        # print(cls_record)
        slot_acc = np.mean(cls_record)

        metrics['general'] = {
            'joint_acc': round(joint_acc, 6),
            'slot_acc': round(slot_acc, 6)
        }
        return metrics

    def gen2bs_dict(self, gen, example_id):
        '''
        extract predicted belief state from the whole generation output
        '''
        # gen = gen.split()
        # bs = gen[gen.index('<BOS>') + 1: gen.index('<EOS>')]  # [d1, s1, v1, <SEP>, d2, s2, v2]
        bs_str = " ".join(gen.split())
        bs = bs_str.split(" <SEP> ")  # ["d1 s1 v1", "d2 s2 v2"]
        bs_dict = {}
        for idx, dsv in enumerate(bs):
            dsv = dsv.split()
            try:
                d, s, v = dsv[0], dsv[1], " ".join(dsv[2:])
                bs_dict["{}-{}".format(d, s)] = v
            except:
                print('Wrong format in example: {} has dsv={}'.format(example_id, dsv))
        return bs_dict, bs_str

    def convert_bar_to_space(self, bs_ref):
        temp_bs_ref = {}
        for ds, v in bs_ref.items():
            temp_bs_ref[ds.lower().replace('-', ' ')] = v
        return temp_bs_ref

    def score_fn(self, batch, bs_pred, bs_pred_str):
        '''
        bs_pred: list (len=1) of dict of {d-s:v}, e.g., {'hotel-area': north}
        '''
        bs_ref = batch['bs_dict']
        # print('score_fn read', bs_ref)
        assert len(bs_pred) == len(bs_ref) == 1  # batch_size==1
        bs_pred, bs_ref = bs_pred[0], bs_ref[0]
        bs_ref = self.convert_bar_to_space(bs_ref)
        bs_pred = self.convert_bar_to_space(bs_pred)
        slot_acc, joint_acc = compute_dst_acc(bs_ref, bs_pred)

        ex = {'turn_input': batch['turn_utt'][0],
              'bs_pred_dict': bs_pred,
              'bs_ref_dict': bs_ref,
              'bs_pred': " | ".join(dict2list(bs_pred)),
              'bs_ref': " | ".join(dict2list(bs_ref)),
              'example_id': batch['example_id'][0],
              'ori bs_pred': bs_pred_str[0],
              'acc-joint': joint_acc,
              'acc-slot': slot_acc
              }
        return ex

    def save_results(self, file_name):
        result_path = os.path.join(self.config.results_path, file_name)
        with open(result_path, 'w') as result_file:
            json.dump(self.results, result_file, indent=4)
            print('result has been saved to', result_path)