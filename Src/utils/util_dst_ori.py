import os
import sys
import json
import operator

cwd = os.getcwd()
sys.path.insert(0, cwd + '/utils/')
from utils.fix_label import fix_general_label_error

''' This file contains functions for processing the provided DST data and scoring accuracy '''

# global variable
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

# NOTE: if you change the relative path of slot_list.json, modify the path below
with open('../Data/MultiWOZ/dst/slot_list.json') as f:
    all_slots = json.load(f)


def dict2list(bs_dict):
    ''' convert belief state dictionary into list of tokens of domain-slot=value '''
    l = [s + '=' + v for s, v in bs_dict.items()]
    return sorted(l)


def compute_dst_acc(gold, pred):
    '''
    Score function for computing joint / slot accuracy given a predicted belief state
    Args:
        gold: ground-truth of belief state
        pred: prediction of belief state
        Both are dictionary of {domain-slot:value}, e.g., {'hotel-area': north}
    '''
    gold, pred = set(dict2list(gold)), set(dict2list(pred))

    # compute joint acc
    joint_acc = 1 if pred == gold else 0

    # compute slot acc
    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.split('=')[0])

    wrong_pred = 0
    for p in pred:
        if p not in gold and p.split('=')[0] not in miss_slot:
            wrong_pred += 1
    slot_acc = (len(all_slots) - miss_gold - wrong_pred) / len(all_slots)
    return slot_acc, joint_acc


def convert_slot(slot):
    if 'book' in slot:
        return slot.replace('book ', '')
    if slot == 'arriveby':
        return 'arriveBy'
    if slot == 'leaveat':
        return 'leaveAt'
    return slot


def allign_slot_name(bs_dict):
    new_bs_dict = {}
    for domain_slot, value in bs_dict.items():
        domain, slot = domain_slot.split('-')
        new_slot = convert_slot(slot)
        new_domain_slot = '{}-{}'.format(domain, new_slot)
        new_bs_dict[new_domain_slot] = value
    return new_bs_dict


def fix_wrong_domain_label(turn_belief_dict, domains, dial_idx, turn_idx):
    ''' remove potential slot-value pairs in wrong domain '''
    remove_keys = []
    for domain_slot in turn_belief_dict:
        domain = domain_slot.split('-')[0]
        if domain not in domains:
            remove_keys.append(domain_slot)

    for key in remove_keys:
        del turn_belief_dict[key]
    return turn_belief_dict


def iterate_dst_file(file_path):
    '''
    This function iterates a given dst file and returns the processed data examples with fixed labels
    Args:
        file_path: data/MultiWOZ/dst/{train/dev/test}_v2.{0/1}.json
    Returns:
        dictionary of examples, where each example contains:
            1) concatenation of utterances of both system and user sides until the current turn
            2) turn level utterances
            3) belief state annotations, a dictionary of {domain-slot: value}, e.g., {'hotel-area': 'north'}
    '''
    with open(file_path) as f:
        dst_data = json.load(f)

    data_container = {}
    for dial_dict in dst_data:
        dial_name = dial_dict['dialogue_idx']
        domains = dial_dict["domains"]
        if 'hospital' in domains or 'police' in domains:  # ignore dialogues in hospital or police domain
            continue

        # process dialogue
        history = ""  # dialogue history
        for turn_idx, turn in enumerate(dial_dict["dialogue"]):
            # get turn level utterance
            usr_utterance = turn['transcript'].strip()
            sys_utterance = turn['system_transcript'].strip()

            # form input sequence to GPT2 with a concatenation of utterances from both system and user sides
            # the utterance are splitted using special tokens <USR> & <SYS>
            # NOTE: feel free to change the format of the input sequence to your model
            if sys_utterance == "":
                turn_utterance = '<USR> ' + usr_utterance + " "
            else:
                turn_utterance = "<SYS> " + sys_utterance + ' <USR> ' + usr_utterance + " "
            history += turn_utterance

            # process belief state label into a dictionary, where keys are domain-slot and values are corresponding value, e.g., {'hotel-area': 'north'}
            bs_dict = turn["belief_state"]
            bs_dict = fix_general_label_error(bs_dict, False, all_slots)  # fix general errors in dataset
            bs_dict = fix_wrong_domain_label(bs_dict, dial_dict['domains'], dial_name,
                                             turn_idx)  # fix wrong domain label
            bs_dict = allign_slot_name(bs_dict)  # allign slot names

            # collect example
            example_idx = "{}-{}".format(dial_name.replace('.json', ''), turn_idx)
            data_container[example_idx] = {}
            data_container[example_idx]['context'] = history  # str
            data_container[example_idx]['turn_utt'] = turn_utterance  # str
            data_container[example_idx]['belief_state'] = bs_dict

    dial_n, example_n = len(dst_data), len(data_container)
    return data_container, dial_n, example_n


if __name__ == '__main__':
    pass
