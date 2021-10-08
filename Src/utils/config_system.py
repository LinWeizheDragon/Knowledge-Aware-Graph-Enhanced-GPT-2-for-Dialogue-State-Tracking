import os
import shutil
import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler

import json
import _jsonnet
import datetime
import time
from easydict import EasyDict
from pprint import pprint
import time
from utils.dirs import create_dirs
from pathlib import Path

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    # parse the configurations from the config json file provided

    try:
        config_dict = json.loads(_jsonnet.evaluate_file(json_file))
        # EasyDict allows to access dict values as attributes (works recursively).
        config = EasyDict(config_dict)
        return config, config_dict
    except ValueError:
        print("INVALID JSON file.. Please provide a good json file")
        exit(-1)

def process_config(args):
    script_dir = os.path.dirname(os.path.realpath('__file__'))
    path = Path(script_dir).parent
    config, _ = get_config_from_json(args.config)

    # Some default paths
    if not config.DATA_FOLDER:
        # Default path
        config.DATA_FOLDER = os.path.join(str(path), 'Data')
    if not config.EXPERIMENT_FOLDER:
        # Default path
        config.EXPERIMENT_FOLDER = os.path.join(str(path), 'Experiments')
    if not config.TENSORBOARD_FOLDER:
        # Default path
        config.TENSORBOARD_FOLDER = os.path.join(str(path), 'Data_TB')

    # Override using passed parameters
    config.cuda = not args.disable_cuda
    if args.device != -1:
        config.gpu_device = args.device
    config.reset = args.reset
    config.mode = args.mode
    if args.experiment_name != '':
        config.experiment_name = args.experiment_name
    config.model_config.graph_model.num_layer = args.num_layer
    config.model_config.graph_model.num_head = args.num_head
    config.model_config.graph_model.num_hop = args.num_hop
    config.model_config.graph_mode = args.graph_mode
    
    # Override using args for only last turn...
    config.data_loader.additional.only_last_turn = args.only_last_turn

    config.data_loader.dummy_dataloader = args.dummy_dataloader
    config.fp16 = args.fp16
    config.test.plot_img = not args.test_disable_plot_img
    if args.test_num_evaluation != -1:
        config.test.num_evaluation = args.test_num_evaluation
    if args.test_batch_size != -1:
        config.test.batch_size = args.test_batch_size
    if args.test_evaluation_name:
        config.test.evaluation_name = args.test_evaluation_name
    config.test_output_attention = args.test_output_attention
    if args.test_num_processes != -1:
        config.test.additional.multiprocessing = args.test_num_processes

    if config.mode == "train":
        config.train.load_best_model = args.load_best_model
        config.train.load_model_path = args.load_model_path
        config.train.load_epoch = args.load_epoch
    else:
        config.test.load_best_model = args.load_best_model
        config.test.load_model_path = args.load_model_path
        config.test.load_epoch = args.load_epoch

    # Generated Paths
    config.log_path = os.path.join(config.EXPERIMENT_FOLDER, config.experiment_name, config.mode)
    config.experiment_path = os.path.join(config.EXPERIMENT_FOLDER, config.experiment_name)
    config.saved_model_path = os.path.join(config.EXPERIMENT_FOLDER, config.experiment_name, "train", 'saved_model')
    if config.mode == "train":
        config.imgs_path = os.path.join(config.EXPERIMENT_FOLDER, config.experiment_name, "train", 'imgs')
    else:
        config.imgs_path = os.path.join(config.EXPERIMENT_FOLDER, config.experiment_name, "test",
                                        config.test.evaluation_name, 'imgs')
        config.results_path = os.path.join(config.EXPERIMENT_FOLDER, config.experiment_name, "test",
                                        config.test.evaluation_name)
    config.tensorboard_path = os.path.join(config.TENSORBOARD_FOLDER, config.experiment_name)

    return config



