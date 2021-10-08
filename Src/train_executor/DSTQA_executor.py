"""
DSTQA_executor.py: Training code of DSTQA:
Li Zhou, Kevin Small. Multi-domain Dialogue State Tracking as Dynamic Knowledge Graph Enhanced Question Answering. In NeurIPS 2019 Workshop on Conversational AI
Code modified from:
https://github.com/alexa/dstqa
By authors of the paper:
Lin, W., Tseng, B. H., & Byrne, B. (2021). Knowledge-Aware Graph-Enhanced GPT-2 for Dialogue State Tracking. EMNLP 2021.
https://arxiv.org/abs/2104.04466v3
"""

import os
import scipy
import numpy as np
import matplotlib.pyplot as pyplot
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
from allennlp.commands.train import train_model_from_file, train_model
import tempfile
from models.dstqa.dstqa_reader import DSTQAReader
from models.dstqa.dstqa import DSTQA
from models.dstqa.dstqa_predictor import DSTQAPredictor
from allennlp.common import Params
from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.training.callbacks.events import Events
from copy import deepcopy

class DSTQAExecutor(BaseExecutor):
    def __init__(self, config, data_loader):
        BaseExecutor.__init__(self, config, data_loader)


    def train(self):
        #############################################
        #
        #                load setup
        #
        #############################################

        batch_size = self.config.train.batch_size
        save_interval = self.config.train.save_interval
        device = self.config.device

        start_time = datetime.datetime.now()
        logdir = os.path.join(self.config.tensorboard_path)  # datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        logger.print(logdir)

        ADDED_GRAPH = True
        if self.config.reset:
            ADDED_GRAPH = False

        # if self.loaded_epoch == 0:
        #     ADDED_GRAPH = False

        writer = SummaryWriter(logdir)

        allennlp_config = deepcopy(self.config.Allennlp_config)

        if self.config.data_loader.dummy_dataloader:
            # Load dummy dataset for fast development
            allennlp_config.train_data_path = self.config.data_loader.additional.dummy_train_data_path
            allennlp_config.validation_data_path = self.config.data_loader.additional.dummy_validation_data_path
            allennlp_config.model.elmo_embedding_path = self.config.data_loader.additional.dummy_elmo_embedding_path

        # pass_config = deepcopy(self.config)
        # pass_config.pop('device')
        #
        # allennlp_config.trainer.callbacks[-1].config = pass_config

        temp_config_path = os.path.join(self.config.experiment_path, "train", 'tmp_train_config.json')
        with open(temp_config_path, 'w') as config_file:
            json.dump(allennlp_config, config_file, indent=4)

        params = Params.from_file(temp_config_path)

        # train_model_from_file(temp_config_path,
        #                       self.config.saved_model_path,
        #                       file_friendly_logging=True,
        #                       force=True)


        recover = False
        force = True
        if self.config.train.load_best_model:
            # Continue training
            recover = True
            force = False

        train_model(params,
                    serialization_dir=self.config.saved_model_path,
                    file_friendly_logging=True,
                    recover=recover,
                    force=force,
                    cache_directory=None, cache_prefix=None)


@Callback.register("my_callback")
class CallbackSystem(Callback):
    def __init__(self, config=None):
        print('callback init')
        self.config = config
        self.model = None

    @handle_event(Events.TRAINING_START)
    def training_start(self, trainer) -> None:
        logger.print('this is the start of training.')
        self.model = trainer.model

    def _save_checkpoint(self, epoch, record_best_model=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # 'scheduler_state_dict': self.scheduler.state_dict(),
        }
        if record_best_model:
            file_name = "model_best.pth.tar"
            path_save_model = os.path.join(self.config.saved_model_path, file_name)
            torch.save(state, path_save_model)
            logger.print('Model Saved:', path_save_model)
        else:
            file_name = "model_{}.pth.tar".format(epoch)
            path_save_model = os.path.join(self.config.saved_model_path, file_name)
            # Save the state
            torch.save(state, path_save_model)
            logger.print('Model Saved:', path_save_model)

            file_name = "model_lastest.pth.tar".format(epoch)
            path_save_model = os.path.join(self.config.saved_model_path, file_name)
            # Save the state
            torch.save(state, path_save_model)
            logger.print('Lastest Model Saved:', path_save_model)

    def _load_checkpoint_model(self, load_epoch=-1, load_best_model=False, load_model_path=""):
        if load_model_path:
            path_save_model = load_model_path
        else:
            if load_best_model:
                file_name = "model_best.pth.tar"
            else:
                if load_epoch == -1:
                    file_name = "model_lastest.pth.tar"
                else:
                    file_name = "model_{}.pth.tar".format(load_epoch)

            path_save_model = os.path.join(self.config.saved_model_path, file_name)

        try:
            logger.print("Loading checkpoint '{}'".format(path_save_model))
            # checkpoint = torch.load(filename)
            checkpoint = torch.load(path_save_model, map_location='cuda:{}'.format(self.config.gpu_device))
            self.loaded_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(path_save_model, checkpoint['epoch']))

        except OSError as e:
            self.loaded_epoch = 0
            print(e)
            print("No checkpoint exists from '{}'. Skipping...".format(path_save_model))
            print("**First time to train**")
