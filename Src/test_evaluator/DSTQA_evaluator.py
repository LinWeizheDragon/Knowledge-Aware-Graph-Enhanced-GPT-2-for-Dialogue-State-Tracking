"""
DSTQA_evaluator.py: Testing code of DSTQA:
Li Zhou, Kevin Small. Multi-domain Dialogue State Tracking as Dynamic Knowledge Graph Enhanced Question Answering. In NeurIPS 2019 Workshop on Conversational AI
Code modified from:
https://github.com/alexa/dstqa
By authors of the paper:
Lin, W., Tseng, B. H., & Byrne, B. (2021). Knowledge-Aware Graph-Enhanced GPT-2 for Dialogue State Tracking. EMNLP 2021.
https://arxiv.org/abs/2104.04466v3
"""

__author__ = "Weizhe Lin"
__copyright__ = "Copyright 2021, Weizhe Lin"
__version__ = "1.0.0"
__email__ = "wl356@cam.ac.uk"
__status__ = "Published for Github"

import torch
import numpy as np
import random
import os
import shutil
import pickle


import argparse
import sys
import json
from easydict import EasyDict
from argparse import Namespace
from typing import List, Iterator, Optional
from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import lazy_groups_of
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor, JsonDict
from allennlp.data import Instance

# Customize
from pprint import pprint
from utils.log_system import logger
from models.dstqa.results_analyzer import DSTQAAnalyser
from test_evaluator.base_evaluator import BaseEvaluator

class DSTQAEvaluator(BaseEvaluator):
    def __init__(self, config, data_loader):
        BaseEvaluator.__init__(self, config, data_loader)
        self.config = config

        if config.test.seed:
            random.seed(config.test.seed)
            torch.random.manual_seed(config.test.seed)
            torch.cuda.manual_seed(config.test.seed)
            logger.print("SEED is set to:", config.test.seed)




    def _evaluate_DST_benchmark(self):
        '''
        This function evaluates the model using DST benchmark
        Test on case-by-case test sets
        :return:
        '''

        output_file = os.path.join(self.config.results_path, "all_predictions.pkl")
        if os.path.exists(output_file):
            with open(output_file, 'rb') as f:
                logger.print("Predictions available. Loading results from", output_file)
                results = pickle.load(f)
        else:
            logger.print("No predictions available. Start predicting on test set")
            if self.config.test.load_best_model:
                weights_file_path = os.path.join(self.config.saved_model_path,
                                                 'best.th')
            else:
                if self.config.test.load_epoch > 0:
                    weights_file_path = os.path.join(self.config.saved_model_path,
                                                     'model_state_epoch_{}.th'.format(str(self.config.test.load_epoch)))
                else:
                    weights_file_path = None

            archive_file_path = os.path.join(self.config.saved_model_path,
                                             'model.tar.gz')
            # input(weights_file_path)
            args = Namespace(weights_file=weights_file_path,
                             archive_file=archive_file_path,
                             output_file=None,
                             input_file=self.config.test.additional.input_file_path,
                             cuda_device=self.config.cuda_device,
                             predictor='dstqa',
                             batch_size=self.config.test.batch_size,
                             silent=False,
                             use_dataset_reader=False,
                             overrides='',
                             dataset_reader_choice="validation",
                             )

            check_for_gpu(args.cuda_device)
            archive = load_archive(args.archive_file,
                                   weights_file=args.weights_file,
                                   cuda_device=args.cuda_device,
                                   overrides=args.overrides)

            archive.config['model']['elmo_embedding_path'] = self.config.data_loader.additional.elmo_embedding_path
            archive.config['train_data_path'] = self.config.data_loader.additional.train_data_path
            archive.config['validation_data_path'] = self.config.data_loader.additional.validation_data_path
            archive.model.init_with_config(self.config, load_data=True)

            predictor = Predictor.from_archive(archive, args.predictor,
                                               dataset_reader_to_load=args.dataset_reader_choice)

            predictor.init_with_config(self.config)

            if args.silent and not args.output_file:
                print("--silent specified without --output-file.")
                print("Exiting early because no output will be created.")
                sys.exit(0)

            manager = _PredictManager(self.config,
                                      predictor,
                                      args.input_file,
                                      args.output_file,
                                      args.batch_size,
                                      not args.silent,
                                      args.use_dataset_reader)
            results = manager.run()

            with open(output_file, 'wb') as outfile:
                pickle.dump(results, outfile)
            logger.print('all predictions are successfully saved to', output_file)

        # More analyses
        analyzer = DSTQAAnalyser(self.config, results)
        analyzer.get_DST_benchmark()
        analysis_file = os.path.join(self.config.results_path, "analysis.json")
        with open(analysis_file, 'w') as outfile:
            json.dump(analyzer.DST_results, outfile)
        logger.print("successfully finished analysis. Results saved to", analysis_file)


    def evaluate(self):
        self._evaluate_DST_benchmark()
        # self._evaluate_full_benchmark()


class _PredictManager:

    def __init__(self,
                 config,
                 predictor: Predictor,
                 input_file: str,
                 output_file: Optional[str],
                 batch_size: int,
                 print_to_console: bool,
                 has_dataset_reader: bool
                 ) -> None:
        self.config = config
        self._predictor = predictor
        self._input_file = input_file
        if output_file is not None:
            self._output_file = open(output_file, "w")
        else:
            self._output_file = None
        self._batch_size = batch_size
        self._print_to_console = print_to_console
        if has_dataset_reader:
            self._dataset_reader = predictor._dataset_reader  # pylint: disable=protected-access
        else:
            self._dataset_reader = None

        self.results = []

    def _predict_json(self, batch_data: List[JsonDict]) -> Iterator[str]:
        if len(batch_data) == 1:
            results = [self._predictor.predict_json(batch_data[0])]
        else:
            results = self._predictor.predict_batch_json(batch_data)
        for output in results:
            yield self._predictor.dump_line(output)

    def _predict_instances(self, batch_data: List[Instance]) -> Iterator[str]:
        if len(batch_data) == 1:
            results = [self._predictor.predict_instance(batch_data[0])]
        else:
            results = self._predictor.predict_batch_instance(batch_data)
        for output in results:
            yield self._predictor.dump_line(output)

    def _maybe_print_to_console_and_file(self,
                                         index: int,
                                         prediction: str,
                                         model_input: str = None) -> None:
        if self._print_to_console:
            if model_input is not None:
                print(f"input {index}: ", model_input)
            print("prediction: ", prediction)
        if self._output_file is not None:
            self._output_file.write(prediction)

    def _get_json_data(self) -> Iterator[JsonDict]:
        if self._input_file == "-":
            for line in sys.stdin:
                if not line.isspace():
                    yield self._predictor.load_line(line)
        else:
            input_file = cached_path(self._input_file)
            with open(input_file, "r") as file_input:
                for line in file_input:
                    if not line.isspace():
                        yield self._predictor.load_line(line)

    def _get_instance_data(self) -> Iterator[Instance]:
        if self._input_file == "-":
            raise ConfigurationError("stdin is not an option when using a DatasetReader.")
        elif self._dataset_reader is None:
            raise ConfigurationError("To generate instances directly, pass a DatasetReader.")
        else:
            yield from self._dataset_reader.read(self._input_file)

    def add_result(self,
                 index: int,
                 prediction: str,
                 model_input: str = None) -> None:
        if model_input:
            self.results.append([index, model_input, prediction])
        else:
            self.results.append([index, prediction])


    def run(self):
        has_reader = self._dataset_reader is not None
        index = 0
        if has_reader:
            for batch in lazy_groups_of(self._get_instance_data(), self._batch_size):
                for model_input_instance, result in zip(batch, self._predict_instances(batch)):
                    self._maybe_print_to_console_and_file(index, result, str(model_input_instance))
                    self.add_result(index, result, str(model_input_instance))
                    index = index + 1
        else:
            for batch_json in lazy_groups_of(self._get_json_data(), self._batch_size):
                for model_input_json, result in zip(batch_json, self._predict_json(batch_json)):
                    self._maybe_print_to_console_and_file(index, result, json.dumps(model_input_json))
                    self.add_result(index, result, json.dumps(model_input_json))
                    index = index + 1

        if self._output_file is not None:
            self._output_file.close()

        return self.results
