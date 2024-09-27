# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" GLUE processors and helpers """
import transformers
import logging
import os
from io import open
import numpy as np
from torch.utils.data import Dataset
import torch
from transformer_utils.data_utils import DataProcessor, InputExample, InputFeatures, file_iter, file_iter_multi_task, file_iter_multi_task_for_overlap
from transformer_utils.tasks.keyphrase import KeyPhraseProcessor, KeyPhraseProcessor2, \
    key_phrase_convert_example_to_features
from transformer_utils.tasks.xml_ee import XMLEEProcessor, XMLEEProcessor2, XMLEEProcessor3, xmlee_phrase_convert_examples_to_features, \
    xmlee_phrase_convert_examples_to_features2, xmlee_phrase_convert_examples_to_features3
import json
from transformers.file_utils import is_tf_available
from transformer_utils.tasks.highlight import *
########
# from transformer_lite.transformer_utils.tasks.highlight import *
# from transformer_lite.transformer_utils.data_utils import DataProcessor, InputExample, InputFeatures,file_iter,file_iter_multi_task
# from transformer_lite.transformer_utils.tasks.keyphrase import KeyPhraseProcessor,KeyPhraseProcessor2, key_phrase_convert_example_to_features
# from transformer_lite.transformer_utils.tasks.xml_ee import XMLEEProcessor,XMLEEProcessor2,xmlee_phrase_convert_examples_to_features,xmlee_phrase_convert_examples_to_features2,xmlee_phrase_convert_examples_to_features3
########
from typing import Any, Dict, List, NewType, Tuple
import collections
from copy import deepcopy
from random import random, randrange, randint, shuffle, choice

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


class QuantusProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir, file_name):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, file_name)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "train")

    def get_dev_examples(self, data_dir, file_name):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "dev")

    def get_predict_examples(self, input_file):
        """See base class."""
        return self._create_examples(
            self._read_tsv(input_file), "predict")

    def get_labels(self, label_file=None):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[self.args.query_index]
            text_b = line[self.args.passage_index]
            if set_type == "predict":
                label = None
                line_data = '\t'.join(line)
            else:
                label = line[self.args.label_index]
                line_data = None

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, line_data=line_data))
        return examples

    def parse_line(self, line, set_type="train"):
        line_dict = dict()

        split_line = line.split("\t")
        text_a = '. '.join([split_line[int(i)] for i in self.args.query_index.split(",")])
        text_a = text_a.replace(" |||", ".")
        text_b = split_line[self.args.passage_index] if self.args.passage_index != -1 else None
        lang = split_line[self.args.language_index] if self.args.language_index != -1 else None
        data = text_a
        guid = hashlib.md5(data.encode(encoding='UTF-8')).hexdigest()

        line_dict["guid"] = guid
        line_dict["text_a"] = text_a
        line_dict["text_b"] = text_b
        line_dict["lang"] = lang

        if set_type != "predict":
            label = split_line[self.args.label_index]
            line_dict["label"] = label
        return line_dict


class MLMProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir, file_name):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, file_name)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "train")

    def get_dev_examples(self, data_dir, file_name):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "dev")

    def get_predict_examples(self, input_file):
        """See base class."""
        return self._create_examples(
            self._read_tsv(input_file), "predict")

    def get_labels(self, label_file=None):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[self.args.query_index]
            text_b = line[self.args.passage_index]
            if set_type == "predict":
                label = None
                line_data = '\t'.join(line)
            else:
                label = line[self.args.label_index]
                line_data = None

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, line_data=line_data))
        return examples

    def parse_line(self, line, set_type="train"):
        line_dict = dict()

        split_line = line.split("\t")
        text_a = split_line[self.args.query_index]
        text_b = split_line[self.args.passage_index]
        data = text_a + text_b
        guid = hashlib.md5(data.encode(encoding='UTF-8')).hexdigest()

        line_dict["guid"] = guid
        line_dict["text_a"] = text_a
        line_dict["text_b"] = text_b

        if set_type != "predict":
            label = split_line[self.args.label_index]
            line_dict["label"] = label
        return line_dict


class MLMQPProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir, file_name):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, file_name)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "train")

    def get_dev_examples(self, data_dir, file_name):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "dev")

    def get_predict_examples(self, input_file):
        """See base class."""
        return self._create_examples(
            self._read_tsv(input_file), "predict")

    def get_labels(self, label_file=None):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[self.args.query_index]
            text_b = line[self.args.passage_index]
            if set_type == "predict":
                label = None
                line_data = '\t'.join(line)
            else:
                label = line[self.args.label_index]
                line_data = None

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, line_data=line_data))
        return examples

    def parse_line(self, line, set_type="train"):
        line_dict = dict()

        split_line = line.split("\t")
        text_a = split_line[self.args.query_index]
        text_b = split_line[self.args.passage_index]
        data = text_a + text_b
        guid = hashlib.md5(data.encode(encoding='UTF-8')).hexdigest()

        line_dict["guid"] = guid
        line_dict["text_a"] = text_a
        line_dict["text_b"] = text_b

        return line_dict


class QuantusKDProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir, file_name):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, file_name)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "train")

    def get_dev_examples(self, data_dir, file_name):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "dev")

    def get_predict_examples(self, input_file):
        """See base class."""
        return self._create_examples(
            self._read_tsv(input_file), "predict")

    def get_labels(self, label_file=None):
        """See base class."""
        return ["0"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[self.args.query_index]
            text_b = line[self.args.passage_index]
            if set_type == "predict":
                label = None
                line_data = '\t'.join(line)
            else:
                label = line[self.args.label_index]
                line_data = None

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, line_data=line_data))
        return examples

    def parse_line(self, line, set_type="train"):
        line_dict = dict()

        split_line = line.split("\t")
        text_a = split_line[self.args.query_index]
        text_b = split_line[self.args.passage_index]
        data = text_a + text_b
        guid = hashlib.md5(data.encode(encoding='UTF-8')).hexdigest()

        line_dict["guid"] = guid
        line_dict["text_a"] = text_a
        line_dict["text_b"] = text_b

        if set_type != "predict":
            label = split_line[self.args.label_index]
            line_dict["label"] = label
        return line_dict


class LangProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir, file_name):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, file_name)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "train")

    def get_dev_examples(self, data_dir, file_name):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "dev")

    def get_predict_examples(self, input_file):
        """See base class."""
        return self._create_examples(
            self._read_tsv(input_file), "predict")

    def get_test_examples(self, data_dir, file_name):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "predict")

    def get_labels(self, label_file=None):
        """See base class."""
        return ['en', 'es', 'fr', 'pt', 'de', 'it', 'ca', 'ru', 'nl', 'cs', 'id',
                'pl', 'ar', 'no', 'gl', 'ja', 'ro', 'hu', 'da', 'el', 'fi', 'bg',
                'he', 'eo', 'hr', 'mk', 'sh', 'et', 'fa', 'ko', 'bn', 'bs', 'lt',
                'hi', 'eu', 'mr', 'si', 'oc', 'az', 'be', 'ml', 'is', 'ba', 'nds',
                'ceb', 'ne', 'kk', 'fy', 'an', 'la', 'fo', 'lb', 'bar', 'mwl', 'ka',
                'arz', 'jv', 'hy', 'gom', 'br', 'as', 'nds_nl', 'mg', 'rm', 'scn',
                'io', 'azb', 'lmo', 'vi', 'sv', 'uk', 'zh', 'sr', 'tr', 'sk', 'sl',
                'simple', 'sq', 'te', 'ta', 'tl', 'tt', 'sw', 'wuu', 'tg', 'ug']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[self.args.query_index]
            text_b = line[self.args.passage_index]
            if set_type == "predict":
                label = None
                line_data = '\t'.join(line)
            else:
                label = line[self.args.label_index]
                line_data = None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, line_data=line_data))
        return examples

    def parse_line(self, line, set_type="train"):
        line_dict = dict()
        split_line = line.split("\t")
        text_a = split_line[self.args.query_index]
        guid = hashlib.md5(text_a.encode(encoding='UTF-8')).hexdigest()
        line_dict["guid"] = guid
        line_dict["text_a"] = text_a
        if set_type != "predict":
            label = split_line[self.args.label_index]
            line_dict["label"] = label
        return line_dict


class RankerBERTProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir, file_name):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, file_name)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "train")

    def get_dev_examples(self, data_dir, file_name):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "dev")

    def get_predict_examples(self, input_file):
        """See base class."""
        return self._create_examples(
            self._read_tsv(input_file), "predict")

    def get_test_examples(self, data_dir, file_name):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "predict")

    def get_labels(self, label_file=None):
        """See base class."""
        return ["0", "1", "2", "3", "4"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[self.args.query_index]
            text_b = line[self.args.passage_index]

            if set_type == "predict":
                label = None
                line_data = '\t'.join(line)
            else:
                label = line[self.args.label_index]
                line_data = None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, line_data=line_data))
        return examples

    def parse_line(self, line, set_type="train"):
        line_dict = dict()

        split_line = line.split("\t")
        text_a = split_line[self.args.query_index]
        text_b = split_line[self.args.passage_index]
        data = text_a + text_b
        guid = hashlib.md5(data.encode(encoding='UTF-8')).hexdigest()
        lang = ""
        if self.args.language_index != -1:
            lang = split_line[self.args.language_index]

        line_dict["guid"] = guid
        line_dict["text_a"] = text_a + " " + lang
        line_dict["text_b"] = text_b

        if set_type != "predict":
            label = split_line[self.args.label_index]
            line_dict["label"] = label
        return line_dict


class RankerBERTKDProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir, file_name):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, file_name)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "train")

    def get_dev_examples(self, data_dir, file_name):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "dev")

    def get_predict_examples(self, input_file):
        """See base class."""
        return self._create_examples(
            self._read_tsv(input_file), "predict")

    def get_test_examples(self, data_dir, file_name):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "predict")

    def get_labels(self, label_file=None):
        """See base class."""
        return ["0", "1", "2", "3", "4"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[self.args.query_index]
            text_b = line[self.args.passage_index]
            if set_type == "predict":
                label = None
                line_data = '\t'.join(line)
            else:
                label = json.loads(line[self.args.label_index])
                line_data = None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, line_data=line_data))
        return examples

    def parse_line(self, line, set_type="train"):
        line_dict = dict()

        split_line = line.split("\t")
        text_a = split_line[self.args.query_index]
        text_b = split_line[self.args.passage_index]
        data = text_a + text_b
        guid = hashlib.md5(data.encode(encoding='UTF-8')).hexdigest()

        line_dict["guid"] = guid
        line_dict["text_a"] = text_a
        line_dict["text_b"] = text_b

        if set_type != "predict":
            label = split_line[self.args.label_index]
            line_dict["label"] = label
        return line_dict


class SingleSentenceProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    # def __init__(self, args):
    #     self.labels = ["0", "1"]
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir, file_name):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, file_name)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "train")

    def get_dev_examples(self, data_dir, file_name):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "dev")

    def get_predict_examples(self, input_file):
        """See base class."""
        return self._create_examples(
            self._read_tsv(input_file), "predict")

    def get_test_examples(self, data_dir, file_name):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "predict")

    def get_labels(self, label_file=None):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[self.args.query_index]
            text_b = None
            if set_type == "predict":
                label = None
                line_data = '\t'.join(line)
            else:
                label = line[self.args.label_index]
                line_data = None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, line_data=line_data))
        return examples

    def parse_line(self, line, set_type="train"):
        line_dict = dict()
        split_line = line.split("\t")
        text_a = split_line[self.args.query_index]
        guid = hashlib.md5(text_a.encode(encoding='UTF-8')).hexdigest()
        line_dict["guid"] = guid
        line_dict["text_a"] = text_a
        if set_type != "predict":
            label = split_line[self.args.label_index]
            line_dict["label"] = label
        return line_dict


class SingleSentenceKDProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    # def __init__(self, args):
    #     self.labels = ["0", "1"]
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir, file_name):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, file_name)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "train")

    def get_dev_examples(self, data_dir, file_name):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "dev")

    def get_predict_examples(self, input_file):
        """See base class."""
        return self._create_examples(
            self._read_tsv(input_file), "predict")

    def get_test_examples(self, data_dir, file_name):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "predict")

    def get_labels(self, label_file=None):
        """See base class."""
        return ["0"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[self.args.query_index]
            text_b = None
            if set_type == "predict":
                label = None
                line_data = '\t'.join(line)
            else:
                label = line[self.args.label_index]
                line_data = None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, line_data=line_data))
        return examples

    def parse_line(self, line, set_type="train"):
        line_dict = dict()
        split_line = line.split("\t")
        text_a = split_line[self.args.query_index]
        guid = hashlib.md5(text_a.encode(encoding='UTF-8')).hexdigest()
        line_dict["guid"] = guid
        line_dict["text_a"] = text_a
        if set_type != "predict":
            label = split_line[self.args.label_index]
            line_dict["label"] = label
        return line_dict


class AdultProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir, file_name):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, file_name)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "train")

    def get_dev_examples(self, data_dir, file_name):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "dev")

    def get_predict_examples(self, input_file):
        """See base class."""
        return self._create_examples(
            self._read_tsv(input_file), "predict")

    def get_test_examples(self, data_dir, file_name):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "predict")

    def get_labels(self, label_file=None):
        """See base class."""
        return ["0", "1", "2"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[self.args.query_index]

            text_b = None
            if set_type == "predict":
                label = None
                line_data = '\t'.join(line)
            else:
                label = line[self.args.label_index]
                line_data = None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, line_data=line_data))
        return examples

    def parse_line(self, line, set_type="train"):
        line_dict = dict()
        split_line = line.split("\t")
        text_a = split_line[self.args.query_index]
        text_b = split_line[self.args.passage_index] if self.args.passage_index != -1 else None
        guid = hashlib.md5(text_a.encode(encoding='UTF-8')).hexdigest()
        line_dict["guid"] = guid
        line_dict["text_a"] = text_a
        line_dict["text_b"] = text_b
        if set_type != "predict":
            label = split_line[self.args.label_index]
            line_dict["label"] = label
        return line_dict


class TopicModelProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir, file_name):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, file_name)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "train")

    def get_evaluation_examples(self, input_file):
        return self._create_evaluation_examples(
            self._read_tsv(input_file), "evaluation")

    def get_dev_examples(self, data_dir, file_name):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "dev")

    def get_predict_examples(self, input_file):
        """See base class."""
        return self._create_examples(
            self._read_tsv(input_file), "predict")

    def get_test_examples(self, data_dir, file_name):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "predict")

    def _create_evaluation_examples(self, lines, set_type):
        data_dict = {}
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[self.args.query_index]
            text_b = None
            language = line[self.args.language_index]
            if set_type == "dev" or set_type == "predict":
                label = None
            else:
                label = line[self.args.label_index]

            line_example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            if language not in data_dict:
                data_dict[language] = [line_example]
            else:
                data_dict[language].append(line_example)
        return data_dict

    def get_labels(self, label_file=None):
        """See base class."""
        label_list = list()
        assert label_file != None, "label file is None"
        if label_file != None:
            with open(label_file, 'r', encoding='utf8') as f:
                data = f.readlines()
                for index, line in enumerate(data):
                    line = line.rstrip()
                    split_line = line.split("\t")
                    label_list.append(split_line[0])

        return label_list

    def get_labels_with_des(self, label_file=None, use_topic=True):
        """See base class."""
        label_list = list()
        assert label_file != None, "label file is None"
        if label_file != None:
            with open(label_file, 'r', encoding='utf8') as f:
                data = f.readlines()
                for index, line in enumerate(data):
                    line = line.rstrip()
                    split_line = line.split("\t")
                    if use_topic:
                        label_list.append((split_line[0], split_line[-1]))
                    else:
                        label_list.append(split_line[-1])

        return label_list

    def covert_label2feature(self, topci_list, tokenizer=None):
        return topci_list

    def encode_label_feature(self, label_list, tokenizer, max_len=32):
        feature_list = tokenizer.batch_encode_plus(
            [example for example in label_list],
            max_length=max_len,
            padding="max_length",
            truncation=True,
            pad_to_max_length=True,
        )
        cur_tensor = dict()
        for key, value in feature_list.data.items():
            if key in ("input_ids", "attention_mask", "token_type_ids") and value is not None and not isinstance(value,
                                                                                                                 str):
                cur_tensor[key] = torch.tensor(value, dtype=torch.long)
        return cur_tensor

    def get_demote_topics(self, label_file=None):
        """See base class."""
        label_list = set()
        assert label_file != None, "demote label file is None"
        if label_file != None:
            with open(label_file, 'r', encoding='utf8') as f:
                data = f.readlines()
                for line in data:
                    line = line.strip().strip("\n")
                    label_list.add(line)
        return label_list

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[self.args.query_index]
            text_b = None
            if set_type == "predict":
                label = None
                line_data = '\t'.join(line)
            else:
                label = json.loads(line[self.args.label_index])
                line_data = None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, line_data=line_data))
        return examples

    def parse_line(self, line, set_type="train"):
        line_dict = dict()
        split_line = line.split("\t")
        text_a = split_line[self.args.query_index]
        lang = ""
        if self.args.language_index != -1:
            lang = split_line[self.args.language_index]
        guid = hashlib.md5(text_a.encode(encoding='UTF-8')).hexdigest()
        line_dict["guid"] = guid
        line_dict["text_a"] = text_a + " " + lang
        if set_type != "predict":
            label = split_line[self.args.label_index]
            line_dict["label"] = json.loads(label)
        return line_dict


class UniversalClassificationProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir, file_name):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, file_name)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "train")

    def get_evaluation_examples(self, input_file):
        return self._create_evaluation_examples(
            self._read_tsv(input_file), "evaluation")

    def get_dev_examples(self, data_dir, file_name):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "dev")

    def get_predict_examples(self, input_file):
        """See base class."""
        return self._create_examples(
            self._read_tsv(input_file), "predict")

    def get_test_examples(self, data_dir, file_name):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "predict")

    def _create_evaluation_examples(self, lines, set_type):
        data_dict = {}
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[self.args.query_index]
            text_b = None
            language = line[self.args.language_index]
            if set_type == "dev" or set_type == "predict":
                label = None
            else:
                label = line[self.args.label_index]

            line_example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            if language not in data_dict:
                data_dict[language] = [line_example]
            else:
                data_dict[language].append(line_example)
        return data_dict

    def get_labels(self, label_file=None):
        """See base class."""
        label_list = list()
        assert label_file != None, "label file is None"
        if label_file != None:
            with open(label_file, 'r', encoding='utf8') as f:
                data = f.readlines()
                for index, line in enumerate(data):
                    line = line.rstrip()
                    split_line = line.split("\t")
                    label_list.append(split_line[0])

        return label_list

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[self.args.query_index]
            text_b = None
            if set_type == "predict":
                label = None
                line_data = '\t'.join(line)
            else:
                label = json.loads(line[self.args.label_index])
                label = label[0]
                line_data = None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, line_data=line_data))
        return examples

    def parse_line(self, line, set_type="train"):
        line_dict = dict()
        split_line = line.split("\t")
        text_a = split_line[self.args.query_index]
        lang = ""
        if self.args.language_index != -1:
            lang = split_line[self.args.language_index]
        guid = hashlib.md5(text_a.encode(encoding='UTF-8')).hexdigest()
        line_dict["guid"] = guid
        line_dict["text_a"] = text_a + " " + lang
        if set_type != "predict":
            label = split_line[self.args.label_index]
            label = json.loads(label)
            line_dict["label"] = label[0]
        return line_dict


def glue_convert_examples_to_features(examples, tokenizer,
                                      max_length=512,
                                      task=None,
                                      label_list=None,
                                      label_map=None,
                                      output_mode=None, task_name="quantus", set_type="dev"):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))
    if label_map == None and set_type != "predict":
        label_map = {label: i for i, label in enumerate(label_list)}

    def get_label(example):
        label = None
        if set_type != "predict":
            if output_mode == "classification":
                if "mlm" in task_name:
                    label = None
                else:
                    label = label_map[example.label]
            elif output_mode == "regression":
                if "topic_model" in task_name:
                    label = np.array([0] * len(label_map))
                    label_index = [label_map[item] for item in example.label if item in label_map]
                    label[label_index] = 1
                    if task_name == "topic_model3":
                        label_sum = np.sum(label)
                        if label_sum != 0:
                            label = label / np.sum(label)
                else:
                    if isinstance(example.label, list):
                        label = np.array(example.label)
                    else:
                        label = float(example.label)
            else:
                raise KeyError(output_mode)
        return label

    labels = [get_label(example) for example in examples]
    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        pad_to_max_length=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    return features


def EE_convert_example_to_features(args, examples, set_type="dev"):
    features = []
    for (ex_index, example) in enumerate(examples):
        if args.data_augmentation and args.currency_list is not None and args.currency_ids is not None and random() < args.ratio_currency:
            line_dict_input_ids = example.input_ids
            min_index = 0
            for index_input_id, input_id in enumerate(line_dict_input_ids):
                if index_input_id < min_index:
                    continue
                if args.currency_ids.__contains__(input_id):
                    for currency_ids_vec in args.currency_ids[input_id]:
                        if currency_ids_vec == -1 or currency_ids_vec == line_dict_input_ids[index_input_id + 1: index_input_id + 1 + len(currency_ids_vec)]:
                            old_currency = [input_id] + (currency_ids_vec if currency_ids_vec != -1 else [])
                            new_currency = args.currency_ids[choice(args.new_currency_list)]
                            if old_currency == new_currency:
                                break
                            new_labels = [example.label[index_input_id]] + [-100 for i in range(len(new_currency) - 1)]
                            new_visual_features = [example.visual_features[index_input_id] for i in new_currency]
                            example.input_ids[index_input_id:index_input_id + len(old_currency)] = new_currency
                            example.label[index_input_id:index_input_id + len(old_currency)] = new_labels
                            example.visual_features[index_input_id:index_input_id + len(old_currency)] = new_visual_features
                            min_index = index_input_id + len(new_currency)
                            break
        num_k = abs(args.max_seq_length - len(example.input_ids))
        if len(example.input_ids) > args.max_seq_length:
            example.input_ids = example.input_ids[:args.max_seq_length]
            example.label = example.label[:args.max_seq_length]
            example.visual_features = example.visual_features[:args.max_seq_length]
            example.attention_mask = ([1 for _ in range(num_k)] + example.attention_mask)[:args.max_seq_length]
        elif len(example.input_ids) < args.max_seq_length:
            example.input_ids = example.input_ids + [example.input_ids[-1] for _ in range(num_k)]
            example.visual_features = example.visual_features + [[0 for tk in range(22)] for _ in range(num_k)]
            example.label = example.label + [-100 for _ in range(num_k)]
            example.attention_mask = (example.attention_mask[num_k:] + [0 for _ in range(3 * num_k)])[:args.max_seq_length]
        assert len(example.input_ids) == args.max_seq_length
        assert len(example.visual_features) == args.max_seq_length
        assert len(example.attention_mask) == args.max_seq_length
        assert len(example.label) == args.max_seq_length
        cur_features = InputFeatures(
            input_ids=example.input_ids, 
            attention_mask=example.attention_mask,
            visual_features=example.visual_features,
            label=example.label)
        features.append(cur_features)
    return features


def topic_convert_example_to_features(example,
                                      tokenizer,
                                      max_length=512,
                                      task=None,
                                      label_list=None,
                                      label_map=None,
                                      output_mode=None, task_name="quantus", set_type="dev"):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(example, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))
    if label_map == None:
        label_map = {label: i for i, label in enumerate(label_list)}

    if is_tf_dataset:
        example = processor.get_example_from_tensor_dict(example)
        example = processor.tfds_map(example)

    # inputs = tokenizer.encode_plus(
    #     example.text_a, example.text_b, add_special_tokens=True, max_length=max_length, return_token_type_ids=True,
    # )
    inputs = tokenizer.encode_plus(
        example.text_a, example.text_b,
        max_length=max_length,
        truncation=True,
        pad_to_max_length=True,
        padding='max_length'
    )

    label = None
    if set_type != "predict":
        if output_mode == "classification":
            if "mlm" in task_name:
                label = None
            else:
                label = label_map[example.label]
        elif output_mode == "regression":
            if "topic_model" in task_name or task_name == "topic_matching":
                label = np.array([0] * len(label_map))
                label_index = [label_map[item] for item in example.label if item in label_map]
                if task_name == "topic_matching":
                    label_index = label_index[:8]
                label[label_index] = 1
                if task_name == "topic_model3":
                    label_sum = np.sum(label)
                    if label_sum != 0:
                        label = label / np.sum(label)
            else:
                if isinstance(example.label, list):
                    label = np.array(example.label)
                else:
                    label = float(example.label)
        else:
            raise KeyError(output_mode)

        # if set_type != "dev":
        #     if example.guid < 5:
        #         logger.info("*** Example ***")
        #         logger.info("guid: %s" % (example.guid))
        #         logger.info("text_a: %s"% (example.text_a))
        #         logger.info("text_b: %s" % (example.text_b))
        #         # logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #         # logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        #         # logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
        #         logger.info("label: %s" % (str(example.label)))
        #         logger.info("converted_label: %s"%(str(label)))

    feature = InputFeatures(**inputs,
                            label=label)

    return feature


class BERTDataset(Dataset):
    def __init__(self, args, corpus_path, tokenizer, seq_len, label_list, lazy_load_processor, task_name,
                 encoding='utf8', corpus_lines=None, on_memory=True, world_size=1, global_rank=0,
                 lazy_load_block_size=1000000):
        self.args = args
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.label_list = label_list
        self.lazy_load_processor = lazy_load_processor
        self.task_name = task_name
        self.world_size = world_size
        self.global_rank = global_rank
        self.output_mode = glue_output_modes[task_name]
        if 'mlm_probability' in self.args:
            self.mlm_probability = self.args.mlm_probability
        else:
            self.mlm_probability = 0

        if "random_mask" in self.args:
            self.random_mask = self.args.random_mask
        else:
            self.random_mask = False

        # used to keep track of full epochs on file
        self.sample_counter = 0

        # for lazy load
        self.lazy_load_block_size = lazy_load_block_size
        self.item_idx = self.lazy_load_block_size

        self.label_map = {label: i for i, label in enumerate(self.label_list)}

        if not self.corpus_lines:
            self.corpus_lines = sum(1 for _ in open(corpus_path, 'r', encoding=encoding) if len(_.rstrip()) > 0)
        # if self.args.multi_task:
        if self.args.overlap is False:
            self.fin_corpus = file_iter_multi_task(self.corpus_path, self.encoding, world_size=self.world_size,
                                                   global_rank=self.global_rank, endless=True)
        else:
            self.fin_corpus = file_iter_multi_task_for_overlap(self.corpus_path, self.encoding, world_size=self.world_size, global_rank=self.global_rank, endless=True,
                                                               overlap_size=self.args.overlap_size, max_seq_length=self.args.max_seq_length)
        # else:
        #     self.fin_corpus = file_iter(self.corpus_path, self.encoding, world_size=self.world_size,
        #                                            global_rank=self.global_rank)

        logger.info('Complete corpus lines: %d' % self.corpus_lines)

    def get_real_len(self):
        return self.corpus_lines

    def __len__(self):
        logger.info(
            'corpus_lines:%d; world_size:%d; global_rank:%d' % (max(self.corpus_lines, self.lazy_load_block_size), self.world_size, self.global_rank))
        return (
                           self.corpus_lines // self.world_size) + 1 if self.corpus_lines % self.world_size > self.global_rank else (
                    self.corpus_lines // self.world_size)

    def __getitem__(self, item):

        if self.item_idx == self.lazy_load_block_size:
            self.all_docs_cache = []
            for i in range(self.lazy_load_block_size):
                try:
                    self.all_docs_cache.append(next(self.fin_corpus))
                except:
                    continue
            self.shuffle_indices = np.random.permutation(len(self.all_docs_cache))
            self.item_idx = 0
        if self.args.overlap:
            line = self.all_docs_cache[self.shuffle_indices[self.item_idx]]
        else:
            line = self.all_docs_cache[self.shuffle_indices[self.item_idx]].replace("\0", "").rstrip()
        # print(self.item_idx)
        self.item_idx += 1
        try:
            if self.args.overlap:
                line_dict = line
            else:
                line_dict = self.lazy_load_processor(line)

            if "mrc" in self.task_name:
                cur_example = HighlightExample(**line_dict)
                # TODO: the answer maybe not valid or larger than
                cur_features_list = highlight_convert_examples_to_features([cur_example],
                                                                           tokenizer=self.tokenizer,
                                                                           max_seq_length=self.args.max_seq_length,
                                                                           doc_stride=self.args.doc_stride,
                                                                           max_query_length=self.args.max_query_length,
                                                                           is_training=True,
                                                                           return_dataset=False, threads=1)

                self.shuffle_feature_indicate = np.random.randint(0, len(cur_features_list))
                cur_features = cur_features_list[self.shuffle_feature_indicate]
            elif "key_phrase" in self.task_name:
                if self.task_name == "key_phrase2":
                    # length = len(line_dict["input_ids"])
                    # print(length)
                    assert len(line_dict["input_ids"]) == self.args.max_seq_length
                    cur_features = InputFeatures(**line_dict)
                else:
                    cur_example = InputExample(**line_dict)
                    cur_features = key_phrase_convert_example_to_features(cur_example,
                                                                          self.tokenizer,
                                                                          max_length=self.args.max_seq_length,
                                                                          task=None,
                                                                          label_list=self.label_list,
                                                                          label_map=self.label_map,
                                                                          output_mode=self.output_mode,
                                                                          task_name=self.task_name, set_type="train")
            elif "xml_ee" in self.task_name:
                if self.task_name == "xml_ee2":
                    cur_example = InputExample(**line_dict)
                    cur_features = xmlee_phrase_convert_examples_to_features(cur_example,
                                                                             self.tokenizer,
                                                                             max_length=self.args.max_seq_length,
                                                                             task=None,
                                                                             label_list=self.label_list,
                                                                             label_map=self.label_map,
                                                                             output_mode=self.output_mode,
                                                                             task_name=self.task_name, set_type="train")

                elif self.task_name == "xml_ee3":
                    cur_example = InputExample(**line_dict)
                    cur_features = xmlee_phrase_convert_examples_to_features3(cur_example,
                                                                              self.tokenizer,
                                                                              max_length=self.args.max_seq_length,
                                                                              task=None,
                                                                              label_list=self.label_list,
                                                                              label_map=self.label_map,
                                                                              output_mode=self.output_mode,
                                                                              task_name=self.task_name,
                                                                              set_type="train")
                elif self.task_name == "xml_ee_features":
                    assert len(line_dict["input_ids"]) == self.args.max_seq_length
                    cur_features = InputFeatures(**line_dict)
                elif self.task_name == "xml_ee_features_logits":
                    if self.args.data_augmentation and self.args.currency_list is not None and self.args.currency_ids is not None and random() < self.args.ratio_currency:
                        line_dict_input_ids = line_dict["input_ids"]
                        min_index = 0
                        for index_input_id, input_id in enumerate(line_dict_input_ids):
                            if index_input_id < min_index:
                                continue
                            if self.args.currency_ids.__contains__(input_id):
                                for currency_ids_vec in self.args.currency_ids[input_id]:
                                    if currency_ids_vec == -1 or currency_ids_vec == line_dict_input_ids[index_input_id + 1: index_input_id + 1 + len(currency_ids_vec)]:
                                        old_currency = [input_id] + (currency_ids_vec if currency_ids_vec != -1 else [])
                                        new_currency = self.args.currency_ids[choice(self.args.new_currency_list)]
                                        if old_currency == new_currency:
                                            break
                                        new_labels = [line_dict["label"][index_input_id]] + [-100 for i in range(len(new_currency) - 1)]
                                        new_visual_features = [line_dict["visual_features"][index_input_id] for i in new_currency]
                                        line_dict["input_ids"][index_input_id:index_input_id + len(old_currency)] = new_currency
                                        line_dict["label"][index_input_id:index_input_id + len(old_currency)] = new_labels
                                        line_dict["visual_features"][index_input_id:index_input_id + len(old_currency)] = new_visual_features
                                        min_index = index_input_id + len(new_currency)
                                        break
                    num_k = abs(self.args.max_seq_length - len(line_dict["input_ids"]))
                    if len(line_dict["input_ids"]) > self.args.max_seq_length:
                        line_dict["input_ids"] = line_dict["input_ids"][:self.args.max_seq_length]
                        line_dict["label"] = line_dict["label"][:self.args.max_seq_length]
                        line_dict["visual_features"] = line_dict["visual_features"][:self.args.max_seq_length]
                        line_dict["attention_mask"] = ([1 for _ in range(num_k)] + line_dict["attention_mask"])[:self.args.max_seq_length]
                    elif len(line_dict["input_ids"]) < self.args.max_seq_length:
                        line_dict["input_ids"] = line_dict["input_ids"] + [line_dict["input_ids"][-1] for _ in range(num_k)]
                        line_dict["visual_features"] = line_dict["visual_features"] + [[0 for tk in range(22)] for _ in range(num_k)]
                        line_dict["label"] = line_dict["label"] + [-100 for _ in range(num_k)]
                        line_dict["attention_mask"] = (line_dict["attention_mask"][num_k:] + [0 for _ in range(3 * num_k)])[:self.args.max_seq_length]
                    assert len(line_dict["input_ids"]) == self.args.max_seq_length
                    cur_features = InputFeatures(**line_dict)
                elif "xml_ee_binary_classification" in self.task_name:
                    if self.args.data_augmentation and self.args.currency_list is not None and self.args.currency_ids is not None and random() < self.args.ratio_currency:
                        line_dict_input_ids = line_dict["input_ids"]
                        min_index = 0
                        for index_input_id, input_id in enumerate(line_dict_input_ids):
                            if index_input_id < min_index:
                                continue
                            if self.args.currency_ids.__contains__(input_id):
                                for currency_ids_vec in self.args.currency_ids[input_id]:
                                    if currency_ids_vec == -1 or currency_ids_vec == line_dict_input_ids[index_input_id + 1: index_input_id + 1 + len(currency_ids_vec)]:
                                        old_currency = [input_id] + (currency_ids_vec if currency_ids_vec != -1 else [])
                                        new_currency = self.args.currency_ids[choice(self.args.new_currency_list)]
                                        if old_currency == new_currency:
                                            break
                                        new_labels = [line_dict["label"][index_input_id]] + [-100 for i in range(len(new_currency) - 1)]
                                        new_visual_features = [line_dict["visual_features"][index_input_id] for i in new_currency]
                                        line_dict["input_ids"][index_input_id:index_input_id + len(old_currency)] = new_currency
                                        line_dict["label"][index_input_id:index_input_id + len(old_currency)] = new_labels
                                        line_dict["visual_features"][index_input_id:index_input_id + len(old_currency)] = new_visual_features
                                        min_index = index_input_id + len(new_currency)
                                        break
                    num_k = abs(self.args.max_seq_length - len(line_dict["input_ids"]))
                    if len(line_dict["input_ids"]) > self.args.max_seq_length:
                        line_dict["input_ids"] = line_dict["input_ids"][:self.args.max_seq_length]
                        line_dict["label"] = line_dict["label"][:self.args.max_seq_length]
                        line_dict["visual_features"] = line_dict["visual_features"][:self.args.max_seq_length]
                        line_dict["attention_mask"] = ([1 for _ in range(num_k)] + line_dict["attention_mask"])[:self.args.max_seq_length]
                    elif len(line_dict["input_ids"]) < self.args.max_seq_length:
                        line_dict["input_ids"] = line_dict["input_ids"] + [line_dict["input_ids"][-1] for _ in range(num_k)]
                        line_dict["visual_features"] = line_dict["visual_features"] + [[0 for tk in range(22)] for _ in range(num_k)]
                        line_dict["label"] = line_dict["label"] + [-100 for _ in range(num_k)]
                        line_dict["attention_mask"] = (line_dict["attention_mask"][num_k:] + [0 for _ in range(3 * num_k)])[:self.args.max_seq_length]
                    assert len(line_dict["input_ids"]) == self.args.max_seq_length
                    cur_features = InputFeatures(**line_dict)
                else:
                    cur_example = InputExample(**line_dict)
                    cur_features = xmlee_phrase_convert_examples_to_features(self.args, cur_example,
                                                                             self.tokenizer,
                                                                             max_length=self.args.max_seq_length,
                                                                             task=None,
                                                                             label_list=self.label_list,
                                                                             label_map=self.label_map,
                                                                             output_mode=self.output_mode,
                                                                             task_name=self.task_name, set_type="train")
            else:
                cur_example = InputExample(**line_dict)
                cur_example.guid = self.sample_counter
                cur_features = topic_convert_example_to_features(cur_example, self.tokenizer,
                                                                 max_length=self.args.max_seq_length,
                                                                 task=None,
                                                                 label_list=self.label_list,
                                                                 label_map=self.label_map,
                                                                 output_mode=self.output_mode,
                                                                 task_name=self.task_name, set_type="train")
        except Exception as e:
            print(e)
            return self.__getitem__(item)

        cur_tensor = self.convert_feature_to_tensor(cur_features)
        self.sample_counter += 1
        return cur_tensor

    def convert_feature_to_tensor(self, cur_feature):
        cur_tensor = dict()
        for key, value in vars(cur_feature).items():
            if key in ("input_ids", "attention_mask", "token_type_ids") and value is not None and not isinstance(value, str):
                cur_tensor[key] = torch.tensor(value, dtype=torch.long)
            elif key in ('visual_features') and value is not None:
                cur_tensor[key] = torch.tensor(value, dtype=torch.long)
            elif key in ('logits') and value is not None:
                cur_tensor[key] = torch.tensor(value, dtype=torch.float)
        if self.output_mode == "classification":
            if "mrc" in self.task_name:
                cur_tensor["labels"] = (torch.tensor(cur_feature.start_position, dtype=torch.long),
                                        torch.tensor(cur_feature.end_position, dtype=torch.long))
            elif "keyphrase_mlm" in self.task_name:
                inputs, labels = self.mask_one_example_keyphrase_tokens(cur_tensor, mask_label=cur_feature.label)
                cur_tensor = {"input_ids": inputs,
                              "labels": labels}
            elif "mlm" in self.task_name:
                if self.args.whole_word_mask:
                    inputs, labels = self.whole_word_mask_tokens(
                        cur_feature.input_ids)
                else:
                    inputs, labels = self.mask_one_example_tokens(cur_tensor)
                cur_tensor = {"input_ids": inputs, "labels": labels}
            else:
                cur_tensor["labels"] = torch.tensor(cur_feature.label, dtype=torch.long)
        else:
            cur_tensor["labels"] = torch.tensor(cur_feature.label, dtype=torch.float)

        if self.random_mask:
            inputs = self.random_mask_inputs(cur_tensor)
            cur_tensor["input_ids"] = inputs
        return cur_tensor

    def random_mask_inputs(self, tensor_dict: dict) -> torch.Tensor:
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        inputs = tensor_dict["input_ids"]

        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(inputs.shape, self.mlm_probability)

        special_tokens_mask = self.tokenizer.get_special_tokens_mask(inputs.tolist(), already_has_special_tokens=True)
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        if self.tokenizer._pad_token is not None:
            padding_mask = inputs.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        return inputs

    def mask_one_example_keyphrase_tokens(self, tensor_dict: dict, mask_label) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = tensor_dict["input_ids"]
        labels = inputs.clone()

        masked_indices = torch.Tensor(mask_label).bool()
        # print(masked_indices)
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # print(labels)
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        # new_tensor_dict["input_ids"] = inputs
        # new_tensor_dict["labels"] = labels
        return inputs, labels

    def mask_one_example_tokens(self, tensor_dict: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        inputs = tensor_dict["input_ids"]
        labels = inputs.clone()

        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # print(p)
        # print(labels.tolist())

        special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True)
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        # new_tensor_dict["input_ids"] = inputs
        # new_tensor_dict["labels"] = labels
        return inputs, labels

    def whole_word_mask_tokens(self, input_ids):  ##list input list output
        token_starts = ["##", "▁"]
        MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                                  ["index", "label"])
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        cand_indices = []
        for (i, token) in enumerate(tokens):
            if token == self.tokenizer.cls_token or token == self.tokenizer.sep_token:
                continue
            # Whole Word Masking means that if we mask all of the wordpieces
            # corresponding to an original word. When a word has been split into
            # WordPieces, the first token does not have any marker and any subsequence
            # tokens are prefixed with ##. So whenever we see the ## token, we
            # append it to the previous set of word indexes.
            #
            # Note that Whole Word Masking does *not* change the training code
            # at all -- we still predict each WordPiece independently, softmaxed
            # over the entire vocabulary.
            if 'xlmroberta' in self.args.model_type:  ##for xlmr model
                if len(cand_indices) >= 1 and not token.startswith(token_starts[1]):
                    cand_indices[-1].append(i)
                else:
                    cand_indices.append([i])
            else:  ##for bert or others
                if len(cand_indices) >= 1 and token.startswith(token_starts[0]):
                    cand_indices[-1].append(i)
                else:
                    cand_indices.append([i])

        num_to_mask = max(1, int(round(len(tokens) * self.args.mlm_probability)))
        shuffle(cand_indices)
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indices:
            if len(masked_lms) >= num_to_mask:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_mask:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)

                masked_token = None
                # 80% of the time, replace with [MASK]
                if random() < 0.8:
                    # masked_token = "[MASK]"
                    masked_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
                else:
                    # 10% of the time, keep original
                    if random() < 0.5:
                        masked_token = self.tokenizer.convert_tokens_to_ids(tokens[index])
                    # 10% of the time, replace with random word
                    else:
                        masked_token = randint(0, len(self.tokenizer))
                masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
                # tokens[index] = self.tokenizer.convert_ids_to_tokens(masked_token) ## has bug for mask id to token
                input_ids[index] = masked_token

        assert len(masked_lms) <= num_to_mask
        masked_lms = sorted(masked_lms, key=lambda x: x.index)
        mask_indices = [p.index for p in masked_lms]
        masked_token_labels = self.tokenizer.convert_tokens_to_ids([p.label for p in masked_lms])
        mask_labels = deepcopy(input_ids)
        for index in range(len(mask_indices)):
            mask_labels[mask_indices[index]] = masked_token_labels[index]
        for index in range(len(input_ids)):
            if index in mask_indices:
                continue
            mask_labels[index] = -100
        if len(set(mask_labels)) == 1 and list(set(mask_labels))[0] == -1:
            mask_labels[0] = 1  # avoid labels with all -1
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(mask_labels, dtype=torch.long)


glue_processors = {
    "cola": transformers.data.processors.glue.ColaProcessor,
    "mnli": transformers.data.processors.glue.ColaProcessor,
    "mnli-mm": transformers.data.processors.glue.ColaProcessor,
    "mrpc": transformers.data.processors.glue.ColaProcessor,
    "sst-2": transformers.data.processors.glue.ColaProcessor,
    "sts-b": transformers.data.processors.glue.ColaProcessor,
    "qqp": transformers.data.processors.glue.ColaProcessor,
    "qnli": transformers.data.processors.glue.ColaProcessor,
    "rte": transformers.data.processors.glue.ColaProcessor,
    "wnli": transformers.data.processors.glue.WnliProcessor,
    "quantus": QuantusProcessor,
    "quantus_single_score": QuantusProcessor,
    "quantuskd": QuantusKDProcessor,
    "rankerbert": RankerBERTProcessor,
    "rankerbertkd": RankerBERTKDProcessor,
    "topic_model": TopicModelProcessor,
    "topic_matching": TopicModelProcessor,
    "topic_model1": TopicModelProcessor,
    "topic_model2": TopicModelProcessor,
    "topic_model3": TopicModelProcessor,
    "singlesentence": SingleSentenceProcessor,
    "singlesentencekd": SingleSentenceKDProcessor,
    "quantus_mrc": HighlightProcessor,  # for KD learning with label data
    "language": LangProcessor,
    "mlm": MLMProcessor,
    "mlmqp": MLMQPProcessor,
    "adult": AdultProcessor,
    "key_phrase": KeyPhraseProcessor,
    "key_phrase_mlm": KeyPhraseProcessor,
    "key_phrase2": KeyPhraseProcessor2,
    "universal": UniversalClassificationProcessor,
    "xml_ee": XMLEEProcessor,
    "xml_ee2": XMLEEProcessor,
    "xml_ee_features": XMLEEProcessor2,
    "xml_ee_features_logits": XMLEEProcessor2,
    "xml_ee3": XMLEEProcessor,
    "xml_ee4_visual": XMLEEProcessor,
    "xml_ee_binary_classification":XMLEEProcessor3,
    "xml_ee_multiclass_classification":XMLEEProcessor2,
}

glue_output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "quantus": "classification",  # temp fix
    "quantus_single_score": "classification",
    "quantuskd": "regression",
    "quantus_mrc": "classification",  # for mrc task
    "rankerbert": "classification",
    "rankerbertkd": "regression",
    "topic_model": "regression",
    "topic_model1": "regression",
    "topic_model2": "regression",
    "topic_model3": "regression",
    "topic_matching": "regression",
    "singlesentence": "classification",
    "singlesentencekd": "regression",
    "language": "classification",
    "mlm": "classification",
    "mlmqp": "classification",
    "adult": "classification",
    "key_phrase": "classification",
    "key_phrase_mlm": "classification",
    "key_phrase2": "classification",
    "universal": "classification",
    "xml_ee": "classification",
    "xml_ee2": "classification",
    "xml_ee3": "classification",
    "xml_ee4_visual": "classification",
    "xml_ee_features": "classification",
    "xml_ee_features_logits": "classification",
    "xml_ee_binary_classification": "classification",
    "xml_ee_multiclass_classification": "classification"
}
glue_tasks_num_labels = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "quantus": 2,
    "quantus_single_score": 2,
    "quantuskd": 1,
    "rankerbert": 5,
    "rankerbertkd": 5,
    "singlesentence": 2,
    "singlesentencekd": 1,
    "quantus_mrc": 2,
    "language": 100,
    "adult": 3,
    "key_phrase": 5,
    "key_phrase_mlm": 5,
    "key_phrase2": 5,
    "xml_ee": 5,
    "xml_ee2": 5,
    "xml_ee3": 5,
    "xml_ee4_visual": 5,
    "xml_ee_features": 5
}
