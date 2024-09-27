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
from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
import copy
import json
from io import open
from sklearn.metrics import roc_auc_score
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from tqdm import tqdm

# import sys
# import csv
# csv.field_size_limit(sys.maxsize)

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, text_b=None, label=None, lang=None, line_data=None, topk=10, visual_feature=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.lang = lang
        self.line_data = line_data
        self.topk = topk
        self.visual_feature = visual_feature

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None, offset_mapping=None,
                 word2token=None, visual_features=None, logits=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.offset_mapping = offset_mapping
        self.word2token = word2token
        self.visual_features = visual_features
        self.logits = logits

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, args):
        self.args = args

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors

        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir, file_name):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir, file_name):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, label_file=None):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def tfds_map(self, example):
        """Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are. 
        This method converts examples to the correct format."""
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf8') as f:
            data = f.readlines()
            lines = []
            for line in data:
                line = line.replace("\0", '').rstrip()
                line = line.split("\t")
                lines.append(line)
            return lines

    def ParseLine(self, line):
        raise NotImplementedError()


def logging_set(log_path="./log.txt", level=logging.DEBUG):
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(filename)s %(funcName)s %(lineno)d: %(message)s',
        filename=log_path,
        datefmt='%y-%m-%d %H:%M:%S',
        level=logging.DEBUG if log_path else level,
        filemode='w'
    )

    if log_path:
        console = logging.StreamHandler()
        console.setLevel(level)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(filename)s %(funcName)s %(lineno)d: %(message)s')

        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)


def file_iter(corpus_path, encoding, world_size=1, global_rank=-1):
    """ support lazy load reading

    :param corpus_path:
    :param encoding:
    :param world_size:
    :param global_rank:
    :return:  [lines]
    """
    device_rank = 0 if global_rank == -1 else global_rank
    with open(corpus_path, "r", encoding=encoding) as f:
        for index, line in enumerate(f):
            if index % 100000 == 0 and iter == 0:
                logger.info("process %d lines data." % index)
            line = line.rstrip()
            if not line:
                continue
            if index % world_size == device_rank:
                # logger.debug('line index: %d' % index)
                yield line


def overlap_func(line, overlap_size, max_seq_length):
    line_dict = json.loads(line.rstrip())
    offsets = [n * (max_seq_length - 2 - overlap_size) for n in range(10) if n * (max_seq_length - 2 - overlap_size) < len(line_dict['input_ids'])]
    result_line = []
    for index, start_offset in enumerate(offsets):
        dict_tmp = {"input_ids": line_dict["input_ids"][start_offset: start_offset + max_seq_length - 2],
                    "attention_mask": line_dict["attention_mask"][start_offset: start_offset + max_seq_length - 2],
                    "label": line_dict["label"][start_offset: start_offset + max_seq_length - 2],
                    "visual_features": line_dict["visual_features"][start_offset: start_offset + max_seq_length - 2]}
        # add cls and sep #
        len_of_visual = len(dict_tmp["visual_features"][0])
        if index != 0:
            for _ in range(len(dict_tmp["visual_features"])):
                dict_tmp["visual_features"][_][-1] = index
        dict_tmp["input_ids"] = [0] + dict_tmp["input_ids"] + [2]
        dict_tmp["attention_mask"] = [1] + dict_tmp["attention_mask"] + [1]
        dict_tmp["label"] = [-100] + dict_tmp["label"] + [-100]
        dict_tmp["visual_features"] = [[0 for i in range(len_of_visual)]] + dict_tmp["visual_features"] + [[0 for i in range(len_of_visual)]]
        padding_length = max_seq_length - len(dict_tmp["input_ids"])
        if padding_length > 0:
            dict_tmp["input_ids"] = dict_tmp["input_ids"] + [1] * padding_length
            dict_tmp["attention_mask"] = dict_tmp["attention_mask"] + [0] * padding_length
            dict_tmp["label"] = dict_tmp["label"] + [-100] * padding_length
            dict_tmp["visual_features"] = dict_tmp["visual_features"] + [[0 for i in range(len_of_visual)]] * padding_length
        assert len(dict_tmp["input_ids"]) == max_seq_length
        assert len(dict_tmp["attention_mask"]) == max_seq_length
        assert len(dict_tmp["label"]) == max_seq_length
        assert len(dict_tmp["visual_features"]) == max_seq_length
        # result_line.append(json.dumps(dict_tmp))
        result_line.append(dict_tmp)
    return result_line


def file_iter_multi_task_for_overlap(corpus_path, encoding, world_size=1, global_rank=-1, endless=True, overlap_size=50, max_seq_length=512):
    """ support lazy load reading

    :param corpus_path:
    :param encoding:
    :param world_size:
    :param global_rank:
    :return:  [lines]
    """
    device_rank = 0 if global_rank == -1 else global_rank
    offset = 0
    global_index = 0
    while True:
        with open(corpus_path, "r", encoding=encoding) as f:
            for index, line in enumerate(f):
                if index % 100000 == 0 and iter == 0:
                    logger.info("process %d lines data." % index)
                line = line.rstrip()
                if not line:
                    continue
                overlap_lines = overlap_func(line, overlap_size, max_seq_length)
                for overlap_line in overlap_lines:
                    if (global_index + offset) % world_size == device_rank:
                        # logger.debug('line index: %d' % index)
                        yield overlap_line
                    global_index += 1
        if not endless:
            break
        else:
            # switch data partition among different devices
            offset += 1


def file_iter_multi_task(corpus_path, encoding, world_size=1, global_rank=-1, endless=True):
    """ support lazy load reading

    :param corpus_path:
    :param encoding:
    :param world_size:
    :param global_rank:
    :return:  [lines]
    """
    device_rank = 0 if global_rank == -1 else global_rank
    offset = 0
    global_index = 0
    while True:
        with open(corpus_path, "r", encoding=encoding) as f:
            for index, line in enumerate(f):
                if index % 100000 == 0 and iter == 0:
                    logger.info("process %d lines data." % index)
                line = line.rstrip()
                if not line:
                    continue
                if (global_index + offset) % world_size == device_rank:
                    # logger.debug('line index: %d' % index)
                    yield line
                global_index += 1
        if not endless:
            break
        else:
            # switch data partition among different devices
            offset += 1
            # logger.info('Global rank: %d, iterate corpus for the %d time, corpus_path: %s' % (global_rank, offset + 1, corpus_path))


def kd_loss(logits_teacher, logits_student, mask, temperature):
    logits_teacher_temp = logits_teacher / temperature
    logits_student_temp = logits_student / temperature

    logits_teacher_soft = F.softmax(logits_teacher_temp, dim=-1)
    logits_student_soft = F.softmax(logits_student_temp, dim=-1)

    loss = -(logits_teacher_soft * torch.log(logits_student_soft + 1e-9))
    loss = torch.mul(loss.sum(dim=-1), mask).mean()

    return loss

