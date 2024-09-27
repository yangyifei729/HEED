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
from transformer_utils.data_utils import DataProcessor, InputExample, InputFeatures,file_iter,file_iter_multi_task
from transformer_utils.tasks.highlight import *
from typing import Any, Dict, List, NewType, Tuple

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)

class GenerationProcessor(DataProcessor):
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
    def parse_line(self, line, set_type = "train"):
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






def generate_convert_examples_to_features(examples, tokenizer,
                                      max_source_length=512,
                                      max_target_length=32,
                                      task=None,
                                      label_list=None,
                                      output_mode=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True, task_name="quantus", set_type = "dev"):
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
        processor = generate_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = generate_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))


    features = []
    for (ex_index, example) in enumerate(examples):
        if set_type != "predict":
            if ex_index % 10000 == 0:
                logger.info("Writing example %d" % (ex_index))
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)


        cur_feature = generate_convert_example_to_features(example, tokenizer,
                                           max_source_length=max_source_length,
                                           max_target_length=max_target_length,
                                           task=None,
                                         label_list=label_list,
                                         output_mode=output_mode,
                                         task_name=task_name, set_type=set_type)

        features.append(cur_feature)
    return features


def generate_convert_example_to_features(example, tokenizer,
                                     max_source_length=512,
                                     max_target_length=32,
                                     task=None,
                                      label_list=None,
                                      output_mode=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True, task_name="quantus", set_type = "dev"):
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
        processor = generate_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = generate_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))


    if is_tf_dataset:
        example = processor.get_example_from_tensor_dict(example)
        example = processor.tfds_map(example)


    inputs_a = tokenizer.encode_plus(
        example.text_a,
        max_length=max_source_length,
        pad_to_max_length=True,)
    inputs_b = tokenizer.encode_plus(
        example.text_b,
        max_length=max_target_length,
        pad_to_max_length=True,)


    feature = (InputFeatures(**inputs_a), InputFeatures(**inputs_b))


    return feature



class BERTDataset(Dataset):
    def __init__(self, args, corpus_path, tokenizer, max_source_length, max_target_length, label_list, lazy_load_processor, task_name, encoding='utf8', corpus_lines=None, on_memory=True, world_size=1, global_rank=0, lazy_load_block_size=1000000):
        self.args = args
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.label_list = label_list
        self.lazy_load_processor = lazy_load_processor
        self.task_name = task_name
        self.world_size = world_size
        self.global_rank = global_rank
        self.output_mode = generate_output_modes[task_name]
        if 'mlm_probability' in self.args:
            self.mlm_probability = self.args.mlm_probability
        else:
            self.mlm_probability = 0

        # used to keep track of full epochs on file
        self.sample_counter = 0

        # for lazy load
        self.lazy_load_block_size = lazy_load_block_size
        self.item_idx = self.lazy_load_block_size

        if not self.corpus_lines:
            self.corpus_lines = sum(1 for _ in open(corpus_path, 'r', encoding=encoding) if len(_.rstrip()) > 0)
        # if self.args.multi_task:
        self.fin_corpus = file_iter_multi_task(self.corpus_path, self.encoding, world_size=self.world_size,
                                               global_rank=self.global_rank, endless=True)

        logger.info('Complete corpus lines: %d' % self.corpus_lines)

    def get_real_len(self):
        return self.corpus_lines

    def __len__(self):
        logger.info('corpus_lines:%d; world_size:%d; global_rank:%d' % (self.corpus_lines, self.world_size, self.global_rank))
        return (self.corpus_lines // self.world_size) + 1 if self.corpus_lines % self.world_size > self.global_rank else (self.corpus_lines // self.world_size)

    def __getitem__(self, item):

        if self.item_idx == self.lazy_load_block_size:
            self.all_docs_cache = []
            for i in range(self.lazy_load_block_size):
                self.all_docs_cache.append(next(self.fin_corpus))
            self.shuffle_indices = np.random.permutation(len(self.all_docs_cache))
            self.item_idx = 0

        line = self.all_docs_cache[self.shuffle_indices[self.item_idx]].replace("\0", "").rstrip()

        self.item_idx += 1
        line_dict = self.lazy_load_processor(line)
        cur_example = InputExample(**line_dict)
        cur_example.guid = self.sample_counter
        try:
            (feature_a, feature_b) = generate_convert_example_to_features(cur_example, self.tokenizer,
                                                 max_source_length=self.max_source_length,
                                                 max_target_length=self.max_target_length,
                                                 task=None,
                                                 label_list=self.label_list,
                                                 output_mode=self.output_mode,
                                                 task_name=self.task_name, set_type="train")

        except:
            return self.__getitem__(item)
        cur_tensor = self.convert_feature_to_tensor(feature_a, feature_b)

        self.sample_counter += 1
        return cur_tensor

    def convert_feature_to_tensor(self, feature_a, feature_b):
        pad_token_id = self.tokenizer.pad_token_id

        cur_tensor = dict()
        for key, value in vars(feature_a).items():
            if key in ("input_ids", "attention_mask", "token_type_ids") and value is not None and not isinstance(value, str):
                cur_tensor[key] = torch.tensor(value, dtype=torch.long)

        y = torch.tensor(vars(feature_b)["input_ids"], dtype=torch.long)
        y_ids = y[:-1].contiguous()
        lm_labels = y[1:].clone()
        lm_labels[y[1:] == pad_token_id] = -100
        cur_tensor["decoder_input_ids"] = y_ids
        cur_tensor["lm_labels"] = lm_labels
        cur_tensor["y"] = y
        return cur_tensor




generate_processors = {
    "generate": GenerationProcessor,
}

generate_output_modes = {
    "generate": "classification",            # temp fix
}
generate_tasks_num_labels = {
    "generate":2,
}
