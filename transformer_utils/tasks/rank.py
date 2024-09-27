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
import csv
import logging
import os
import sys
from io import open
from sklearn.metrics import roc_auc_score
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from torch.utils.data import Dataset
import torch
import copy
from tqdm import tqdm
from transformer_utils.data_utils import DataProcessor, file_iter, file_iter_multi_task
import json
from transformers.file_utils import is_tf_available

if is_tf_available():
    import tensorflow as tf

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
    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None, language=None, line_data=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label
        self.language = language
        self.line_data = line_data

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

    def __init__(self, input_ids_a, attention_mask_a, token_type_ids_a,
                       input_ids_b, attention_mask_b, token_type_ids_b, label):
        self.input_ids_a = input_ids_a
        self.attention_mask_a = attention_mask_a
        self.token_type_ids_a = token_type_ids_a
        self.input_ids_b = input_ids_b
        self.attention_mask_b = attention_mask_b
        self.token_type_ids_b = token_type_ids_b

        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class PairWiseProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

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
            text_c = line[self.args.passage_index+1]

            if set_type == "predict":
                label = None
                line_data = '\t'.join(line)
            else:
                label = line[self.args.label_index]
                line_data = None

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=label, line_data=line_data))
        return examples
    def parse_line(self, line, set_type = "train"):
        split_line = line.split("\t")
        text_a = split_line[self.args.query_index]
        text_b = split_line[self.args.passage_index]
        text_c = split_line[self.args.passage_index+1]
        if set_type != "predict":
            label = split_line[self.args.label_index]
        else:
            label = None
        return text_a, text_b, text_c, label


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def padding_logic(input_ids, token_type_ids, attention_mask, max_length, pad_on_left, pad_token, mask_padding_with_zero, pad_token_segment_id):
    padding_length = max_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
        token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
    return input_ids, attention_mask, token_type_ids

def rank_convert_examples_to_features(examples, tokenizer,
                                      max_length=512,
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
        processor = rank_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = rank_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if set_type != "predict":
            if ex_index % 10000 == 0:
                logger.info("Writing example %d" % (ex_index))
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)

        # inputs = tokenizer.encode_plus(
        #     example.text_a,
        #     example.text_b,
        #     add_special_tokens=True,
        #     max_length=max_length,
        # )
        def convert_text_pair_to_feature(text_a, text_b):
            inputs = tokenizer.encode_plus(
                text_a, text_b, add_special_tokens=True, max_length=max_length, return_token_type_ids=True,
            )
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
            assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                                max_length)
            assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                                max_length)
            return input_ids, attention_mask, token_type_ids

        input_ids_a, attention_mask_a, token_type_ids_a = convert_text_pair_to_feature(example.text_a, example.text_b)
        input_ids_b, attention_mask_b, token_type_ids_b = convert_text_pair_to_feature(example.text_a, example.text_c)

        label = None
        if set_type != "predict":
            # if output_mode == "classification":
            #     if task_name == "topic_model":
            #         label = np.array([0] * len(label_map))
            #
            #         label_index = [label_map[item] for item in example.label if item in label_map]
            #         label[label_index] = 1
            #     else:
            #         label = label_map[example.label]
            #
            # elif output_mode == "regression":
            #     label = float(example.label)
            if output_mode == "classification":
                label = label_map[example.label]
            elif output_mode == "regression":
                if task_name == "topic_model":
                    label = np.array([0] * len(label_map))
                    label_index = [label_map[item] for item in example.label if item in label_map]
                    label[label_index] = 1
                else:
                    if isinstance(example.label, list):
                        label = np.array(example.label)
                    else:
                        label = float(example.label)
            else:
                raise KeyError(output_mode)
            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("label: %s" % (str(example.label)))
                logger.info("converted_label: %s" % (str(label)))
        feature = InputFeatures(input_ids_a, attention_mask_a, token_type_ids_a,
                                input_ids_b, attention_mask_b, token_type_ids_b, label)

        features.append(feature)

    if is_tf_available() and is_tf_dataset:
        def gen():
            for ex in features:
                yield ({'input_ids': ex.input_ids,
                         'attention_mask': ex.attention_mask,
                         'token_type_ids': ex.token_type_ids},
                        ex.label)

        return tf.data.Dataset.from_generator(gen,
            ({'input_ids': tf.int32,
              'attention_mask': tf.int32,
              'token_type_ids': tf.int32},
             tf.int64),
            ({'input_ids': tf.TensorShape([None]),
              'attention_mask': tf.TensorShape([None]),
              'token_type_ids': tf.TensorShape([None])},
             tf.TensorShape([])))

    return features

def rank_convert_example_to_features(example, tokenizer,
                                      max_length=512,
                                      task=None,
                                      label_list=None,
                                      output_mode=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True, task_name="quantus"):
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
        processor = rank_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = rank_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    if is_tf_dataset:
        example = processor.get_example_from_tensor_dict(example)
        example = processor.tfds_map(example)

    # inputs = tokenizer.encode_plus(
    #     example.text_a,
    #     example.text_b,
    #     add_special_tokens=True,
    #     max_length=max_length,
    # )

    def convert_text_pair_to_feature(text_a, text_b):
        inputs = tokenizer.encode_plus(
            text_a, text_b, add_special_tokens=True, max_length=max_length, return_token_type_ids=True,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)
        return input_ids, attention_mask, token_type_ids

    input_ids_a, attention_mask_a, token_type_ids_a = convert_text_pair_to_feature(example.text_a, example.text_b)
    input_ids_b, attention_mask_b, token_type_ids_b = convert_text_pair_to_feature(example.text_a, example.text_c)



    if output_mode == "classification":
        label = label_map[example.label]
    elif output_mode == "regression":
        if task_name == "topic_model":
            label = np.array([0] * len(label_map))
            label_index = [label_map[item] for item in example.label if item in label_map]
            label[label_index] = 1
        else:
            if isinstance(example.label, list):
                label = np.array(example.label)
            else:
                label = float(example.label)
    else:
        raise KeyError(output_mode)

    if example.guid < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("query %s"%(example.text_a))
        logger.info("passage1 %s" % (example.text_b))
        logger.info("passage2 %s" % (example.text_c))
        logger.info("label: %s" % (str(example.label)))
        logger.info("converted_label: %s"%(str(label)))

    feature = InputFeatures(input_ids_a, attention_mask_a, token_type_ids_a,
                            input_ids_b, attention_mask_b, token_type_ids_b, label)
    return feature



class BERTDataset(Dataset):
    def __init__(self, args, corpus_path, tokenizer, seq_len, label_list, lazy_load_processor, task_name, encoding='utf8', corpus_lines=None, on_memory=True, world_size=1, global_rank=0, lazy_load_block_size=1000000):
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
        self.output_mode = rank_output_modes[task_name]

        # used to keep track of full epochs on file
        self.sample_counter = 0

        # for lazy load
        self.lazy_load_block_size = lazy_load_block_size
        self.item_idx = self.lazy_load_block_size

        # load samples into memory
        if on_memory:
            self.all_docs = []
            device_rank = 0 if global_rank == -1 else global_rank
            self.corpus_lines = 0
            with open(corpus_path, "r", encoding=encoding) as f:
                for index, line in enumerate(f):
                    line = line.rstrip()
                    if len(line) > 0:
                        self.corpus_lines += 1
                        if index % world_size == device_rank:
                            logger.info(index)
                            self.all_docs.append(line)
        else:
            if not self.corpus_lines:
                self.corpus_lines = sum(1 for _ in open(corpus_path, 'r', encoding=encoding) if len(_.rstrip()) > 0)
            if self.args.multi_task:
                self.fin_corpus = file_iter_multi_task(self.corpus_path, self.encoding, world_size=self.world_size,
                                                       global_rank=self.global_rank, endless=True)
            else:
                self.fin_corpus = file_iter(self.corpus_path, self.encoding, world_size=self.world_size,
                                                       global_rank=self.global_rank)
        logger.info('Complete corpus lines: %d' % self.corpus_lines)

    def get_real_len(self):
        return self.corpus_lines

    def __len__(self):
        logger.info('corpus_lines:%d; world_size:%d; global_rank:%d' % (self.corpus_lines, self.world_size, self.global_rank))
        return (self.corpus_lines // self.world_size) + 1 if self.corpus_lines % self.world_size > self.global_rank else (self.corpus_lines // self.world_size)

    def __getitem__(self, item):
        if self.on_memory:
            line = self.all_docs[item]
        else:
            if self.args.multi_task:

                if self.item_idx == self.lazy_load_block_size:
                    self.all_docs_cache = []
                    for i in range(self.lazy_load_block_size):
                        self.all_docs_cache.append(next(self.fin_corpus))
                    self.shuffle_indices = np.random.permutation(len(self.all_docs_cache))
                    self.item_idx = 0
            else:
                if self.item_idx == self.lazy_load_block_size:
                    if len(self) - (self.sample_counter % len(self)) >= self.lazy_load_block_size:
                        read_size = self.lazy_load_block_size
                    else:
                        read_size = len(self) - (self.sample_counter % len(self))
                    logger.info('Read %d samples into memory' % read_size)
                    self.all_docs_cache = []
                    for i in range(read_size):
                        self.all_docs_cache.append(next(self.fin_corpus))

                    self.shuffle_indices = np.random.permutation(len(self.all_docs_cache))
                    self.item_idx = 0

            line = self.all_docs_cache[self.shuffle_indices[self.item_idx]].replace("\0", "").rstrip()

        self.item_idx += 1
        text_a, text_b, text_c, label = self.lazy_load_processor(line)

        cur_example = InputExample(guid=self.sample_counter, text_a=text_a, text_b=text_b, text_c = text_c, label=label)
        try:
            cur_features = rank_convert_example_to_features(cur_example, self.tokenizer,
                                                     max_length=self.args.max_seq_length,
                                                     task=None,
                                                     label_list=self.label_list,
                                                     output_mode=self.output_mode,
                                                     pad_on_left=bool(self.args.model_type in ['xlnet']),
                                                     pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                                                     pad_token_segment_id=4 if self.args.model_type in ['xlnet'] else 0,
                                                    task_name=self.task_name)
        except:
            return self.__getitem__(item)
        if self.output_mode == "classification":
            cur_tensors = (torch.tensor(cur_features.input_ids_a, dtype=torch.long),
                           torch.tensor(cur_features.attention_mask_a, dtype=torch.long),
                           torch.tensor(cur_features.token_type_ids_a, dtype=torch.long),

                           torch.tensor(cur_features.input_ids_b, dtype=torch.long),
                           torch.tensor(cur_features.attention_mask_b, dtype=torch.long),
                           torch.tensor(cur_features.token_type_ids_b, dtype=torch.long),

                           torch.tensor(cur_features.label, dtype=torch.long))

        else:
            cur_tensors = (torch.tensor(cur_features.input_ids_a, dtype=torch.long),
                           torch.tensor(cur_features.attention_mask_a, dtype=torch.long),
                           torch.tensor(cur_features.token_type_ids_a, dtype=torch.long),

                           torch.tensor(cur_features.input_ids_b, dtype=torch.long),
                           torch.tensor(cur_features.attention_mask_b, dtype=torch.long),
                           torch.tensor(cur_features.token_type_ids_b, dtype=torch.long),

                               torch.tensor(cur_features.label, dtype=torch.float))

        self.sample_counter += 1
        return cur_tensors


rank_processors = {
    "pairwise": PairWiseProcessor,
}

rank_output_modes = {
    "pairwise": "classification",            # temp fix
}
rank_tasks_num_labels = {
    "pairwise":2,
}
