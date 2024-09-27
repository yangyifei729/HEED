import torch
import copy
import json
import numpy as np
from transformer_utils.data_utils import DataProcessor, file_iter_multi_task
from torch.utils.data import Dataset
from transformers.file_utils import is_tf_available
if is_tf_available():
    import tensorflow as tf
import hashlib
import os
import logging
logger = logging.getLogger(__name__)
import random

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
    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None, line_data=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label
        self.line_data=line_data
    # def text_to_feature(self, text, tokenizer, max_seq_length):
    #     inputs = tokenizer.encode_plus(
    #         text,
    #         max_length=max_seq_length,
    #         pad_to_max_length=True,
    #         truncation=True,
    #     )
    #     return inputs
    # def to_triple_feature(self, tokenizer, max_seq_length):
    #     a = self.text_to_feature(self.text_a, tokenizer, max_seq_length)
    #     b = self.text_to_feature(self.text_b, tokenizer, max_seq_length)
    #     c = self.text_to_feature(self.text_c, tokenizer, max_seq_length)
    #     return InputFeatures(a), InputFeatures(b), InputFeatures(c)



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

    def __init__(self, input_ids=None, attention_mask=None, token_type_ids=None, input_len=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.input_len = input_len



class MatchProcessor(DataProcessor):
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
            guid = "%s-%s" % (set_type, i)
            text_a = line[self.args.query_index]
            text_b = line[self.args.passage_index]
            text_c = line[self.args.passage_index2] if self.args.passage_index2 != -1 else None
            if set_type == "predict":
                label = None
                text_c = None
                line_data = '\t'.join(line)
            else:
                # label = line[self.args.label_index]
                label = None
                line_data = None

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=label, line_data=line_data))
        return examples

    def parse_line(self, line, set_type = "train"):
        line_dict = dict()
        split_line = line.split("\t")
        text_a = split_line[self.args.query_index]
        guid = hashlib.md5(text_a.encode(encoding='UTF-8')).hexdigest()
        line_dict["guid"] = guid
        line_dict["text_a"] = text_a
        text_b = split_line[self.args.passage_index] if self.args.passage_index != -1 else None
        line_dict["text_b"] = text_b
        if set_type != "predict":
            text_c = split_line[self.args.passage_index2] if self.args.passage_index2 != -1 else None
            line_dict["text_c"] = text_c
        return line_dict

def match_convert_example_to_features(example,
                                      tokenizer,
                                      query_max_length=128,
                                      doc_max_length=32,
                                      label_list=None,
                                      set_type="dev"):

    a = tokenizer.encode_plus(
        example.text_a, None,
        max_length=query_max_length,
        truncation=True,
        pad_to_max_length=True,
        padding='max_length')
    feature_a = InputFeatures(**a)
    feature_b = None
    if example.text_b != None:
        b = tokenizer.encode_plus(
            example.text_b, None,
            max_length=doc_max_length,
            truncation=True,
            pad_to_max_length=True,
            padding='max_length')
        feature_b = InputFeatures(**b)

    feature_c = None
    if set_type != "predict":
        if example.text_c != None:
            negative_case = example.text_c
        else:
            offset = random.randint(0, int(len(label_list)/10))
            diff = list(set(label_list[offset*10:offset*10+1000]) ^ set(example.text_b))
            negative_case = random.sample(diff, 1)[0]
        c = tokenizer.encode_plus(
            negative_case, None,
            max_length=doc_max_length,
            truncation=True,
            pad_to_max_length=True,
            padding='max_length')
        feature_c = InputFeatures(**c)

    return feature_a, feature_b, feature_c

def match_convert_examples_to_features(examples,
                                      tokenizer,
                                      query_max_length=128,
                                      doc_max_length=32,
                                      label_list=None,
                                      set_type = "dev"):
    def get_text_c(example):
        if example.text_c != None:
            negative_case = example.text_c
        else:
            diff = list(set(label_list) ^ set(example.text_b))
            negative_case = random.sample(diff, 1)[0]
        return negative_case

    def get_batch_features(text_list, temp_seq_length):
        batch_encoding = tokenizer.batch_encode_plus(
            text_list,
            max_length=temp_seq_length,
            padding="max_length",
            truncation=True,
            pad_to_max_length=True,
        )
        features = []
        for i in range(len(examples)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}

            feature = InputFeatures(**inputs)
            features.append(feature)
        return features

    text_list_a = [example.text_a for example in examples]
    features_list_a = get_batch_features(text_list_a, query_max_length)

    features_list_b = [None] * len(examples)
    if examples[0].text_b != None:
        text_list_b = [example.text_b for example in examples]
        features_list_b = get_batch_features(text_list_b, doc_max_length)

    features_list_c = [None] * len(examples)
    if set_type != "predict":

        text_list_c = [get_text_c(example) for example in examples]
        features_list_c = get_batch_features(text_list_c, doc_max_length)
    # features = []
    # for feature_a, feature_b, feature_c in zip(features_list_a,features_list_b,features_list_c):
    #     features.append((feature_a,feature_b,feature_c))
    return features_list_a, features_list_b, features_list_c

class BERTDataset(Dataset):

    def __init__(self, args, corpus_path, tokenizer, seq_len, label_list,
                 lazy_load_processor, task_name, encoding='utf8', corpus_lines=None,
                 on_memory=True, world_size=1, global_rank=0, lazy_load_block_size=1000000):
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
                try:
                    self.all_docs_cache.append(next(self.fin_corpus))
                except:
                    continue
            self.shuffle_indices = np.random.permutation(len(self.all_docs_cache))
            self.item_idx = 0

        line = self.all_docs_cache[self.shuffle_indices[self.item_idx]].replace("\0", "").rstrip()

        self.item_idx += 1
        try:
        # if True:
            line_dict = self.lazy_load_processor(line)
            cur_example = InputExample(**line_dict)
            cur_example.guid = self.sample_counter

            cur_features = match_convert_example_to_features(cur_example,
                                                             self.tokenizer,
                                                             query_max_length=self.args.max_seq_length,
                                                             doc_max_length=self.args.max_passage_seq_length,
                                                             label_list=self.label_list,
                                                             set_type="train")
        except Exception as e:
            print(e)
            return self.__getitem__(item)
        index_dict = {0:"a", 1:"b", 2:"c"}
        cur_tensor = {index_dict[index]:self.convert_feature_to_tensor(item) for index, item in enumerate(cur_features) if item != None}
        self.sample_counter += 1
        return cur_tensor

    def convert_feature_to_tensor(self, cur_feature):
        if cur_feature == None:
            return None
        cur_tensor = dict()
        for key, value in vars(cur_feature).items():
            if key in ("input_ids", "attention_mask", "token_type_ids") and value is not None and not isinstance(value, str):
                cur_tensor[key] = torch.tensor(value, dtype=torch.long)

        return cur_tensor



glue_processors = {
    "match":MatchProcessor
}

glue_output_modes = {
    "match":"classification"
}
