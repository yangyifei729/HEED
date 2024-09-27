import sys
import json
import hashlib
import os
import logging
from transformers.file_utils import is_tf_available
# from transformer_utils.datasets_helper import glue_processors,glue_output_modes
from transformer_utils.data_utils import DataProcessor
from transformer_utils.data_utils import InputFeatures, InputExample
from sklearn.preprocessing import minmax_scale
# from transformer_utils.tasks.glue import glue_processors, glue_output_modes
import numpy as np

logger = logging.getLogger(__name__)
import copy
from collections import deque, defaultdict
import argparse

PAD_TOKEN_ID = 1

if is_tf_available():
    import tensorflow as tf


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

    def __init__(self, guid, text_a, text_b=None, label=None, language=None, line_data=None, offset_mapping=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
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


class InputExampleEE(object):
    "A single training/test example for EE token classification"
    def __init__(self, guid, input_ids, attention_mask, label, visual_features):
        self.guid = guid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label = label
        self.visual_features = visual_features



class XMLEEProcessor(DataProcessor):
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
        if label_file is None:
            return ['O', 'B', 'I', 'E', 'U']
        else:
            # prefix = ['B', 'I', 'E', 'U']
            prefix = ['B', 'I']
            label_list = ["O", "X"]
            with open(label_file, encoding="utf-8") as fin:
                for line in fin:
                    entity_type = line.strip()
                    for item in prefix:
                        label_list.append(item + "-" + entity_type)
            return label_list

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[self.args.query_index]
            text_b = line[self.args.passage_index]
            if self.args.language_index == -1:
                language = None
            else:
                language = line[self.args.language_index]
            if set_type == "predict":
                label = None
                line_data = '\t'.join(line)
            else:
                label = line[self.args.label_index]
                line_data = None

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=language))
        return examples

    def parse_line(self, line, set_type="train"):
        line_dict = dict()
        line = line.rstrip()
        split_line = line.split("\t")
        text_a = split_line[int(self.args.query_index)].strip()
        text_b = split_line[int(self.args.passage_index)] if int(self.args.passage_index) != -1 else None
        line_dict['visual_feature'] = None if self.args.visual_feature_index is None else json.loads(split_line[int(self.args.visual_feature_index)])
        data = text_a
        guid = hashlib.md5(data.encode(encoding='UTF-8')).hexdigest()
        line_dict["guid"] = guid
        line_dict["text_a"] = text_a
        line_dict["text_b"] = text_b
        if set_type != "predict":
            label = split_line[int(self.args.label_index)]
            line_dict["label"] = json.loads(label)

        return line_dict


class XMLEEProcessor2(DataProcessor):
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
        with open(os.path.join(data_dir, file_name), "r") as f:
            lines = f.readlines()
        return self._create_examples(lines, "dev")

    def get_predict_examples(self, input_file):
        """See base class."""
        return self._create_examples(
            self._read_tsv(input_file), "predict")

    def get_labels(self, label_file=None):
        """See base class."""
        if label_file is None:
            return ['O', 'B', 'I', 'E', 'U']
        else:
            # prefix = ['B', 'I', 'E', 'U']
            prefix = ['B', 'I']
            label_list = ["O"]
            with open(label_file, encoding="utf-8") as fin:
                for line in fin:
                    entity_type = line.strip()
                    for item in prefix:
                        label_list.append(item + "-" + entity_type)
            return label_list

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            line_data = json.loads(line)
            examples.append(
                InputExampleEE(guid=guid, 
                input_ids=line_data['input_ids'], 
                attention_mask=line_data['attention_mask'], 
                label=line_data['label'], 
                visual_features=line_data['visual_features'])
            )
        return examples

    def parse_line(self, line, set_type="train"):
        line = line.rstrip()
        split_line = line.split("\t")
        line_dict = json.loads(split_line[-1])
        if "visaul_features" in line_dict:
            line_dict["visual_features"] = line_dict["visaul_features"]
            del line_dict["visaul_features"]
        return line_dict



class XMLEEProcessor3(DataProcessor):
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
        with open(os.path.join(data_dir, file_name), "r") as f:
            lines = f.readlines()
        return self._create_examples(lines, "dev")

    def get_predict_examples(self, input_file):
        """See base class."""
        return self._create_examples(
            self._read_tsv(input_file), "predict")

    def get_labels(self, label_file=None):
        """See base class."""
        if label_file is None:
            return ['O', 'B', 'I', 'E', 'U']
        else:
            label_list = ["O"]
            with open(label_file, encoding="utf-8") as fin:
                for line in fin:
                    entity_type = line.strip()
                    label_list.append(entity_type)
            return label_list

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            line_data = json.loads(line)
            examples.append(
                InputExampleEE(guid=guid, 
                input_ids=line_data['input_ids'], 
                attention_mask=line_data['attention_mask'], 
                label=line_data['label'], 
                visual_features=line_data['visual_features'])
            )
        return examples

    def parse_line(self, line, set_type="train"):
        line = line.rstrip()
        split_line = line.split("\t")
        line_dict = json.loads(split_line[-1])
        if "visaul_features" in line_dict:
            line_dict["visual_features"] = line_dict["visaul_features"]
            del line_dict["visaul_features"]
        return line_dict


import re
import string


def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)


def check_match(start_idx, full_token, tag):
    for idx in range(len(tag)):
        if full_token[start_idx + idx] != tag[idx]:
            return False

    return True


def find_partial_token(full_token, tag, ignoreFirstTokenInMatch=False):
    if len(tag) <= 2 and ignoreFirstTokenInMatch:
        # don't do anything
        return []

    if ignoreFirstTokenInMatch:
        tag = tag[1:]

    end_idx = len(full_token) - len(tag) + 1

    match_start_end = []

    for idx in range(end_idx):
        ismatch = check_match(idx, full_token, tag)

        if ismatch:
            if ignoreFirstTokenInMatch:
                match_start_end.append([max(idx - 1, 0), idx + len(tag)])
            else:
                match_start_end.append([idx, idx + len(tag)])

    return match_start_end


class IdxTag_Converter(object):
    ''' idx2tag : a tag list like ['O','B','I','E','U']
        tag2idx : {'O': 0, 'B': 1, ..., 'U':4}
    '''

    def __init__(self, idx2tag):
        self.idx2tag = idx2tag
        tag2idx = {}
        for idx, tag in enumerate(idx2tag):
            tag2idx[tag] = idx
        self.tag2idx = tag2idx

    def convert_idx2tag(self, index_list):
        tag_list = [self.idx2tag[index] for index in index_list]
        return tag_list

    def convert_tag2idx(self, tag_list):
        index_list = [self.tag2idx[tag] for tag in tag_list]
        return index_list


def filter_overlap(positions):
    '''delete overlap keyphrase positions'''
    previous_e = -1
    filter_positions = []
    for i, (s, e) in enumerate(positions):
        if s <= previous_e:
            continue
        filter_positions.append(positions[i])
        previous_e = e
    return filter_positions


def xmlee_phrase_convert_examples_to_features2(example, tokenizer,
                                               max_length=512,
                                               task=None,
                                               label_list=None,
                                               label_map=None,
                                               output_mode=None,
                                               task_name="quantus", set_type="dev"):
    is_tf_dataset = False
    if is_tf_available() and isinstance(example, tf.data.Dataset):
        is_tf_dataset = True

    raw_text = example.text_a.replace(" <sep> ", " ") if set_type != "predict" else example.text_a

    # doc_tokens = tokenizer.encode(raw_text, add_special_tokens=True)

    def test_cleaning(text):
        text = text.split()
        for i in range(len(text)):
            if text[i][:4] == "http":
                text[i] = "http"
        return " ".join(text)

    raw_text = test_cleaning(raw_text)

    if set_type == "predict":
        inputs = tokenizer.encode_plus(
            raw_text,
            max_length=max_length,
            truncation=False,
            pad_to_max_length=True,
            padding='max_length',
            return_offsets_mapping=True,
            return_overflowing_tokens=True
        )
    else:
        inputs = tokenizer.encode_plus(
            raw_text,
            max_length=max_length,
            truncation=False,
            pad_to_max_length=True,
            padding='max_length',
            return_offsets_mapping=True,
            return_overflowing_tokens=True
        )

    if label_map == None:
        label_map = {label: i for i, label in enumerate(label_list)}
    label = None

    def word2piece_map(raw_text, text_b, offset_mapping, max_length=128):
        word2token = []
        piece_index = 0
        word_index = 0

        def count_maping(text, word2token, piece_index, word_index):
            words = text.split()
            current_end = 0
            while piece_index < len(offset_mapping) and word_index < len(words) + 1:
                if offset_mapping[piece_index][1] == 0:
                    piece_index += 1
                    try:
                        if offset_mapping[piece_index][1] == 0:
                            piece_index += 1
                            break
                    except:
                        break
                    continue
                if offset_mapping[piece_index][1] > current_end:
                    word2token.append(piece_index)
                    if word_index == len(words):
                        break
                    current_end += len(words[word_index])
                    word_index += 1
                    piece_index += 1
                else:
                    piece_index += 1
            return piece_index, word_index

        piece_index, word_index = count_maping(raw_text, word2token, piece_index, word_index)

        token2word = [0] * len(offset_mapping)
        original_place = 0
        for i, item in enumerate(word2token):
            while original_place < item:
                token2word[original_place] = i - 1
                original_place += 1
        while original_place < len(token2word):
            token2word[original_place] = len(word2token) - 1
            original_place += 1

        word_index_offset = word_index
        piece_index_offset = piece_index
        if text_b != None:
            piece_index, word_index = count_maping(text_b, word2token, piece_index, word_index)

        return word2token, token2word, word_index_offset, piece_index_offset

    token2piece_offset = word2piece_map(raw_text, example.text_b, inputs.data["offset_mapping"])

    def get_label(labels, token2piece_offset, doc_tokens):
        orig_label = ['X' for _ in range(len(doc_tokens))]

        word_mapping = token2piece_offset[0]

        for char_index in word_mapping:
            orig_label[char_index] = "O"

        for e_type, se_index in labels.items():
            if e_type == "Product":
                continue
            word_s = se_index[0]
            word_e = se_index[1]
            if word_e >= len(word_mapping):
                continue
            for tmp_i in range(word_s, word_e):
                char_start = word_mapping[tmp_i]
                orig_label[char_start] = "I-" + e_type
            char_start = word_mapping[word_s]
            orig_label[char_start] = "B-" + e_type

        c_label = [label_map[item] for item in orig_label]
        return c_label

    if set_type != "predict":
        label = get_label(example.label, token2piece_offset, inputs.data["input_ids"])
    # inputs["attention_mask"] = [1 if i < (len(inputs["input_ids"]) - inputs["input_ids"].count(PAD_TOKEN_ID)) else 0 for i in range(len(inputs["input_ids"]))]
    feature = InputFeatures(**inputs, label=label)
    return feature


def xmlee_phrase_convert_examples_to_features(args, example, tokenizer,
                                              max_length=512,
                                              task=None,
                                              label_list=None,
                                              label_map=None,
                                              output_mode=None,
                                              task_name="quantus", set_type="dev"):
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
    # if label_map==None:
    #     label_map = {label: i for i, label in enumerate(label_list)}

    if is_tf_dataset:
        example = processor.get_example_from_tensor_dict(example)
        example = processor.tfds_map(example)

    raw_text = example.text_a.replace(" <sep> ", " ") if set_type != "predict" else example.text_a

    # doc_tokens = tokenizer.encode(raw_text, add_special_tokens=True)
    def test_cleaning(text):
        text = text.split()
        for i in range(len(text)):
            if text[i][:4] == "http":
                text[i] = "http"
        return " ".join(text)

    raw_text = test_cleaning(raw_text)

    if set_type == "predict":
        inputs = tokenizer.encode_plus(
            raw_text, example.text_b,
            max_length=max_length,
            truncation=True,
            pad_to_max_length=True,
            padding='max_length',
            return_offsets_mapping=True
        )
    else:
        inputs = tokenizer.encode_plus(
            raw_text, example.text_b,
            max_length=max_length,
            truncation=True,
            pad_to_max_length=True,
            padding='max_length',
            return_offsets_mapping=True
        )

    if label_map == None:
        label_map = {label: i for i, label in enumerate(label_list)}
    label = None

    def word2piece_map(raw_text, text_b, offset_mapping, inputs, tokenizer):
        word2token = []
        piece_index = 0
        word_index = 0

        def count_maping(text, word2token, piece_index, word_index):
            words = text.split()
            current_end = -1
            # split_char_number = 0
            last_end = piece_index
            while piece_index >= len(offset_mapping):
                break
            # skip first 0
            while offset_mapping[piece_index][1] == 0:
                last_end = piece_index
                piece_index += 1
            while word_index < len(words):
                current_end += len(words[word_index]) + 1
                while piece_index < len(offset_mapping):
                    if offset_mapping[piece_index][1] == 0:
                        break
                    if offset_mapping[piece_index][1] == current_end:
                        while piece_index + 1 < len(offset_mapping) and offset_mapping[piece_index + 1][1] == \
                                offset_mapping[piece_index][1]:
                            piece_index += 1
                        word2token.append(last_end + 1)
                        last_end = piece_index
                        piece_index += 1
                        break
                    piece_index += 1
                word_index += 1

            return piece_index, word_index

        piece_index, word_index = count_maping(raw_text, word2token, piece_index, word_index)

        token2word = [0] * len(offset_mapping)
        original_place = 0
        for i, item in enumerate(word2token):
            while original_place < item:
                token2word[original_place] = i - 1
                original_place += 1
        while original_place < len(token2word):
            token2word[original_place] = len(word2token) - 1
            original_place += 1

        word_index_offset = word_index
        piece_index_offset = piece_index
        if text_b != None:
            piece_index, word_index = count_maping(text_b, word2token, piece_index, 0)

        return word2token, token2word, word_index_offset, piece_index_offset

    token2piece_offset = word2piece_map(raw_text, example.text_b, inputs.data["offset_mapping"], inputs, tokenizer)

    def get_label(labels, token2piece_offset, doc_tokens):
        orig_label = ['X' for _ in range(len(doc_tokens))]

        word_mapping = token2piece_offset[0]

        for char_index in word_mapping:
            orig_label[char_index] = "O"

        for e_type, se_index in labels.items():
            if e_type == "Product":
                continue
            for item in se_index:
                word_s = item[0]
                word_e = item[1] + 1
                if word_e >= len(word_mapping):
                    continue
                for tmp_i in range(word_s, word_e):
                    char_start = word_mapping[tmp_i]
                    orig_label[char_start] = "I-" + e_type
                char_start = word_mapping[word_s]
                orig_label[char_start] = "B-" + e_type

        c_label = [label_map[item] for item in orig_label]
        return c_label

    if set_type != "predict":
        label = get_label(example.label, token2piece_offset, inputs.data["input_ids"])
    inputs.data["offset_mapping"] = token2piece_offset
    # inputs["attention_mask"] = [1 if i < (len(inputs["input_ids"]) - inputs["input_ids"].count(PAD_TOKEN_ID)) else 0 for i in range(len(inputs["input_ids"]))]

    visual_feature, pre = [], 0

    if example.visual_feature:

        for _ in token2piece_offset[1]:
            visual_feature.append([0] * len(example.visual_feature[0]))

        for idx, word_idx in enumerate(token2piece_offset[1]):
            if word_idx == -1: continue

            visual_feature[idx] = example.visual_feature[word_idx]

        scale_size = [args.equal_size,
                      args.max_bounding_box_size, args.max_bounding_box_size, args.max_bounding_box_size,
                      args.max_bounding_box_size,
                      args.max_color_size, args.max_color_size, args.max_color_size, args.max_color_size,
                      args.equal_size,
                      args.max_FontWeight_size, args.max_FontWeight_size,
                      args.equal_size, args.equal_size, args.equal_size, args.equal_size,
                      args.max_bounding_box_size, args.max_bounding_box_size, args.max_bounding_box_size,
                      args.max_bounding_box_size,
                      args.equal_size
                      ]

        for i, size in enumerate(scale_size):
            visual_feature = np.array(visual_feature)
            visual_feature[:, i] = np.trunc(minmax_scale(visual_feature[:, i], feature_range=[0, size - 1])).astype(
                np.int32)
        visual_feature = visual_feature.tolist()

    feature = InputFeatures(**inputs, label=label, visual_features=visual_feature)

    return feature


def xmlee_phrase_convert_examples_to_features_new(args, example, tokenizer, processor=None,
                                              max_length=512,
                                              task=None,
                                              label_list=None,
                                              label_map=None,
                                              output_mode=None,
                                              task_name="quantus", set_type="dev"):
    is_tf_dataset = False
    if is_tf_available() and isinstance(example, tf.data.Dataset):
        is_tf_dataset = True
    if task is not None:
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))
    # if label_map==None:
    #     label_map = {label: i for i, label in enumerate(label_list)}

    if is_tf_dataset:
        example = processor.get_example_from_tensor_dict(example)
        example = processor.tfds_map(example)

    raw_text = example.text_a.replace(" <sep> ", " ") if set_type != "predict" else example.text_a

    # doc_tokens = tokenizer.encode(raw_text, add_special_tokens=True)
    def test_cleaning(text):
        text = text.split()
        for i in range(len(text)):
            if text[i][:4] == "http":
                text[i] = "http"
        return " ".join(text)

    raw_text = test_cleaning(raw_text)

    if set_type == "predict":
        inputs = tokenizer.encode_plus(
            raw_text, example.text_b,
            max_length=max_length,
            truncation=True,
            pad_to_max_length=True,
            padding='max_length',
            return_offsets_mapping=True
        )
    else:
        inputs = tokenizer.encode_plus(
            raw_text, example.text_b,
            max_length=max_length,
            truncation=True,
            pad_to_max_length=True,
            padding='max_length',
            return_offsets_mapping=True
        )

    if label_map == None:
        label_map = {label: i for i, label in enumerate(label_list)}
    label = None

    def word2piece_map(raw_text, text_b, offset_mapping, inputs, tokenizer):
        word2token = []
        piece_index = 0
        word_index = 0

        def count_maping(text, word2token, piece_index, word_index):
            words = text.split()
            current_end = -1
            # split_char_number = 0
            last_end = piece_index
            while piece_index >= len(offset_mapping):
                break
            # skip first 0
            while offset_mapping[piece_index][1] == 0:
                last_end = piece_index
                piece_index += 1
            while word_index < len(words):
                current_end += len(words[word_index]) + 1
                while piece_index < len(offset_mapping):
                    if offset_mapping[piece_index][1] == 0:
                        break
                    if offset_mapping[piece_index][1] == current_end:
                        while piece_index + 1 < len(offset_mapping) and offset_mapping[piece_index + 1][1] == \
                                offset_mapping[piece_index][1]:
                            piece_index += 1
                        word2token.append(last_end + 1)
                        last_end = piece_index
                        piece_index += 1
                        break
                    piece_index += 1
                word_index += 1

            return piece_index, word_index

        piece_index, word_index = count_maping(raw_text, word2token, piece_index, word_index)

        token2word = [0] * len(offset_mapping)
        original_place = 0
        for i, item in enumerate(word2token):
            while original_place < item:
                token2word[original_place] = i - 1
                original_place += 1
        while original_place < len(token2word):
            token2word[original_place] = len(word2token) - 1
            original_place += 1

        word_index_offset = word_index
        piece_index_offset = piece_index
        if text_b != None:
            piece_index, word_index = count_maping(text_b, word2token, piece_index, 0)

        return word2token, token2word, word_index_offset, piece_index_offset

    token2piece_offset = word2piece_map(raw_text, example.text_b, inputs.data["offset_mapping"], inputs, tokenizer)

    def get_label(labels, token2piece_offset, doc_tokens):
        orig_label = ['X' for _ in range(len(doc_tokens))]

        word_mapping = token2piece_offset[0]

        for char_index in word_mapping:
            orig_label[char_index] = "O"

        for e_type, se_index in labels.items():
            if e_type == "Product":
                continue
            for item in se_index:
                word_s = item[0]
                word_e = item[1] + 1
                if word_e >= len(word_mapping):
                    continue
                for tmp_i in range(word_s, word_e):
                    char_start = word_mapping[tmp_i]
                    orig_label[char_start] = "I-" + e_type
                char_start = word_mapping[word_s]
                orig_label[char_start] = "B-" + e_type

        c_label = [label_map[item] for item in orig_label]
        return c_label

    if set_type != "predict":
        label = get_label(example.label, token2piece_offset, inputs.data["input_ids"])
    inputs.data["offset_mapping"] = token2piece_offset
    # inputs["attention_mask"] = [1 if i < (len(inputs["input_ids"]) - inputs["input_ids"].count(PAD_TOKEN_ID)) else 0 for i in range(len(inputs["input_ids"]))]

    visual_feature, pre = [], 0

    if example.visual_feature:

        for _ in token2piece_offset[1]:
            visual_feature.append([0] * len(example.visual_feature[0]))

        for idx, word_idx in enumerate(token2piece_offset[1]):
            if word_idx == -1: continue

            visual_feature[idx] = example.visual_feature[word_idx]

        scale_size = [args.equal_size,
                      args.max_bounding_box_size, args.max_bounding_box_size, args.max_bounding_box_size,
                      args.max_bounding_box_size,
                      args.max_color_size, args.max_color_size, args.max_color_size, args.max_color_size,
                      args.equal_size,
                      args.max_FontWeight_size, args.max_FontWeight_size,
                      args.equal_size, args.equal_size, args.equal_size, args.equal_size,
                      args.max_bounding_box_size, args.max_bounding_box_size, args.max_bounding_box_size,
                      args.max_bounding_box_size,
                      args.equal_size
                      ]

        for i, size in enumerate(scale_size):
            visual_feature = np.array(visual_feature)
            visual_feature[:, i] = np.trunc(minmax_scale(visual_feature[:, i], feature_range=[0, size - 1])).astype(
                np.int32)
        visual_feature = visual_feature.tolist()

    feature = InputFeatures(**inputs, label=label, visual_features=visual_feature)

    return feature


def xmlee_phrase_convert_examples_to_features3(example, tokenizer,
                                               max_length=512,
                                               task=None,
                                               label_list=None,
                                               label_map=None,
                                               output_mode=None,
                                               task_name="quantus", set_type="dev"):
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
    # if label_map==None:
    #     label_map = {label: i for i, label in enumerate(label_list)}

    if is_tf_dataset:
        example = processor.get_example_from_tensor_dict(example)
        example = processor.tfds_map(example)

    raw_text = example.text_a.replace(" <sep> ", " ") if set_type != "predict" else example.text_a

    # doc_tokens = tokenizer.encode(raw_text, add_special_tokens=True)
    def test_cleaning(text):
        text = text.split()
        for i in range(len(text)):
            if text[i][:4] == "http":
                text[i] = "http"
        return " ".join(text)

    raw_text = test_cleaning(raw_text)

    if set_type == "predict":
        inputs = tokenizer.encode_plus(
            raw_text, example.text_b,
            max_length=max_length,
            truncation=True,
            pad_to_max_length=True,
            padding='max_length',
            return_offsets_mapping=True,
        )
    else:
        inputs = tokenizer.encode_plus(
            raw_text, example.text_b,
            max_length=max_length,
            truncation=True,
            pad_to_max_length=True,
            padding='max_length',
            return_offsets_mapping=True,
        )

    if label_map == None:
        label_map = {label: i for i, label in enumerate(label_list)}
    label = None

    def word2piece_map(raw_text, text_b, offset_mapping):
        word2token = []
        piece_index = 0
        word_index = 0

        def count_maping(text, word2token, piece_index, word_index):
            words = text.split()
            current_end = -1
            # split_char_number = 0
            last_end = piece_index
            while piece_index >= len(offset_mapping):
                break
            # skip first 0
            while offset_mapping[piece_index][1] == 0:
                last_end = piece_index
                piece_index += 1
            while word_index < len(words):
                current_end += len(words[word_index]) + 1
                while piece_index < len(offset_mapping):
                    if offset_mapping[piece_index][1] == 0:
                        break
                    if offset_mapping[piece_index][1] == current_end:
                        while piece_index + 1 < len(offset_mapping) and offset_mapping[piece_index + 1][1] == \
                                offset_mapping[piece_index][1]:
                            piece_index += 1
                        word2token.append(last_end + 1)
                        last_end = piece_index
                        piece_index += 1
                        break
                    piece_index += 1
                word_index += 1

            return piece_index, word_index

        piece_index, word_index = count_maping(raw_text, word2token, piece_index, word_index)

        token2word = [0] * len(offset_mapping)
        original_place = 0
        for i, item in enumerate(word2token):
            while original_place < item:
                token2word[original_place] = i - 1
                original_place += 1
        while original_place < len(token2word):
            token2word[original_place] = len(word2token) - 1
            original_place += 1

        word_index_offset = word_index
        piece_index_offset = piece_index
        if text_b != None:
            piece_index, word_index = count_maping(text_b, word2token, piece_index, word_index=0)

        return word2token, token2word, word_index_offset, piece_index_offset

    token2piece_offset = word2piece_map(raw_text, example.text_b, inputs.data["offset_mapping"])

    def get_label(labels, token2piece_offset, doc_tokens):
        orig_label = ['O' for _ in range(len(doc_tokens))]

        word_mapping = token2piece_offset[0]

        for char_index in range(len(doc_tokens)):
            if doc_tokens[char_index] < 3:
                orig_label[char_index] = "X"

        for e_type, se_index in labels.items():
            if e_type == "Product":
                continue
            word_s = se_index[0]
            word_e = se_index[1] + 1

            if word_e > len(word_mapping):
                continue

            char_start = word_mapping[word_s]
            if word_e == len(word_mapping):
                char_end = word_mapping[word_e - 1]
                while char_end < len(doc_tokens):
                    if doc_tokens[char_end] < 3:
                        break
                    char_end += 1
            else:
                char_end = word_mapping[word_e]
            if char_end - char_start == 1:
                orig_label[char_start] = "U-" + e_type
                continue
            for tmp_i in range(char_start, char_end):
                orig_label[tmp_i] = "I-" + e_type
            orig_label[char_end - 1] = "E-" + e_type
            orig_label[char_start] = "B-" + e_type

        c_label = [label_map[item] for item in orig_label]
        return c_label, orig_label

    if set_type != "predict":
        label, orig_label = get_label(example.label, token2piece_offset, inputs.data["input_ids"])
    # inputs["attention_mask"] = [1 if i < (len(inputs["input_ids"]) - inputs["input_ids"].count(PAD_TOKEN_ID)) else 0 for i in range(len(inputs["input_ids"]))]
    feature = InputFeatures(**inputs, label=label)
    return feature


def decode_ngram(orig_tokens, token_logits, converter, n, tokenizer, pooling=None):
    '''
    Combine n-gram score and sorted
    Inputs :
        n : n_gram
        orig_tokens : document lower cased words' list
        token_logits : each token has five score : for 'O', 'B', 'I', 'E', 'U' tag
        pooling : pooling method :  mean / min / log_mean
    Outputs : sorted phrase and socre list
    '''
    ngram_ids = [4] if n == 1 else [1] + [2 for _ in range(n - 2)] + [3]
    offsets = [i for i in range(len(ngram_ids))]

    # combine n-gram scores
    phrase_set = defaultdict(int)
    valid_length = len(orig_tokens) - n  # start and end are special <s>
    for i in range(1, valid_length):
        n_gram = tokenizer.decode(orig_tokens[i:i + n])

        if pooling == 'mean':
            n_gram_score = float(np.mean([token_logits[i + bias][tag] for bias, tag in zip(offsets, ngram_ids)]))
        elif pooling == 'min':
            n_gram_score = min([token_logits[i + bias][tag] for bias, tag in zip(offsets, ngram_ids)])
        elif pooling == 'log_mean':
            n_gram_score = float(
                np.mean([np.log(token_logits[i + bias][tag]) for bias, tag in zip(offsets, ngram_ids)]))
        else:
            logger.info('not %s pooling method !' % pooling)

        phrase_set[n_gram] += n_gram_score

    return list(phrase_set.items())


def decode_ngram_min_pooling(orig_tokens, token_logits, converter, n, tokenizer):
    '''
    Combine n-gram score and sorted
    This is a fast version of method decode_ngram with min pooling
    Inputs :
        n : n_gram
        orig_tokens : document lower cased words' list
        token_logits : each token has five score : for 'O', 'B', 'I', 'E', 'U' tag
    Outputs : sorted phrase index and socre list
    '''
    # combine n-gram scores
    phrase_set = defaultdict(int)
    que = deque()

    if n == 1:
        for i in range(1, len(orig_tokens) - 1):
            phrase_set[(i, i + 1)] += token_logits[i][4]
    elif n == 2:
        for i in range(1, len(orig_tokens) - 2):
            phrase_set[(i, i + 2)] += min(token_logits[i][1], token_logits[i + 1][3])
    else:
        for i in range(2, len(orig_tokens) - 2):
            while que and i - que[0] >= n - 2:
                que.popleft()
            while que and token_logits[que[-1]][2] > token_logits[i][2]:
                que.pop()
            que.append(i)
            if i >= n - 1:
                n_gram_score = min([token_logits[i - n + 2][1], token_logits[que[0]][2], token_logits[i + 1][3]])
                phrase_set[(i - n + 2, i + 2)] += n_gram_score

    return list(phrase_set.items())


def add_preprocess_opts(parser):
    parser.add_argument("--query_index", type=int, default=0)
    parser.add_argument("--passage_index", type=int, default=1)
    parser.add_argument("--label_index", type=int, default=1)
    parser.add_argument("--visual_feature_index", type=int, default=2)


if __name__ == "__main__":
    from transformers import XLMRobertaTokenizerFast
    from transformer_utils.data_utils import InputFeatures, InputExample
    tokenizer = XLMRobertaTokenizerFast.from_pretrained("D:\\job\\entityextractionxml\\transformer_lite\\model\\xlm_new_base\\")
    parser = argparse.ArgumentParser()
    add_preprocess_opts(parser)
    args = parser.parse_args()
    print(args)
    label_list = ['O', 'X', 'B-Name', 'I-Name', 'B-MainImage', 'I-MainImage', 'B-Manufacturer', 'I-Manufacturer', 'B-Price', 'I-Price', 'B-Rating', 'I-Rating', 'B-NumberofReviews', 'I-NumberofReviews', 'B-ProductCode', 'I-ProductCode', 'B-OutOfStock', 'I-OutOfStock']
    label_map = {label: i for i, label in enumerate(label_list)}
    examples = list()
    processor = XMLEEProcessor(args)
    error_count = 0

    fw = open("D:\\job\\rampup\\PICL\\EdgeProductModels2\\V1\\NewTest\\FR\\overlap\\test.conll.list_label_format_features_xlmroberta_seq512.tsv", 'w', encoding='utf8')
    with open("D:\\job\\rampup\\PICL\\EdgeProductModels2\\V1\\NewTest\\FR\\overlap\\test.conll.list_label_format.tsv",
              'r', encoding='utf8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            # try:
            line = line.replace('\n', '')
            line_dict = processor.parse_line(line)
            cur_example = InputExample(**line_dict)
            data = xmlee_phrase_convert_examples_to_features_new(args,
                                                                 cur_example,
                                                                 tokenizer,
                                                                 processor=processor,
                                                                 max_length=512,
                                                                 task=None,
                                                                 label_list=label_list,
                                                                 label_map=label_map,
                                                                 output_mode="classification",
                                                                 task_name="xml_ee", set_type="train")
            data = {
                "input_ids": data.input_ids,
                "attention_mask": data.attention_mask,
                "label": data.label
            }
            fw.write(json.dumps(data))
            fw.write('\n')
            # except Exception as e:
            #     error_count += 1

    print(f"error count:{error_count}")
