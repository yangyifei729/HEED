import sys
import json
import hashlib
import os
import logging
from transformers.file_utils import is_tf_available
# from transformer_utils.datasets_helper import glue_processors,glue_output_modes
from transformer_utils.data_utils import DataProcessor
from transformer_utils.data_utils import InputFeatures, InputExample

#from transformer_utils.tasks.glue import glue_processors, glue_output_modes
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
    def __init__(self, guid, text_a, text_b=None, label=None, language=None, line_data=None):
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

class KeyPhraseProcessor(DataProcessor):
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
        return ['O','B','I','E','U']

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
        line = line.rstrip()
        split_line = line.split("\t")
        text_a = split_line[self.args.query_index]
        text_b = split_line[self.args.passage_index] if self.args.passage_index != -1 else None
        data = text_a
        guid = hashlib.md5(data.encode(encoding='UTF-8')).hexdigest()
        line_dict["guid"] = guid
        line_dict["text_a"] = text_a
        line_dict["text_b"] = text_b

        if set_type != "predict":
            label = split_line[self.args.label_index]
            line_dict["label"] = json.loads(label)

        return line_dict

class KeyPhraseProcessor2(DataProcessor):
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
        return ['O','B','I','E','U']

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
    # def parse_line(self, line, set_type = "train"):
    #     line_dict = dict()
    #     line = line.rstrip()
    #     split_line = line.split("\t")
    #     text_a = split_line[self.args.query_index]
    #     text_b = split_line[self.args.passage_index] if self.args.passage_index != -1 else None
    #     data = text_a
    #     guid = hashlib.md5(data.encode(encoding='UTF-8')).hexdigest()
    #     line_dict["guid"] = guid
    #     line_dict["text_a"] = text_a
    #     line_dict["text_b"] = text_b
    #
    #     if set_type != "predict":
    #         label = split_line[self.args.label_index]
    #         line_dict["label"] = json.loads(label)
    #     return line_dict

    def parse_line(self, line, set_type="train"):
        line = line.rstrip()
        split_line = line.split("\t")
        line_dict = json.loads(split_line[-1])
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

def key_phrase_convert_examples_to_features(examples,
                                            tokenizer,
                                            max_length=512,
                                            task=None,
                                            label_list=None,
                                            label_map=None,
                                            output_mode=None,
                                            task_name="quantus",
                                            set_type= "dev"):
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

    def get_label(example, input_ids):
        label = None
        if set_type != "predict":
            doc_tokens = input_ids
            # +2 to handle starting and ending special token
            # if len(doc_tokens) > max_length + 2:
            #     doc_tokens = doc_tokens[:max_length + 1] + [doc_tokens[-1]]

            doc_original_str = tokenizer.decode(doc_tokens)
            doc_valid_str_lower = doc_original_str.lower()

            trgs_tokens = set()
            for phrase in example.label:
                if len(phrase) < 1:
                    continue
                clean_phrase = ' '.join(phrase).strip('.').strip(',').strip('?').strip().lower()
                clean_phrase = remove_punc(clean_phrase)
                kp_occurance = [m.start() for m in re.finditer(clean_phrase, doc_valid_str_lower)]

                for str_idx in kp_occurance:
                    kp = doc_original_str[str_idx: str_idx + len(clean_phrase)]
                    trgs_tokens.add(kp)  # auto dedup because of set
                    # trgs_tokens.add("$ " + kp) # bpe id: 1629 , remove this for xlm-roberta

            if len(trgs_tokens) == 0:
                # this record has no positive label
                gg

            start_end_token_id_flatten = []
            for kp in list(trgs_tokens):
                kp_token = tokenizer.encode(kp, add_special_tokens=False)
                if len(kp_token) == 0:
                    # this kp doesn't exist as it
                    continue

                if 3 in kp_token:
                    # 3 is <unk> in roberta tokenizer
                    continue

                # if kp_token[0] == 1629:
                if kp_token[0] == 6:  # for xlm roberta, in which the impact of $ is greatly reduced
                    kp_token = kp_token[
                               1:]  # excluding first and last token, this is needed to prevent auto rstrip in encoding

                start_end_pos = find_partial_token(doc_tokens, kp_token)
                start_end_pos_ignore_first = find_partial_token(doc_tokens, kp_token, ignoreFirstTokenInMatch=True)

                start_end_pos = start_end_pos + start_end_pos_ignore_first
                print(start_end_pos_ignore_first)
                if len(start_end_pos) == 0:
                    continue

                start_end_token_id_flatten += start_end_pos

            if len(start_end_token_id_flatten) == 0:
                gg

            sorted_positions = sorted(start_end_token_id_flatten, key=lambda x: x[0])
            filter_positions = filter_overlap(sorted_positions)

            # convert label
            orig_label = ['O' for _ in range(len(doc_tokens))]
            for start_end in filter_positions:
                s = start_end[0]
                e = start_end[1] - 1  # exclusive

                if s == e:
                    orig_label[s] = 'U'

                elif (e - s) == 1:
                    orig_label[s] = 'B'
                    orig_label[e] = 'E'

                elif (e - s) >= 2:
                    orig_label[s] = 'B'
                    orig_label[e] = 'E'
                    for i in range(s + 1, e):
                        orig_label[i] = 'I'
                else:
                    logger.info('ERROR')
                    break

            label = [label_map[item] for item in orig_label]
        return label
    
    if set_type == "predict":
        batch_encoding = tokenizer.batch_encode_plus(
            [example.text_a for example in examples],
            max_length=max_length,
            # pad_to_max_length=True,
            truncation = True,
            pad_to_max_length = True,
            padding = 'max_length',
            return_offsets_mapping = True
        )
    else:
        batch_encoding = tokenizer.batch_encode_plus(
            [(example.text_a, example.text_b) for example in examples],
            max_length=max_length,
            # pad_to_max_length=True,
            truncation = True,
            pad_to_max_length = True,
            padding = 'max_length'
        )


    labels = [get_label(example, input_ids) for example, input_ids in zip(examples, batch_encoding["input_ids"])]
    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)
    return features


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
    ngram_ids = [4] if n == 1 else [1] + [2 for _ in range(n-2)] + [3]
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
            phrase_set[(i, i + 2)] += min(token_logits[i][1], token_logits[i+1][3])
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

def key_phrase_convert_example_to_features(example, tokenizer,
                                           max_length=512,
                                           task=None,
                                           label_list=None,
                                           label_map=None,
                                           output_mode=None,
                                           task_name="quantus", set_type = "dev"):

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

    if set_type == "predict":
        inputs = tokenizer.encode_plus(
            raw_text, example.text_b,
            max_length=max_length,
            # pad_to_max_length=True,
            truncation = True,
            pad_to_max_length = True,
            padding = 'max_length',
            return_offsets_mapping = True
        )
    else:
        inputs = tokenizer.encode_plus(
        raw_text, example.text_b,
        max_length=max_length,
        truncation=True,
        pad_to_max_length=True,
        padding='max_length'
        )

    if label_map == None:
        label_map = {label: i for i, label in enumerate(label_list)}
    label = None
    if set_type != "predict":
        doc_tokens = inputs["input_ids"]
        # print(len(doc_tokens))
        # +2 to handle starting and ending special token
        # if len(doc_tokens) > max_length + 2:
        #     doc_tokens = doc_tokens[:max_length + 1] + [doc_tokens[-1]]
        doc_original_str = tokenizer.decode(doc_tokens)
        doc_valid_str_lower = doc_original_str.lower()

        trgs_tokens = set()

        for phrase in example.label:
            if len(phrase) < 1:
                continue
            clean_phrase = ' '.join(phrase).strip('.').strip(',').strip('?').strip().lower()
            # clean_phrase = remove_punc(clean_phrase)
            kp_occurance = [m.start() for m in re.finditer(clean_phrase, doc_valid_str_lower)]

            for str_idx in kp_occurance:
                kp = doc_original_str[str_idx: str_idx + len(clean_phrase)]
                trgs_tokens.add(kp)  # auto dedup because of set
                # trgs_tokens.add("$ " + kp) # bpe id: 1629 , remove this for xlm-roberta

        if len(trgs_tokens) == 0:
            print("error in len(trgs_tokens)")
            exit(1)

        start_end_token_id_flatten = []
        for kp in list(trgs_tokens):
            kp_token = tokenizer.encode(kp, add_special_tokens=False)
            if len(kp_token) == 0:
                # this kp doesn't exist as it
                continue

            if 3 in kp_token:
                # 3 is <unk> in roberta tokenizer
                continue

            # if kp_token[0] == 1629:
            if kp_token[0] == 6:  # for xlm roberta, in which the impact of $ is greatly reduced
                kp_token = kp_token[1:]  # excluding first and last token, this is needed to prevent auto rstrip in encoding

            start_end_pos = find_partial_token(doc_tokens, kp_token)
            start_end_pos_ignore_first = find_partial_token(doc_tokens, kp_token, ignoreFirstTokenInMatch=True)

            start_end_pos = start_end_pos + start_end_pos_ignore_first

            if len(start_end_pos) == 0:
                continue

            start_end_token_id_flatten += start_end_pos

        if len(start_end_token_id_flatten) == 0:
            # print(333333)
            gg

        sorted_positions = sorted(start_end_token_id_flatten, key=lambda x: x[0])
        filter_positions = filter_overlap(sorted_positions)

        # convert label
        orig_label = ['O' for _ in range(len(doc_tokens))]
        for start_end in filter_positions:
            s = start_end[0]
            e = start_end[1] - 1  # exclusive

            if s == e:
                orig_label[s] = 'U'

            elif (e - s) == 1:
                orig_label[s] = 'B'
                orig_label[e] = 'E'

            elif (e - s) >= 2:
                orig_label[s] = 'B'
                orig_label[e] = 'E'
                for i in range(s + 1, e):
                    orig_label[i] = 'I'
            else:
                logger.info('ERROR')
                break

        label = [label_map[item] for item in orig_label]
        assert len(doc_tokens) == len(label)
    # inputs["attention_mask"] = [1 if i < (len(inputs["input_ids"]) - inputs["input_ids"].count(PAD_TOKEN_ID)) else 0 for i in range(len(inputs["input_ids"]))]
    feature = InputFeatures(**inputs, label=label)
    return feature

def add_preprocess_opts(parser):
    
    parser.add_argument("--query_index", type=int, default=1)
    parser.add_argument("--passage_index", type=int, default=-1)
    parser.add_argument("--label_index", type=int, default=3)


if __name__ == "__main__":
    from transformers import XLMRobertaTokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained("/data/shichen/model/xlm_new_base/")
    parser = argparse.ArgumentParser()
    add_preprocess_opts(parser)
    args = parser.parse_args()
    print(args)
    label_list = ['O','B','I','E','U']
    label_map = {label: i for i, label in enumerate(label_list)}
    examples = list()
    processor = KeyPhraseProcessor(args)
    error_count = 0

    fw = open("/data/shichen/data/openkp/openkpgdi_test_features_xlmroberta_seq512.tsv", 'w', encoding='utf8')
    with open("/data/shichen/data/openkp/openkpgdi_test.tsv", 'r', encoding='utf8') as f:
        while True:
            line = f.readline()
            if not line:
                break

            try:
                line = line.strip()
                line_dict = processor.parse_line(line)
                cur_example = InputExample(**line_dict)
                data = key_phrase_convert_example_to_features(cur_example,
                                                            tokenizer,
                                                            max_length=512,
                                                            task=None,
                                                            label_list=label_list,
                                                            label_map=label_map,
                                                            output_mode="classification",
                                                            task_name="key_phrase", set_type = "train")
                data = {
                    "input_ids": data.input_ids,
                    "attention_mask": data.attention_mask,
                    "label": data.label
                }
                fw.write(json.dumps(data))
                fw.write('\n')
            except Exception as e:
                error_count += 1
            
    print(f"error count:{error_count}")










