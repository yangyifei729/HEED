import torch
import collections
from typing import Tuple
from copy import deepcopy
from random import random, randrange, randint, shuffle, choice
# from torch.utils.data import Dataset
# from transformer_utils.data_utils import DataProcessor, InputFeatures, file_iter, file_iter_multi_task
# from transformer_utils.tasks.glue import glue_convert_example_to_features
# from transformer_utils.data_utils import DataProcessor, InputExample, InputFeatures,file_iter,file_iter_multi_task
import transformers
from transformer_utils.tasks.glue import *
from transformer_utils.tasks.keyphrase import *
from transformer_utils.tasks.highlight import HighlightExample, highlight_convert_examples_to_features, HighlightProcessor
import numpy as np
import logging

logger = logging.getLogger(__name__)
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
        self.fin_corpus = file_iter_multi_task(self.corpus_path, self.encoding, world_size=self.world_size,
                                               global_rank=self.global_rank, endless=True)
        # else:
        #     self.fin_corpus = file_iter(self.corpus_path, self.encoding, world_size=self.world_size,
        #                                            global_rank=self.global_rank)

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
        line_dict = self.lazy_load_processor(line)
        try:
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
            else:
                cur_example = InputExample(**line_dict)
                cur_example.guid = self.sample_counter
                cur_features = glue_convert_example_to_features(cur_example, self.tokenizer,
                                                         max_length=self.args.max_seq_length,
                                                         task=None,
                                                         label_list=self.label_list,
                                                         label_map = self.label_map,
                                                         output_mode=self.output_mode,
                                                         task_name=self.task_name, set_type="train")

        except:
            return self.__getitem__(item)

        cur_tensor = self.convert_feature_to_tensor(cur_features)

        self.sample_counter += 1
        return cur_tensor

    def convert_feature_to_tensor(self, cur_feature):
        cur_tensor = dict()
        for key, value in vars(cur_feature).items():
            if key in ("input_ids", "attention_mask", "token_type_ids") and value is not None and not isinstance(value, str):
                cur_tensor[key] = torch.tensor(value, dtype=torch.long)
        if self.output_mode == "classification":
            if "mrc" in self.task_name:
                cur_tensor["labels"] = (torch.tensor(cur_feature.start_position, dtype=torch.long),
                                        torch.tensor(cur_feature.end_position, dtype=torch.long))
            elif "mlm" in self.task_name:
                if self.args.whole_word_mask:
                    inputs, labels = self.whole_word_mask_tokens(
                        cur_feature.input_ids)
                else:
                    inputs, labels = self.mask_one_example_tokens(cur_tensor)
                cur_tensor = {"input_ids": inputs,
                              "labels": labels}
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



    def whole_word_mask_tokens(self, input_ids): ##list input list output
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

