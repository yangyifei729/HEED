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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function
import os
import sched

from torch import load
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import collections
import argparse
import glob
import random
import json
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(filename)s %(funcName)s %(lineno)d: %(message)s')
import configparser
import numpy as np
import datetime
from typing import Union, Any, Dict, Optional, List
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformer_utils.modeling_moe import *

from transformers import AdamW, get_linear_schedule_with_warmup

from transformer_utils.data_utils import logging_set, kd_loss
from transformer_utils.metrics import glue_compute_metrics as compute_metrics
from transformer_utils.tasks.glue import BERTDataset
from transformer_utils.tasks.glue import glue_output_modes as output_modes
from transformer_utils.tasks.glue import glue_processors as processors
from transformer_utils.tasks.glue import glue_convert_examples_to_features as convert_examples_to_features
from transformer_utils.tasks.glue import EE_convert_example_to_features
from transformer_utils.tasks.highlight import highlight_convert_examples_to_features

from utils.utils_adversarial import FGSM
from utils.utils_other import get_config_section
from transformer_utils.data_utils_inference import LabelingDataset_mt, get_eval_dataloader


from deepEE_eval.deepEE_eval_moe1 import inference_mt, get_real_label_mt, get_real_result_from_real_preds_add_mt, \
    trans_conll_file_to_overlap_file, change_to_bio, get_metric, min_F1

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(filename)s %(lineno)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
# logger = logging.get_logger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassificationMT, BertTokenizer),
    # 'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassificationMT, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassificationMT, RobertaTokenizerFast),
    # 'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    # 'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    'xlmroberta': (XLMRobertaConfig, XLMRobertaForSequenceClassificationMT, XLMRobertaTokenizerFast),

    'robertatextcnn': (RobertaConfig, RobertaTextCNNForSequenceClassificationMT, RobertaTokenizer),
    'xlmrobertatextcnn': (XLMRobertaConfig, XLMRobertaTextCNNForSequenceClassificationMT, XLMRobertaTokenizer),

    'robertatextlstm': (RobertaConfig, RobertaLSTMForSequenceClassificationMT, RobertaTokenizer),
    'xlmrobertalstm': (XLMRobertaConfig, XLMRobertaLSTMForSequenceClassificationMT, XLMRobertaTokenizer),

    'robertadpcnn': (RobertaConfig, RobertaDPCNNForSequenceClassificationMT, RobertaTokenizer),
    'xlmrobertadpcnn': (XLMRobertaConfig, XLMRobertaDPCNNForSequenceClassificationMT, XLMRobertaTokenizer),
    'deberta': (DebertaConfig, DebertaForSequenceClassificationMT, DebertaTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train_mt(args, model, tokenizer, model_teacher=None, config=None):
    """  Train the model. """
    # global_rank = rank, number for each gpu, for machine0 0-7 gpu is rank 0-7
    # local rank, number on a specific machine, local rank 0-7 for machine0 local rank 0-7 for machine1
    # print(args.gloabl_rank)

    print(args.global_rank)

    if args.global_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.logging_dir)
    
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    #### Begin for multi-task
    # since we have multi-file for one task, thus we have list of list dataloader
    train_dataloader_list_of_list = []
    for index, task_name in enumerate(args.task_name_list):
        # load train dataset
        train_dataloader_list = []
        task_name_train_files = args.train_file_list[index]
        for index_train_files, task_one_train_file in enumerate(task_name_train_files):
            train_dataset = load_and_cache_examples(args, index, task_name, task_one_train_file, 
                                                args.dev_file_list[index], tokenizer, evaluate=False, lazy_load_block_size=args.lazy_load_block_size)
            


            if args.all_in_memory:
                train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
            else:
                train_sampler = None # None sampler, if your dataset is not large, I recommend to use all_in_memory

            try:
                train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                    num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
            except:
                train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                            num_workers=args.num_workers)
            train_dataloader_list.append(train_dataloader)
            logger.info("  task_name = {}, file idx {}".format(task_name, index_train_files))
            logger.info("  task file name: {}".format(task_one_train_file))
            logger.info("  Num examples = %d", (len(train_dataset)))
        
        train_dataloader_list_of_list.append(train_dataloader_list)

    # train_iter_list = [iter(dl) for dl in train_dataloader_list]
    train_iter_list_of_list = [[iter(dl) for dl in train_dataloader_list] for train_dataloader_list in train_dataloader_list_of_list]

    ### END add for multi-task
    # this part can be fixed in future
    if args.max_steps > 0 and args.multi_task:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader_list[0]) // args.gradient_accumulation_steps
        ) + 1 # may change logic here, why only dataloader[0]
        t_one_epoch = len(train_dataloader_list[0]) // args.gradient_accumulation_steps
        print(t_one_epoch)
    else:
        t_one_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        print("Calculate total steps for multi-tasks")
        t_total = sum([len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs for \
            train_dataloader in train_dataloader_list for train_dataloader_list in train_dataloader_list_of_list])
    
    ### Begin add for multi-task
    # assert args.max_steps > 0
    step_optimize_total = t_total
    ### End add for multi-task
    
    if args.special_tokens != None:
        special_tokens = args.special_tokens.split(",")
        SPECIAL_TOKENS_DICT = {'additional_special_tokens': special_tokens}
        tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    
    model.resize_token_embeddings(len(tokenizer))

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=step_optimize_total)
    if (
            args.model_name_or_path is not None
            and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"), map_location=args.device)
        )
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    
    if args.continue_train and (
        args.output_dir is not None 
        and os.path.isfile(os.path.join(args.output_dir, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.output_dir, "scheduler.pt"))
    ):
        optimizer.load_state_dict(torch.load(os.path.join(args.output_dir, "optimizer.pt"), map_location=args.device))
        scheduler.load_state_dict(torch.load(os.path.join(args.output_dir, "scheduler.pt")))
    
    if args.use_deepscale:
        # DeepScale init and wrap model/optimizer
        model, optimizer, _, _ = deepscale.initialize(args=args,
                                                      model=model,
                                                      optimizer=optimizer,
                                                      lr_scheduler=scheduler,
                                                      dist_init_required=False)
    else:
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        
        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        
        # Distributed training (should be after apex fp16 initialization)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                                output_device=args.local_rank,
                                                                find_unused_parameters=True)
    
    # adversarial_training
    if args.adv_training == "fgm":
        # make changes to word_embedding
        adv = FGSM(model=model, param_name="word_embeddings")

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Current device global rank = %d", (-1 if args.local_rank == -1 else torch.distributed.get_rank()))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", step_optimize_total)

    global_step = 0
    pre_step = 0
    tr_loss, logging_loss, loss_dict, logging_loss_dict = 0.0, 0.0, collections.defaultdict(int), collections.defaultdict(int)

    if args.continue_train and os.path.isfile(os.path.join(args.output_dir, "step_dict.json")):
        with open(os.path.join(args.output_dir, "step_dict.json"), 'r') as f:
            step_dict = json.loads(f.read())
        global_step, pre_step, tr_loss = step_dict["global_step"], step_dict["step"], step_dict["tr_loss"]
    
    model.zero_grad()
    if args.global_rank in [-1, 0]:
        print(model)
    set_seed(args) # Added here for reproductibility (even between python 2 and 3)

    sample_probs_for_task = [i / sum(args.train_strategy_task_list) for i in args.train_strategy_task_list] # sample some data from each task, with some probability
    sample_probs_for_task_list_of_list = [[i / sum(x) for i in x] for x in args.train_strategy_train_file_list]
    
    accum_cnt = 0
    previous_sample_idx = -1

    for step in trange(pre_step, int(step_optimize_total)):
        if not args.multi_task_gradacc:
            # use this if back prop once feed data to model and calculate the loss
            sample_idx = np.random.choice(len(sample_probs_for_task), p=sample_probs_for_task)
        else:
            sample_idx = previous_sample_idx if accum_cnt > 0 else np.random.choice(len(sample_probs_for_task), p=sample_probs_for_task)
        
        # logger.info(f"Select task idx {sample_idx} where the task name is {args.task_name_list[sample_idx]}")

        selected_task_files = train_iter_list_of_list[sample_idx]
        selected_task_train_strategy = sample_probs_for_task_list_of_list[sample_idx]
        sampled_idx_for_specific_task = np.random.choice(len(selected_task_train_strategy), p=selected_task_train_strategy) 
        final_train_file_selected = selected_task_files[sampled_idx_for_specific_task]

        try:
            batch = next(final_train_file_selected)
        except StopIteration:
            if args.local_rank in [-1, 0]:
                logger.info(f"The iter {sampled_idx_for_specific_task} and out iter {sample_idx} in train_iter_list_of_list exhausted. Iter it again. ")
            train_iter_list_of_list[sample_idx][sampled_idx_for_specific_task] = iter(train_dataloader_list_of_list[sample_idx][sampled_idx_for_specific_task])
            batch = next(train_iter_list_of_list[sample_idx][sampled_idx_for_specific_task])
            
        inputs = dict()
        for k, v in batch.items():
            if isinstance(v, list):
                new_list = []
                for value in v:
                    if isinstance(value, list):
                        new_list.append([t.to(args.device) for t in value])
                    else:
                        new_list.append(value.to(args.device))
                inputs[k] = new_list
            elif isinstance(v, tuple):
                new_list = [value.to(args.device) for value in v]
                inputs[k] = tuple(new_list)
            else:
                inputs[k] = v.to(args.device)
        inputs["task_index"] = sample_idx
        # 这个task_index 决定了model 用哪个 classifier head
        if args.model_type in ["xlmroberta", "xlm", "roberta", "distilbert", "camembert"]:
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]

        model.train()

        if args.distillation:   # False
            model_teacher.eval()
            outputs_teacher = model_teacher(**inputs)
            loss_original_teacher, logits_teacher = outputs_teacher[0], outputs_teacher[1]
            inputs["logits"] = logits_teacher
            # print("logits teacher:", logits_teacher)
            outputs = model(**inputs)
            loss, logits = outputs[0], outputs[1] # model outputs are always tuple in pytorch-transformers (see doc)
        else:
            outputs = model(**inputs)
            loss, logits = outputs[0], outputs[1] # model outputs are always tuple in pytorch-transformers (see doc)

        if args.use_deepscale:
            model.backward(loss)
            model.step()
        else:
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.training_loss_scale:
                loss = loss * args.training_loss_scale[sample_idx][sampled_idx_for_specific_task]
            
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
        
        tr_loss += loss.item()
        step_dict = {"step": step, "global_step": global_step, "tr_loss": tr_loss}
        loss_dict[f"loss_head_{sample_idx}_task_{args.task_name_list[sample_idx]}"] += (
            loss.item() / args.training_loss_scale[sample_idx][sampled_idx_for_specific_task]
        )

        if accum_cnt == args.gradient_accumulation_steps - 1:
            if not args.use_deepscale:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
            loss_dict[f'global_step_head_{sample_idx}_task_{args.task_name_list[sample_idx]}'] += 1
            global_step += 1
        accum_cnt = (accum_cnt + 1) % args.gradient_accumulation_steps
        if args.global_rank in [-1, 0] and args.logging_steps > 0 and global_step > 1 and step % (
                args.logging_steps * args.gradient_accumulation_steps) == 0:
            logs = {}
            loss_scalar = (tr_loss - logging_loss) / args.logging_steps
            learning_rate_scalar = scheduler.get_lr()[0]
            logs['learning_rate'] = learning_rate_scalar
            logs['loss_all_{}_task_{}'.format(sample_idx, args.task_name_list[sample_idx])] = loss_scalar
            for i in range(len(args.task_name_list)):
                logs[f'loss_head_{i}_task_{args.task_name_list[i]}'] = (loss_dict[f'loss_head_{i}_task_{args.task_name_list[i]}'] -
                                                                        logging_loss_dict[f'loss_head_{i}_task_{args.task_name_list[i]}']) / (
                                                                                   loss_dict[f'global_step_head_{i}_task_{args.task_name_list[i]}'] -
                                                                                   logging_loss_dict[f'global_step_head_{i}_task_{args.task_name_list[i]}'] + 0.000001)
            logging_loss = tr_loss
            logging_loss_dict = loss_dict.copy()

            for key, value in logs.items():
                tb_writer.add_scalar(key, value, global_step)
            logger.info(json.dumps(
                {**logs, **{'global_step': global_step, "local_step": step, "total_step": step_optimize_total}}))
            
        
        if args.global_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            # tokenizer.save_pretrained(args.output_dir)
            torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(args.output_dir, "scheduler.pt"))
            step_dict_str = json.dumps(step_dict)
            with open(os.path.join(args.output_dir, "step_dict.json"), 'w') as f:
                f.write(step_dict_str)
        

            ######## save output ########
        previous_sample_idx = sample_idx
    
    # save
    torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(args.output_dir, "scheduler.pt"))
    step_dict_str = json.dumps(step_dict)
    with open(os.path.join(args.output_dir, "step_dict.json"), 'w') as f:
        f.write(step_dict_str)
    
    if not args.use_deepscale:
        if args.local_rank in [-1, 0]:
            tb_writer.close()
    
    return global_step, tr_loss / global_step

def load_and_cache_examples(args, index, task, train_file, dev_file, tokenizer, evaluate=False, lazy_load_block_size=1000000):
    args.currency_list = None
    args.currency_ids = None
    if args.data_augmentation and args.currency is not None:
        currency_list = np.load(args.currency, allow_pickle=True).tolist()
        if args.old_currency != "all":
            old_currency_v = args.old_currency.split(',')
            for key in list(currency_list.keys()):
                if key not in old_currency_v and currency_list[key] not in old_currency_v:
                    currency_list.pop(key)
        for key in list(currency_list.keys()):
            if currency_list.__contains__(currency_list[key]) is False:
                currency_list[currency_list[key]] = key
        currency_ids = {}
        for key in currency_list:
            key_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(key))
            print(key, tokenizer.tokenize(key), tokenizer.convert_tokens_to_ids(tokenizer.tokenize(key)),
                  currency_list[key], tokenizer.tokenize(currency_list[key]),
                  tokenizer.convert_tokens_to_ids(tokenizer.tokenize(currency_list[key])))
            currency_ids[key] = key_ids
            if currency_ids.__contains__(key_ids[0]):
                if (-1 if len(key_ids) == 1 else key_ids[1:]) not in currency_ids[key_ids[0]]:
                    currency_ids[key_ids[0]].append(-1 if len(key_ids) == 1 else key_ids[1:])
            else:
                currency_ids[key_ids[0]] = [-1] if len(key_ids) == 1 else [key_ids[1:]]
        args.currency_list = currency_list
        args.currency_ids = currency_ids
        new_currency_ = []
        for curr in args.new_currency.split(','):
            if currency_list.__contains__(curr):
                if curr not in new_currency_:
                    new_currency_.append(curr)
                if currency_list.__contains__(curr) and currency_list[curr] not in new_currency_:
                    new_currency_.append(currency_list[curr])
        args.new_currency_list = new_currency_

    if not args.use_deepscale:
        if args.local_rank not in [-1, 0] and not evaluate:
            torch.distributed.barrier() # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # can be replaced by yaml/config file in future, temp for now
    processor = processors[task](args)
    output_mode = output_modes[task]

    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels(label_file=args.topic_vocab_list[index])
    lazy_load_processor = processor.parse_line
    if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta', 'xlmroberta']:
        # HACK(label indices are swapped in RoBERTa pretrained model)
        label_list[1], label_list[2] = label_list[2], label_list[1]
    
    if evaluate == True:
        examples = processor.get_dev_examples(args.data_dir, dev_file)
        if "mrc" in task:
            features = highlight_convert_examples_to_features(examples, tokenizer, args.max_seq_length,
                                                              args.doc_stride, args.max_query_length,
                                                              is_training=False)
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_example_index)
        else:
            features = convert_examples_to_features(examples, tokenizer, label_list=label_list,
                                                    max_length=args.max_seq_length, output_mode=output_mode,
                                                    # pad on the left for xlnet
                                                    task_name=task, set_type="dev")
        
        if not args.use_deepscale:
            if args.local_rank == 0 and not evaluate:
                torch.distributed.barrier() # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        if output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
        if features[0].token_type_ids != None:
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels, all_token_type_ids)
        else:
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)
    else:
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        world_size, global_rank = (1, 0) if args.local_rank == -1 else (
            torch.distributed.get_world_size(), torch.distributed.get_rank()
        )


        dataset = BERTDataset(args, os.path.join(args.data_dir, train_file),
                            tokenizer, seq_len=args.max_seq_length,
                            label_list=label_list, lazy_load_processor=lazy_load_processor,
                            corpus_lines=None, on_memory=args.all_in_memory, task_name=task,
                            world_size=world_size, global_rank=global_rank,
                            lazy_load_block_size=args.lazy_load_block_size
        )
    
    if not args.use_deepscale:
        if args.local_rank == 0 and not evaluate:
            torch.distributed.barrier() # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    return dataset


def task_head_mapping(multi_task_head_mapping, model_path):
    mt_head_mapping = {}
    for mapping in multi_task_head_mapping.split(','):
        new_head, old_head = mapping.split(':')
        if new_head != old_head:
            mt_head_mapping[new_head.strip()] = old_head.strip().lower()
    
    state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location='cpu')
    logger.info("~~~~~~~~~~~~~~~~~before_convert~~~~~~~~~~~~~~~~~")
    for key, value in state_dict.items():
        if key.startswith("classifier_heads") and key.endswith("out_proj.weight"):
            logger.info("key name: {}, value size: {}".format(key, value.size()))
    
    state_dict_copy = state_dict.copy()

    if len(mt_head_mapping) > 0:
        weight_to_change = ['classifier_heads.{}.dense.weight', 'classifier_heads.{}.dense.bias',
                            'classifier_heads.{}.out_proj.weight', 'classifier_heads.{}.out_proj.bias']
        for new_key in mt_head_mapping:
            for tpl in weight_to_change:
                new_key_complete = tpl.format(new_key)
                old_key_split = mt_head_mapping[new_key].split('_')
                if len(old_key_split) == 2:
                    old_key, mode = old_key_split[0], old_key_split[1]
                else:
                    old_key, mode = old_key_split[0], None
                old_key_complete = tpl.format(old_key)

                if mt_head_mapping[new_key] == "-1":
                    del state_dict[new_key_complete]
                    logger.info('Delete state_dict[%s]' % (new_key_complete))
                else:
                    if not ('out_proj' in old_key_complete and mode == 'half'):
                        state_dict[new_key_complete] = state_dict_copy[old_key_complete].clone()
                        logger.info('Replace state_dict[%s] with state_dict[%s]' % (new_key_complete, old_key_complete))
                    else:
                        del state_dict[new_key_complete]
                        logger.info('Delete state_dict[%s]' % (new_key_complete))
    logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~after_convert~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for key, value in state_dict.items():
        if key.startswith("classifier_heads") and key.endswith("out_proj.weight"):
            logger.info("key name: {}, value size: {}".format(key, value.size()))
    return state_dict
    

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--mt_training_config_path", default=None, type=str, 
                        help="The config file for multitask training, contains train file and other parameters")
    # parser.add_argument("--task_names", default=None, type=str, required=True,
    #                     help="The name of the task to train selected in the list: " + ", ".join(processors.keys()) +
    #                         "for multi-task, provide a task list. eg: quantuskd,rankerbert")
    # parser.add_argument("--distill", action='store_true',
    #                 help="train distillation")
    # parser.add_argument("--temperature", default="5", type=str,
    #                     help="distill temperature")
    # parser.add_argument("--training_distill_loss_scale", default=None, type=str,
    #                     help="training distill loss scale")
    parser.add_argument("--model_name_or_path_teacher", default=None, type=str,
                        help="Path to pre-trained teacher model")
    parser.add_argument("--teacher_model_config_path", default="", type=str,
                    help="config name for teacher model")

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--query_index",
                        default=0,
                        type=int,
                        help="query_index")
    parser.add_argument("--passage_index",
                        default=1,
                        type=int,
                        help="passage_index")
    parser.add_argument("--label_index",
                        default=2,
                        type=int,
                        help="label_index")
    parser.add_argument("--language_index",
                        default=-1,
                        type=int,
                        help="label_index")
    parser.add_argument("--logging_dir", default=None, type=str,
                        help="the dir of logging saved.")

    ## Other parameters
    parser.add_argument("--encoder_config_name", default="", type=str,
                    help="config name for xmlee encoder")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=100, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                            "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--all_in_memory", action='store_true',
                        help="Whether to load all data in memory.")

    parser.add_argument("--freeze_encoder", action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--lazy_load_block_size", default=1000000, type=int,
                        help="On lazy load mode, each time the data loader would read N samples into memory")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    # parser.add_argument("--train_file",
    #                     default=None,
    #                     type=str,
    #                     help="The train file name. for multi-task, provide a train file list. eg: t1.tsv,t2.tsv,t3.tsv")
    parser.add_argument("--dev_file",
                        default=None,
                        type=str,
                        help="The dev file name. for multi-task, provide a dev file list. eg: t1.tsv,t2.tsv,t3.tsv")
    # delete this due to we have used config file for multi-task training
    # parser.add_argument("--topic_vocab",
    #                     default=None,
    #                     type=str,
    #                     help="The train file name.")

    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--enable_trt", action='store_true',
                        help="use optimized version for DLIS")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")

    parser.add_argument('--DLIS_mode', action='store_true',
                        help="Whether to use DLIS_mode to output predictions")

    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                            "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--use_deepscale', action='store_true',
                        help="Whether to use deepscale")
    parser.add_argument("--global_rank", type=int, default=0,
                        help="For distributed training: global_rank")
    parser.add_argument("--world_size", type=int, default=1,
                        help="For distributed training: world_size")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="default num_workers")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--log_path', type=str, default=None, help="For distant debugging.")

    #### Begin add args for adversial training or model robustness
    parser.add_argument("--R_dropout", action='store_true',
                        help="Whether to add R-dropout tech to the model training, where model pass the same input x two times, and \
                        add kl loss to the model output logits as regulizer.")
    parser.add_argument("--R_dropout_alpha", type=float, default=0.0, help="hyper-parameters to control the kl-loss contribution, R-dropout loss function is \
                        ce_loss + \alpha kl_loss")
    parser.add_argument("--adv_training", default=None, choices=["fgm"], help="fgm adversial training")


    #### Begin add args for multi-task
    # parser.add_argument("--train_strategy",
    #                     default=None,
    #                     type=str, required=True,
    #                     help="The train strategies among training data, integers separated by comma. eg: '1,1'")
    parser.add_argument("--multi_task", action='store_true', help="Set this flag if you are using multiple task training.")
    parser.add_argument("--multi_task_gradacc", action='store_true', default=False,
                        help="Let N be the gradient accumulation step. If True: train only 1 task for N steps, then update model weight;If False, train multiple tasks for N steps totally, then update the model weight")
    parser.add_argument("--multi_task_head_mapping", default=None,
                        help="E.g., If pretrained weight has 2 heads (regression, classification), current training has 3 head (classification, regression, regression), 0:1,1:0,2:0 means: new head 0 will load weight from pretrained head 1, new head 1 will load weight from pretrained head 0, new head 2 will load from pretrained head 0 as well. Also, you can load partial weight from regression head to classification head, vise versa. To do that, just add an the mapping should be: 0:0_half,1:0,2:0; If you want to initalize a new head from scratch, just set new head as -1. E.g. 0:1,1:-1,2:0 means the new head 1 will not load from any pretrained head")
    parser.add_argument("--drop_fc", action='store_true', help="drop classifier layer")

    # parser.add_argument("--training_loss_scale", default=None, type=str,
    #                     help="The scale separated by comma for every task loss for backward.")

    #### Begin add args for multi-mrc task
    parser.add_argument("--version_2_with_negative", action='store_true',
                        help="If true, the SQuAD examples contain some that do not have an answer.")
    parser.add_argument("--doc_stride", default=32, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=20, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will be truncated to this length.")
    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    parser.add_argument("--evaluate_mrc_training", action='store_true',
                        help="If true, the SQuAD examples contain some that do not have an answer.")
    parser.add_argument("--score_index", default=None, type=str, help="Use which head to do predict")
    parser.add_argument("--aether_mode", action='store_true',
                        help="Whether run on aether")
    parser.add_argument("--write_split_score", action='store_true',
                        help="Whether to do write_split_score on predict output.")
    parser.add_argument("--write_class1_prob", action='store_true',
                        help="Works when write_split_score is True. For mts, write single score; For mtt, write probability of class 1")
    parser.add_argument("--predict_file", default="None", type=str, help="The predict_file name.")
    parser.add_argument("--use_cached_squad_format_data", action='store_true',
                        help="If true, will use the cached squad format data with name $predict_file + squad.json")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.")
    parser.add_argument("--output_file", default=None, type=str,
                        help='predict file name; If given, the output file would be os.path.join(output_dir, output_file); If not given, will follow old logic: os.path.join(args.output_dir, "{}_prediction.tsv".format(os.path.basename(args.predict_file))')
    parser.add_argument("--get_ms_metric", action='store_true', help="evaluate and get ms metric")
    parser.add_argument("--dense_mrc_head", action='store_true', help="create mrc heads with one dense layer")
    # add for mlm
    parser.add_argument("--mlm_probability", default=0.15, type=float,
                        help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument("--whole_word_mask", action='store_true',
                        help="Whether to do whole word masking.")
    parser.add_argument("--pooler", default="first", type=str, help="pooling method for full connection")
    parser.add_argument("--random_mask", action='store_true', help="whether random mask")

    # add for special token support
    parser.add_argument("--special_tokens",
                        default=None,
                        type=str,
                        help="The train strategies among training data, integers separated by comma. eg: '1,1'")
    parser.add_argument("--enable_transformer_layer", action='store_true', help="whether to enble transformer layer when using LSTM and CNN")
    parser.add_argument("--save_optimizer", action='store_true', help="whether to save optimizer")
    parser.add_argument("--continue_train", action='store_true', help="whether to continue_train")

    # add small embedding head
    parser.add_argument("--use_embedding_head", action='store_true', help="whether to use small embedding")
    parser.add_argument("--embedding_dim", default=64, type=int,
                        help="small embedding size")
    parser.add_argument("--hidden_size", default=768, type=int, help="small hidden size")
    
    # add multiple experts args
    parser.add_argument("--experts_per_modal", default=1, type=int, help="number of cls per modal")


    # xml_ee
    parser.add_argument("--max_part_size", type=int, default=10)
    parser.add_argument("--equal_size", type=int, default=2)
    parser.add_argument("--max_bounding_box_size", type=int, default=200)
    parser.add_argument("--max_color_size", type=int, default=100)
    parser.add_argument("--max_FontWeight_size", type=int, default=100)
    parser.add_argument("--max_FontSize_size", type=int, default=100)
    parser.add_argument("--num_hidden_layers", type=int, default=6)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--intermediate_size", type=int, default=12)
    parser.add_argument("--output_hidden_states", action='store_true', help="whether to output hidden states")
    parser.add_argument("--drop_hard_label", action='store_true', help="whether to drop hard label")
    parser.add_argument("--temperature_ratio", default=1.0, type=float, help="Ratio of temperature")
    parser.add_argument("--crossentropy_loss_ratio", default=0.5, type=float, help="Ratio of temperature")
    parser.add_argument("--kl_loss_ratio", default=100.0, type=float, help="Ratio of temperature")
    parser.add_argument("--mse_loss_ratio", default=0.001, type=float, help="Ratio of temperature")
    parser.add_argument("--multiple", default=1, type=int, help="The number of times dataloader-0 runs before dataloader-1 runs")
    parser.add_argument("--multi_train_file", action='store_true', help="multi train file on 1 task")
    parser.add_argument("--incomplete_entity", default="MainImage,Price,Name", type=str, help="entities that need to be completed")
    parser.add_argument("--use_teacher_entity", default="Manufacturer,NumberofReviews", type=str, help="entities that use teacher labels")
    parser.add_argument("--only_full_UHRS", action='store_true', help="only_full_UHRS")
    parser.add_argument("--distillation", action='store_true', help="train distillation")
    parser.add_argument("--data_augmentation", action='store_true', help="data argmenatation for currency")
    parser.add_argument("--currency", type=str, default=None, help="")
    parser.add_argument("--old_currency", type=str, default="USD,CNY,CHF,JPY,£,$,¥,₣,₴,€,円,EUR", help="")
    parser.add_argument("--new_currency", type=str, default="CNY,CHF,JPY", help="")
    parser.add_argument("--ratio_currency", default=0.2, type=float, help="Ratio of currency")
    parser.add_argument("--overlap", action='store_true', help="wether use overlap")
    parser.add_argument("--overlap_size", default=50, type=int, help="The number of overlap_size")
    parser.add_argument("--teacher_path", default="EE_teacher_checkpoint", type=str, help="Teacher model path")
    parser.add_argument("--mask_boundingbox", action='store_true', help="wether to mask bounding box")
    parser.add_argument("--abandon_visual_features", action='store_true', help="wether use visual_features")
    parser.add_argument("--inference_mt_all_tasks", action='store_true', help="wether inference on all taskss")


    args, _ = parser.parse_known_args()
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        print("~~~~~~~~~~~in the nccl init~~~~~~~~~~~~~~~~")
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
        world_size, global_rank = (1, 0) if args.local_rank == -1 else (
            torch.distributed.get_world_size(), torch.distributed.get_rank())
        args.global_rank = global_rank
        args.world_size = world_size
    
    args.device = device
    args.is_master = args.local_rank == -1 or torch.distributed.get_rank() == 0

    if args.log_path:
        if not os.path.exists(os.path.dirname(args.log_path)):
            os.makedirs(os.path.dirname(args.log_path))
        log_path = args.log_path + ('' if args.local_rank == -1 else '.rank%d' % args.local_rank)
    else:
        log_path = None
    if args.is_master:
        logging_set(log_path, logging.INFO)
    else:
        logging_set(log_path, logging.WARN)
    
    logger.info("Print the args:")
    for key, value in sorted(args.__dict__.items()):
        logger.info("{} = {}".format(key, value))
    
    # Setup logging
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s", 
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    
    # Set seed
    set_seed(args)

    ### BEGIN add for multi-task training config file loading
    config = configparser.RawConfigParser()
    # print(os.path.exists(args.mt_training_config_path))
    config.read(args.mt_training_config_path)
    config_dict = get_config_section(config)

    # set task name one by one
    args.task_name_list = []
    args.topic_vocab_list = []
    args.train_file_list = []
    args.train_strategy_list = []
    args.training_loss_scale = []
    for key, value in config_dict.items():
        args.task_name_list.append(value.get("task_name", None)) # Done
        args.topic_vocab_list.append(value.get("topic_vocab", None)) # Done
        
        # args.tran_file_list can be list of list, every task_name with multiple training files.
        task_related_training_files = []
        for one_train_file in value["train_file"].split(","):
            task_related_training_files.append(one_train_file.strip())
        args.train_file_list.append(task_related_training_files)

        # args.dev_file_list can be list of list, every task_name with multiple dev files.
        # task_related_dev_files = []
        # for one_dev_file in value["dev_file"].split(","):
        #     task_related_dev_files.append(one_dev_file.strip())
        # args.dev_file_list.append(task_related_dev_files)
        
        # args.train_strategy_list can be list of list, every multiple training files with multiple training strategy.
        task_related_training_strategy_list = []
        for one_train_strategy in value["train_strategy"].split(','):
            task_related_training_strategy_list.append(one_train_strategy.strip())
        args.train_strategy_list.append(task_related_training_strategy_list)

        # args.training_loss_scale can be list of list, every train file with different training loss scale
        task_realted_training_loss_scale = []
        for one_train_scale in value["training_loss_scale"].split(','):
            task_realted_training_loss_scale.append(one_train_scale.strip())
        args.training_loss_scale.append(task_realted_training_loss_scale) # Done

    logger.info(f"Task name list: {args.task_name_list}")
    logger.info(f"Topic vocab list: {args.topic_vocab_list}")
    logger.info(f"Train file list: {args.train_file_list}")
    logger.info(f"Train strategy list: {args.train_strategy_list}")
    logger.info(f"Training loss scale: {args.training_loss_scale}")

    ### BEGIN add for multi-task
    # args.task_name_list = args.task_names.lower().split(",")
    for task_name in args.task_name_list:
        if task_name not in processors:
            raise ValueError("Task not found: %s" % (task_name))

    processor_list = [processors[task_name](args) for task_name in args.task_name_list]
    args.output_mode_list = [output_modes[task_name] for task_name in args.task_name_list]
    logger.info(f"processor_list: {processor_list}") # show processor list
    logger.info(f"output_mode_list: {args.output_mode_list}") # show output mode list
    args.output_mode = args.output_mode_list

    # can be set to None for ee cases, since no use of vocab
    # if args.topic_vocab == None:
    #     args.topic_vocab_list = [None] * len(args.task_name_list)
    # else:
    #     args.topic_vocab_list = args.topic_vocab.split(",")
    #     if len(args.topic_vocab_list) == 1:
    #         args.topic_vocab_list = args.topic_vocab_list * len(args.task_name_list)
    # here we need to use topic vocab, can be fixed in future.
    label_list_list = [processor.get_labels(topic_vocab) for topic_vocab, processor in 
                        zip(args.topic_vocab_list, processor_list)]
    
    
    if args.training_loss_scale:
        args.training_loss_scale = [[float(i) for i in x] for x in args.training_loss_scale] # list of list, each task with a specific training loss sclae
    
    num_labels_list = [len(label_list) for label_list in label_list_list]
    ### END add for multi-task

    if not args.use_deepscale:
        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier() # Make sure only the first process in distributed training will download model & vocab
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    ### BEGIN add for multi-task
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          finetuning_task="dummy",
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    encoder_config = config_class.from_pretrained(args.encoder_config_name, finetuning_task="dummy")
    ### END add for multi-task

    ##### Begin Add Additional args
    def add_config(config):
        config.distillation = args.distillation
        config.only_full_UHRS = args.only_full_UHRS
        config.use_teacher_entity = args.use_teacher_entity
        config.incomplete_entity = args.incomplete_entity
        config.label_list_list = label_list_list
        config.enable_trt = args.enable_trt
        config.output_mode = args.output_mode
        config.max_seq_length = args.max_seq_length
        config.enable_transformer_layer = args.enable_transformer_layer
        config.pooler = args.pooler
        config.use_embedding_head = args.use_embedding_head
        config.embedding_dim = args.embedding_dim
        config.finetuning_task_list = args.task_name_list
        config.output_mode_list = args.output_mode_list
        config.num_labels_list = num_labels_list
        config.dense_mrc_head = args.dense_mrc_head
        config.visual_feature_size = 22
        config.hidden_size = args.hidden_size
        config.num_hidden_layers = args.num_hidden_layers
        config.drop_hard_label = args.drop_hard_label

        # config for visual feature encoder
        encoder_config.equal_size = args.equal_size
        encoder_config.max_bounding_box_size = args.max_bounding_box_size
        encoder_config.max_color_size = args.max_color_size
        encoder_config.max_FontWeight_size = args.max_FontWeight_size
        encoder_config.max_FontSize_size = args.max_FontSize_size
        encoder_config.max_part_size = args.max_part_size
        config.equal_size = args.equal_size
        config.max_bounding_box_size = args.max_bounding_box_size
        config.max_color_size = args.max_color_size
        config.max_FontWeight_size = args.max_FontWeight_size
        config.max_FontSize_size = args.max_FontSize_size
        config.max_part_size = args.max_part_size
        config.num_hidden_layers = args.num_hidden_layers
        config.num_attention_heads = args.num_attention_heads
        config.intermediate_size = args.intermediate_size
        config.output_hidden_states = args.output_hidden_states
        config.temperature_ratio = args.temperature_ratio
        config.crossentropy_loss_ratio = args.crossentropy_loss_ratio
        config.kl_loss_ratio = args.kl_loss_ratio
        config.mse_loss_ratio = args.mse_loss_ratio
        config.R_dropout = args.R_dropout
        config.R_dropout_alpha = args.R_dropout_alpha
        config.abandon_visual_features = args.abandon_visual_features
        config.inference_mt_all_tasks = args.inference_mt_all_tasks
        # multiple experts args
        config.experts_per_modal = args.experts_per_modal


    def add_config_teacher(config):
        config.distillation = args.distillation
        config.only_full_UHRS = args.only_full_UHRS
        config.use_teacher_entity = args.use_teacher_entity
        config.incomplete_entity = args.incomplete_entity
        config.label_list_list = label_list_list
        config.enable_trt = args.enable_trt
        config.output_mode = args.output_mode
        config.max_seq_length = args.max_seq_length
        config.enable_transformer_layer = args.enable_transformer_layer
        config.pooler = args.pooler
        config.use_embedding_head = args.use_embedding_head
        config.embedding_dim = args.embedding_dim
        config.finetuning_task_list = args.task_name_list
        config.output_mode_list = args.output_mode_list
        config.num_labels_list = num_labels_list
        config.dense_mrc_head = args.dense_mrc_head
        config.visual_feature_size = 22
        config.drop_hard_label = args.drop_hard_label

        # config for visual feature encoder
        encoder_config.equal_size = args.equal_size
        encoder_config.max_bounding_box_size = args.max_bounding_box_size
        encoder_config.max_color_size = args.max_color_size
        encoder_config.max_FontWeight_size = args.max_FontWeight_size
        encoder_config.max_FontSize_size = args.max_FontSize_size
        encoder_config.max_part_size = args.max_part_size
        config.equal_size = args.equal_size
        config.max_bounding_box_size = args.max_bounding_box_size
        config.max_color_size = args.max_color_size
        config.max_FontWeight_size = args.max_FontWeight_size
        config.max_FontSize_size = args.max_FontSize_size
        config.max_part_size = args.max_part_size
        config.output_hidden_states = args.output_hidden_states
        config.temperature_ratio = args.temperature_ratio
        config.crossentropy_loss_ratio = args.crossentropy_loss_ratio
        config.kl_loss_ratio = args.kl_loss_ratio
        config.mse_loss_ratio = args.mse_loss_ratio
        config.R_dropout = args.R_dropout
        config.R_dropout_alpha = args.R_dropout_alpha
        config.abandon_visual_features = args.abandon_visual_features
        config.inference_mt_all_tasks = args.inference_mt_all_tasks


    def add_model_config(config): # for evaluation
        config.add_visual_features = True
        config.max_bounding_box_size = args.max_bounding_box_size
        config.max_color_size = args.max_color_size
        config.max_FontWeight_size = args.max_FontWeight_size
        config.max_FontSize_size = args.max_FontSize_size
        config.equal_size = args.equal_size
        config.max_part_size = args.max_part_size
        config.drop_hard_label = args.drop_hard_label
        config.sliding_window_size = -1
        # logger.info("PyTorch: setting up devices")
        if args.no_cuda:
            device = torch.device("cpu")
            n_gpu = 0
        elif args.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            n_gpu = torch.cuda.device_count()
            device_ids = list(range(n_gpu))
            random.shuffle(device_ids)
            config.device_ids = device_ids
            device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(0, 50000))
            device = torch.device("cuda", args.local_rank)
            n_gpu = 1
        config.device = device
        config.n_gpu = n_gpu
        config.num_attention_heads = args.num_attention_heads
        config.hidden_size = args.hidden_size
        config.intermediate_size = args.intermediate_size
        config.num_hidden_layers = args.num_hidden_layers
        config.output_hidden_states = args.output_hidden_states
        config.batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        config.test_onnx = False
        config.abandon_visual_features = False
        config.inference_mt_all_tasks = False



    add_config(config)
    ##### End Add Additional args

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.continue_train and (
            args.output_dir is not None
            and os.path.isfile(os.path.join(args.output_dir, "pytorch_model.bin"))):
        model = model_class.from_pretrained(args.output_dir,
                                            from_tf=bool('.ckpt' in args.output_dir),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        if args.drop_fc:
            model = model_class(config=config)
            model_dict = model.state_dict()
            state_dict = torch.load(os.path.join(args.model_name_or_path, "pytorch_model.bin"), map_location='cpu')
            logger.info("~~~~~~~~~~~~~~~~~before_convert~~~~~~~~~~~~~~~~~")
            for key, value in state_dict.items():
                if key.startswith("classifier"):
                    logger.info("drop key name: {}, value size: {}".format(key, value.size()))
                    continue
                if key not in model_dict:
                    # drop_weight_list.append(key)
                    logger.info("drop key name: {}, value size: {}".format(key, value.size()))
                    continue
                else:
                    model_dict[key] = value
            model.load_state_dict(model_dict)
        else:
            if args.multi_task_head_mapping:
                new_state_dict = task_head_mapping(args.multi_task_head_mapping, args.model_name_or_path)
                model = model_class(config=config)
                model.load_state_dict(new_state_dict)
            else:
                # walk to this
                model = model_class.from_pretrained(args.model_name_or_path,
                                                    from_tf=bool('.ckpt' in args.model_name_or_path),
                                                    config=config,
                                                    cache_dir=args.cache_dir if args.cache_dir else None)
    
    if args.freeze_encoder:
        model.freeze_encoder()
    
    if args.distillation:
        teacher_model_config = config_class.from_pretrained(args.teacher_model_config_path,
                                          finetuning_task="dummy",
                                          cache_dir=args.cache_dir if args.cache_dir else None)
        add_config_teacher(teacher_model_config)
        model_teacher = model_class.from_pretrained(args.model_name_or_path_teacher,
                                                    from_tf=bool('.ckpt' in args.model_name_or_path_teacher),
                                                    config=teacher_model_config,
                                                    cache_dir=args.cache_dir if args.cache_dir else None)

        for name, param in model_teacher.named_parameters():
            param.requires_grad = False
        
        model_teacher.to(args.device)

    if not args.use_deepscale:
        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # assert args.train_file != None
    # args.train_file_list = args.train_file.split(",")
    # args.train_strategy_list = list(map(float, args.train_strategy.split(",")))
    # args.train_strategy_list = [s.strip() for s in str(args.train_strategy).split(",")]
    for s_list in args.train_strategy_list:
        for s in s_list:
            assert float(s) % 1 == 0
    args.train_strategy_train_file_list = [[int(s) for s in s_list] for s_list in args.train_strategy_list]
    args.train_strategy_task_list = [sum(s_list) for s_list in args.train_strategy_train_file_list]
    
    # print("~~~~~~~~~~~~~~~~~~~~~")
    # print(args.train_strategy)
    # print(args.train_strategy_list)
    # Training
    if args.do_train:
        # if args.dev_file != None:
        #     args.dev_file_list = args.dev_file.split(",")
        #     if len(args.train_file_list) != len(args.dev_file_list):
        #         args.dev_file_list = [None] * len(args.train_file_list)
        # else:
        #     args.dev_file_list = [None] * len(args.train_file_list)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        if args.dev_file != None:  
            args.dev_file_list = args.dev_file.split(",") 

        assert len(args.dev_file_list) == len(args.task_name_list)

        
        if args.distillation:
            global_step, tr_loss = train_mt(args, model, tokenizer, model_teacher=model_teacher)
        else:
            global_step, tr_loss = train_mt(args, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        # tokenizer.save_pretrained(args.output_dir)    # 直接忽略 需要把hf下载的tokenizer.json复制到存储的文件夹中，这个文件包含了词表

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        # model = model_class.from_pretrained(args.output_dir)
        # tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        # model.to(args.device)

    # Evaluation, will update in future
    # Evaluation on the dev sets

    if args.do_eval and args.local_rank in [-1, 0]:
        # tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                    do_lower_case=args.do_lower_case,
                                                    cache_dir=args.cache_dir if args.cache_dir else None)
        checkpoints = [args.output_dir]

        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            # logger.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        if args.dev_file != None:  
            args.dev_file_list = args.dev_file.split(",") 

        assert len(args.dev_file_list) == len(args.task_name_list)

        tmp_best_f1 = -1
        tmp_best_checkpoint = ''

        task_index_map = {"MainImage":0, "Name":1, "Price":2}
        task_name_map = {0:"MainImage", 1:"Name", 2:"Price"}
        test_dataloader_list = {}
        test_dataset_list = {}
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
        
            # model = model_class.from_pretrained(checkpoint)
            # model.to(args.device)
            # if args.n_gpu > 1:
            #     model = torch.nn.DataParallel(model)
            
            # # Distributed training (should be after apex fp16 initialization)
            # if args.local_rank != -1:
            #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
            #                                                         output_device=args.local_rank,
            #                                                         find_unused_parameters=True)
            
            currency_list = np.load('./data/currency.npy', allow_pickle=True).tolist()
            # def evaluate_mt(args, model, tokenizer, task_index, prefix=""):
            
            all_cand_res_for_this_model = []

            model_config = config_class.from_pretrained(checkpoint,
                                                    finetuning_task="dummy",
                                                    cache_dir=args.cache_dir if args.cache_dir else None)
            add_model_config(model_config)

            model = model_class.from_pretrained(
                        checkpoint,
                        from_tf=False,
                        config=model_config,
                        cache_dir=None,
                    )
            

            vocab_clip_mapping = {}
            args.vocab_clip_mapping_file = None

            for task_index, task_name in enumerate(args.task_name_list): # 有三个xml_ee_binary_classification的task  分别对应 image name 和 price
                # 每一个task单独一个evaluate_mt的函数？
                tmp_task = task_name_map[task_index]
                if tmp_task not in test_dataloader_list:
                    entity_type = os.path.basename(args.dev_file_list[task_index]).split('_')[3]    # filename 'like V2_dev_conll_price_features_tsv.tsv'
                    label_this = ['O', entity_type]
                    label2id_this = {"O":0, entity_type:1}
                    label_map_this = {0:"O", 1:entity_type}
                    overlap_conll = trans_conll_file_to_overlap_file(os.path.join(args.data_dir, args.dev_file_list[task_index]), True)
                    test_dataset = LabelingDataset_mt(overlap_conll, tokenizer, label_this, args.vocab_clip_mapping_file is not None, vocab_clip_mapping, 512, model_config)
                    test_dataset_list[tmp_task] = test_dataset
                    test_dataloader = get_eval_dataloader(test_dataset, batch_size=model_config.batch_size)
                    test_dataloader_list[tmp_task] = test_dataloader
                else:
                    test_dataset = test_dataset_list[tmp_task]
                    test_dataloader = test_dataloader_list[tmp_task]
                

                test_label_ids, prediction_onnx = inference_mt(test_dataloader, model, task_index, model_config)
                real_prediction_onnx, real_label_ids = get_real_label_mt(prediction_onnx, test_label_ids, label_map_this, label2id_this)
                real_prediction_onnx_origin, examples_origin, label_real_from_example_ori, real_label_ids_ori= get_real_result_from_real_preds_add_mt(real_prediction_onnx, test_dataset.__get_examples__(), real_label_ids)
                
                real_prediction_onnx_origin_optimized = change_to_bio(real_prediction_onnx_origin)

                real_label_ids_ori_optimized = change_to_bio(real_label_ids_ori)

                out_res = get_metric(real_prediction_onnx_origin_optimized, real_label_ids_ori_optimized, examples_origin, currency_list)
                all_cand_res_for_this_model.append(out_res)
            min_f1 = min_F1(all_cand_res_for_this_model)
            logger.info('min_f1 {} for checkpoint {}'.format(str(min_f1), checkpoint) )
            
            if min_f1 > tmp_best_f1:
                tmp_best_f1 = min_f1
                tmp_best_checkpoint = checkpoint 
            logger.info("The current best checkpoint on dev set is : %s", tmp_best_checkpoint)
            logger.info("The current best f1 on dev set is : %s", str(tmp_best_f1))
        # 这个只是在dev 集上的最好
        logger.info("The best checkpoint on dev set is : %s", tmp_best_checkpoint)
        logger.info("The best f1 on dev set is : %s", str(tmp_best_f1))

        # Evaluate the best checkpoint on the test set.
        # Update future


            # for index, task_name in enumerate(args.task_name_list): # 有三个xml_ee_binary_classification的task  分别对应 image name 和 price
            #     # result = evaluate_mt(args, model, tokenizer, task_name, prefix=prefix)
            #     result = evaluate_mt(args, model, tokenizer, index, prefix=prefix)
            #     result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            #     if task_name not in results:
            #         results[task_name] = result
            #     results[task_name].update(result)

    # return results


if __name__ == "__main__":
    main()



