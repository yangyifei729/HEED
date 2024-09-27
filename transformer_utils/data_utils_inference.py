from dataclasses import dataclass
from typing import List, Optional, Union, Dict
from collections import defaultdict
from sklearn.preprocessing import minmax_scale
import logging
from torch.utils.data.dataset import Dataset
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm, trange
logger = logging.getLogger(__name__)
from transformers.data.data_collator import DataCollator, default_data_collator
# from numpy import *
import numpy as np
from scipy.special import softmax
import onnxruntime as rt
from seqeval.metrics import accuracy_score, classification_report, f1_score
import os
import heapq

@dataclass
class InputExample_pred:
    xml_path: str
    words: List[str]
    IsImage: List[str]
    IsPrecededByWS: List[str]
    IsPrecededByLineBreak: List[str]
    IsSameBoundingBox: List[str]
    IsSameElement: List[str]

@dataclass
class InputExample:
    """
    A single training/test example for token classification.
    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    xml_path: str
    words: List[str]
    words_ori: List[str]
    isImage: List[int]
    IsPrecededByWS: List[int]
    IsPrecededByLineBreak: List[int]
    bounding_box_IsSame: List[int]
    IsClipped: List[int]
    IsVisible: List[int]
    FontWeight: List[int]
    FontSize: List[int]
    bounding_X: List[int]
    bounding_Y: List[int]
    bounding_W: List[int]
    bounding_H: List[int]
    color_A: List[int]
    color_R: List[int]
    color_G: List[int]
    color_B: List[int]
    labels: Optional[List[str]]
    IsSameElement: List[int]
    bounding_Xe: List[int]
    bounding_Ye: List[int]
    bounding_We: List[int]
    bounding_He: List[int]
    isAnchor: List[int]
    part: List[int]

@dataclass
class InputExample_sliding_window:
    """
    A single training/test example for token classification.
    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    xml_path: str
    IsImage: List[int]
    IsSameElement: List[int]
    words: List[str]
    words_ori: List[str]
    labels: Optional[List[str]]
    visual_features: List[List[int]]
    xml_dic: Optional[List[int]]

@dataclass
@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]

    isImage: List[int]
    IsPrecededByWS: List[int]
    IsPrecededByLineBreak: List[int]
    bounding_box_IsSame: List[int]
    IsClipped: List[int]
    IsVisible: List[int]
    FontWeight: List[int]
    FontSize: List[int]
    bounding_X: List[int]
    bounding_Y: List[int]
    bounding_W: List[int]
    bounding_H: List[int]
    color_A: List[int]
    color_R: List[int]
    color_G: List[int]
    color_B: List[int]

    bounding_Xe: List[int]
    bounding_Ye: List[int]
    bounding_We: List[int]
    bounding_He: List[int]
    isAnchor: List[int]
    part: List[int]

    visual_features: Optional[List[List[int]]] = None
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None
    logits: Optional[List[float]] = None

@dataclass
class InputFeatures_sliding_window:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    visual_features: Optional[List[List[int]]] = None
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None
    logits: Optional[List[float]] = None



def trans_conll_file_to_overlap_file(conll_file_path, nodrop_no_label=False):
    overlap_conll_file = []
    content = []
    split_ = [i * 250 for i in range(10)]
    firstline = False
    number = 0
    conll_file = open(conll_file_path, 'r', encoding='utf-8')
    for line in conll_file:
        line_v = line.split(' ')
        if len(line_v) != 23:
            number += 1
            if number % 1000 == 0:
                print(number)
            if firstline:
                if not content:
                    continue
                has_label = False
                for line_tmp in content:
                    if 'O' != line_tmp.split(' ')[1]:
                        has_label = True
                if has_label or nodrop_no_label:
                    for part, start_index in enumerate(split_):
                        overlap_conll_file.append(xml_path)
                        for tmp_content in content[start_index: start_index + 400]:
                            overlap_conll_file.append(tmp_content + ' ' + str(part) + '\n')
                        if start_index + 400 >= len(content):
                            break
                else:
                    print(xml_path)
            xml_path = line
            firstline = True
            content = []
        else:
            content.append(line.replace('\n', ''))
    if content:
        if nodrop_no_label:
            for part, start_index in enumerate(split_):
                    overlap_conll_file.append(xml_path)
                    for tmp_content in content[start_index: start_index + 400]:
                        overlap_conll_file.append(tmp_content + ' ' + str(part) + '\n')
                    if start_index + 400 >= len(content):
                        break
        elif has_label:
            for part, start_index in enumerate(split_):
                    overlap_conll_file.append(xml_path)
                    for tmp_content in content[start_index: start_index + 400]:
                        overlap_conll_file.append(tmp_content + ' ' + str(part) + '\n')
                    if start_index + 400 >= len(content):
                        break
        else:
            pass
    return overlap_conll_file



def read_examples_from_file(data, config):
    guid_index = 1
    examples, words, words_ori, labels = [[] for i in range(4)]
    number__ = 0
    isImage, bounding_X, bounding_Y, bounding_W, bounding_H, color_A, color_R, color_G, color_B, bounding_box_IsSame, FontWeight, FontSize, IsPrecededByWS, IsPrecededByLineBreak, IsClipped, IsVisible = [[] for i in range(16)]
    bounding_Xe, bounding_Ye, bounding_We, bounding_He, isAnchor, part, IsSameElement = [[] for i in range(7)]
    xml_path = ''
    for line in data:
        if line.startswith("-DOCSTART-") or line == "" or line == "\n" or len(line.split(' ')) < 4:
            if words:
                number__ += 1
                if number__ % 10000 == 0:
                    print("read_examples_from_file:", number__)
                examples.append(
                    InputExample(guid=f"{guid_index}", xml_path=xml_path, words=words, words_ori=words_ori,
                                 labels=labels,
                                 isImage=isImage,
                                 bounding_X=bounding_X, bounding_Y=bounding_Y, bounding_W=bounding_W,
                                 bounding_H=bounding_H, color_A=color_A, color_R=color_R, color_G=color_G,
                                 color_B=color_B, bounding_box_IsSame=bounding_box_IsSame, FontWeight=FontWeight,
                                 FontSize=FontSize, IsPrecededByWS=IsPrecededByWS,
                                 IsPrecededByLineBreak=IsPrecededByLineBreak, IsClipped=IsClipped,
                                 IsVisible=IsVisible, bounding_Xe=bounding_Xe, bounding_Ye=bounding_Ye,
                                 bounding_We=bounding_We, bounding_He=bounding_He, isAnchor=isAnchor, part=part, IsSameElement=IsSameElement))
                guid_index += 1
                words, words_ori, labels = [[] for i in range(3)]
                isImage, bounding_X, bounding_Y, bounding_W, bounding_H, color_A, color_R, color_G, color_B, bounding_box_IsSame, FontWeight, FontSize, IsPrecededByWS, IsPrecededByLineBreak, IsClipped, IsVisible = [[] for i in range(16)]
                bounding_Xe, bounding_Ye, bounding_We, bounding_He, isAnchor, part, IsSameElement = [[] for i in range(7)]
                xml_path = ""
            xml_path = line.strip()
        else:
            splits = line.replace('\n', '').split(" ")
            words.append("#IMAGE" if splits[2] == '1' else splits[0])
            words_ori.append(splits[0])
            labels.append(splits[1])
            #######################
            if config.add_visual_features:
                isImage.append(int(splits[2]))
                bounding_X.append(int(splits[3]))
                bounding_Y.append(int(splits[4]))
                bounding_W.append(int(splits[5]))
                bounding_H.append(int(splits[6]))
                color_A.append(int(splits[7]))
                color_R.append(int(splits[8]))
                color_G.append(int(splits[9]))
                color_B.append(int(splits[10]))
                bounding_box_IsSame.append(int(splits[11]))
                FontWeight.append(int(splits[12]))
                FontSize.append(int(splits[13]))
                IsPrecededByWS.append(int(splits[14]))
                IsPrecededByLineBreak.append(int(splits[15]))
                IsClipped.append(int(splits[16]))
                IsVisible.append(int(splits[17]))
                IsSameElement.append(1 if len(bounding_Xe) > 0 and int(splits[18]) == bounding_Xe[-1] and int(splits[19]) == bounding_Ye[-1] and int(splits[20]) == bounding_We[-1] and int(splits[21]) == bounding_He[-1] else 0)
                bounding_Xe.append(int(splits[18]))
                bounding_Ye.append(int(splits[19]))
                bounding_We.append(int(splits[20]))
                bounding_He.append(int(splits[21]))
                isAnchor.append(int(splits[22]))
                part.append((int(splits[23])))

    if words:
        examples.append(
            InputExample(guid=f"{guid_index}", xml_path=xml_path, words=words, words_ori=words_ori, labels=labels,
                         isImage=isImage,
                         bounding_X=bounding_X, bounding_Y=bounding_Y, bounding_W=bounding_W,
                         bounding_H=bounding_H, color_A=color_A, color_R=color_R, color_G=color_G,
                         color_B=color_B, bounding_box_IsSame=bounding_box_IsSame,
                         FontWeight=FontWeight, FontSize=FontSize, IsPrecededByWS=IsPrecededByWS,
                         IsPrecededByLineBreak=IsPrecededByLineBreak, IsClipped=IsClipped,
                         IsVisible=IsVisible, bounding_Xe=bounding_Xe, bounding_Ye=bounding_Ye,
                         bounding_We=bounding_We, bounding_He=bounding_He, isAnchor=isAnchor,
                         part=part, IsSameElement=IsSameElement))
    return examples


def read_examples_from_file_sliding_window(data, config):
    guid_index = 1
    examples = []
    number__ = 0
    IsImage, IsSameElement = [], []
    words = []
    words_ori = []
    labels = []
    visual_features = []
    xml_path = ''
    for line in open(data, 'r', encoding='utf-8'):
        if line.startswith("-DOCSTART-") or line == "" or line == "\n" or len(line.split(' ')) < 4:
            if words:
                number__ += 1
                if number__ % 10000 == 0:
                    print("read_examples_from_file:", number__)
                examples.append(InputExample_sliding_window(guid=f"{guid_index}", xml_path=xml_path, words=words, IsImage=IsImage, IsSameElement=IsSameElement, words_ori=words_ori, labels=labels, visual_features=visual_features, xml_dic=[]))
                guid_index += 1
                words = []
                words_ori = []
                labels = []
                visual_features = []
                IsImage, IsSameElement = [], []
                xml_path = ""
            xml_path = line.strip()
        else:
            splits = line.replace('\n', '').split(" ")
            if splits[2] == '1':
                words.append("#IMAGE")
            else:
                words.append(splits[0])
            words_ori.append(splits[0])
            labels.append(splits[1])
            #######################
            if config.add_visual_features:
                visual_features.append([int(x) for x in splits[2:23] + [0]])
                IsImage.append(int(splits[2]))
                Element = [int(ele) for ele in splits[18:22]]
                if len(IsSameElement) == 0:
                    lastElement = Element
                    IsSameElement.append(0)
                else:
                    IsSameElement.append(1 if Element == lastElement else 0)
                    lastElement = Element
    if words:
        examples.append(InputExample_sliding_window(guid=f"{guid_index}", xml_path=xml_path, IsImage=IsImage, IsSameElement=IsSameElement, words=words, words_ori=words_ori, labels=labels, visual_features=visual_features, xml_dic=[]))
    return examples


def convert_examples_to_features(
        examples: List[InputExample],
        label_list: List[str],
        max_seq_length: int,
        tokenizer,
        use_vocab_clip=False,
        vocab_clip_mapping={},
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        config=None
) -> List[InputFeatures]:
    """ Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # TODO clean up all this to leverage built-in features of tokenizers
    label_map = defaultdict(int)
    label_list = ['O', 'B-MainImage', 'I-MainImage', 'B-Manufacturer', 'I-Manufacturer', 'B-Name',
                  'I-Name', 'B-Price', 'I-Price', 'B-Rating', 'I-Rating', 'B-NumberofReviews',
                  'I-NumberofReviews', 'B-ProductCode', 'I-ProductCode', 'B-OutOfStock',
                  'I-OutOfStock', "B-Address", "I-Address"]
    for i, label in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = []
        label_ids = []
        isImage, bounding_X, bounding_Y, bounding_W, bounding_H, color_A, color_R, color_G, color_B, bounding_box_IsSame, FontWeight, FontSize, IsPrecededByWS, IsPrecededByLineBreak, IsClipped, IsVisible = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        bounding_Xe, bounding_Ye, bounding_We, bounding_He, isAnchor, part = [], [], [], [], [], []
        for word, label, isImage_tmp, bounding_X_tmp, bounding_Y_tmp, bounding_W_tmp, bounding_H_tmp, color_A_tmp, color_R_tmp, color_G_tmp, color_B_tmp, bounding_box_IsSame_tmp, FontWeight_tmp, FontSize_tmp, IsPrecededByWS_tmp, IsPrecededByLineBreak_tmp, IsClipped_tmp, IsVisible_tmp, bounding_Xe_tmp, bounding_Ye_tmp, bounding_We_tmp, bounding_He_tmp, isAnchor_tmp, part_tmp in zip(
                example.words, example.labels, example.isImage, example.bounding_X, example.bounding_Y,
                example.bounding_W, example.bounding_H, example.color_A, example.color_R, example.color_G,
                example.color_B, example.bounding_box_IsSame, example.FontWeight, example.FontSize,
                example.IsPrecededByWS, example.IsPrecededByLineBreak, example.IsClipped, example.IsVisible,
                example.bounding_Xe, example.bounding_Ye, example.bounding_We, example.bounding_He, example.isAnchor,
                example.part):
            word_tokens = tokenizer.tokenize(word)

            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                label_ids_ori = []
                for j in range(len(label_ids)):
                    if label_ids[j] != pad_token_label_id:
                        label_ids_ori.append(label_ids[j])

                isImage.extend([isImage_tmp] * len(word_tokens))
                bounding_X.extend([bounding_X_tmp] * len(word_tokens))
                bounding_Y.extend([bounding_Y_tmp] * len(word_tokens))
                bounding_W.extend([bounding_W_tmp] * len(word_tokens))
                bounding_H.extend([bounding_H_tmp] * len(word_tokens))
                color_A.extend([color_A_tmp] * len(word_tokens))
                color_R.extend([color_R_tmp] * len(word_tokens))
                color_G.extend([color_G_tmp] * len(word_tokens))
                color_B.extend([color_B_tmp] * len(word_tokens))
                bounding_box_IsSame.extend([bounding_box_IsSame_tmp] * len(word_tokens))
                FontWeight.extend([FontWeight_tmp] * len(word_tokens))
                FontSize.extend([FontSize_tmp] * len(word_tokens))
                IsPrecededByWS.extend([IsPrecededByWS_tmp] * len(word_tokens))
                IsPrecededByLineBreak.extend([IsPrecededByLineBreak_tmp] * len(word_tokens))
                IsClipped.extend([IsClipped_tmp] * len(word_tokens))
                IsVisible.extend([IsVisible_tmp] * len(word_tokens))

                bounding_Xe.extend([bounding_Xe_tmp] * len(word_tokens))
                bounding_Ye.extend([bounding_Ye_tmp] * len(word_tokens))
                bounding_We.extend([bounding_We_tmp] * len(word_tokens))
                bounding_He.extend([bounding_He_tmp] * len(word_tokens))
                isAnchor.extend([isAnchor_tmp] * len(word_tokens))
                part.extend([part_tmp] * len(word_tokens))

            else:
                tokens.extend([''])
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                isImage.extend([0])
                bounding_X.extend([0])
                bounding_Y.extend([0])
                bounding_W.extend([0])
                bounding_H.extend([0])
                color_A.extend([0])
                color_R.extend([0])
                color_G.extend([0])
                color_B.extend([0])
                bounding_box_IsSame.extend([0])
                FontWeight.extend([0])
                FontSize.extend([0])
                IsPrecededByWS.extend([0])
                IsPrecededByLineBreak.extend([0])
                IsClipped.extend([0])
                IsVisible.extend([0])
                bounding_Xe.extend([0])
                bounding_Ye.extend([0])
                bounding_We.extend([0])
                bounding_He.extend([0])
                isAnchor.extend([0])
                part.extend([0])

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            isImage = isImage[: (max_seq_length - special_tokens_count)]
            bounding_X = bounding_X[: (max_seq_length - special_tokens_count)]
            bounding_Y = bounding_Y[: (max_seq_length - special_tokens_count)]
            bounding_W = bounding_W[: (max_seq_length - special_tokens_count)]
            bounding_H = bounding_H[: (max_seq_length - special_tokens_count)]
            color_A = color_A[: (max_seq_length - special_tokens_count)]
            color_R = color_R[: (max_seq_length - special_tokens_count)]
            color_G = color_G[: (max_seq_length - special_tokens_count)]
            color_B = color_B[: (max_seq_length - special_tokens_count)]
            bounding_box_IsSame = bounding_box_IsSame[: (max_seq_length - special_tokens_count)]
            FontWeight = FontWeight[: (max_seq_length - special_tokens_count)]
            FontSize = FontSize[: (max_seq_length - special_tokens_count)]
            IsPrecededByWS = IsPrecededByWS[: (max_seq_length - special_tokens_count)]
            IsPrecededByLineBreak = IsPrecededByLineBreak[: (max_seq_length - special_tokens_count)]
            IsClipped = IsClipped[: (max_seq_length - special_tokens_count)]
            IsVisible = IsVisible[: (max_seq_length - special_tokens_count)]

            bounding_Xe = bounding_Xe[: (max_seq_length - special_tokens_count)]
            bounding_Ye = bounding_Ye[: (max_seq_length - special_tokens_count)]
            bounding_We = bounding_We[: (max_seq_length - special_tokens_count)]
            bounding_He = bounding_He[: (max_seq_length - special_tokens_count)]
            isAnchor = isAnchor[: (max_seq_length - special_tokens_count)]
            part = part[: (max_seq_length - special_tokens_count)]
        bounding_X = [int(x) for x in minmax_scale(bounding_X, feature_range=[0, config.max_bounding_box_size - 1])]
        bounding_Y = [int(x) for x in minmax_scale(bounding_Y, feature_range=[0, config.max_bounding_box_size - 1])]
        bounding_W = [int(x) for x in minmax_scale(bounding_W, feature_range=[0, config.max_bounding_box_size - 1])]
        bounding_H = [int(x) for x in minmax_scale(bounding_H, feature_range=[0, config.max_bounding_box_size - 1])]
        color_A = [int(x) for x in minmax_scale(color_A, feature_range=[0, config.max_color_size - 1])]
        color_R = [int(x) for x in minmax_scale(color_R, feature_range=[0, config.max_color_size - 1])]
        color_G = [int(x) for x in minmax_scale(color_G, feature_range=[0, config.max_color_size - 1])]
        color_B = [int(x) for x in minmax_scale(color_B, feature_range=[0, config.max_color_size - 1])]
        FontWeight = [int(x) for x in minmax_scale(FontWeight, feature_range=[0, config.max_FontWeight_size - 1])]
        FontSize = [int(x) for x in minmax_scale(FontSize, feature_range=[0, config.max_FontSize_size - 1])]
        bounding_Xe = [int(x) for x in minmax_scale(bounding_Xe, feature_range=[0, config.max_bounding_box_size - 1])]
        bounding_Ye = [int(x) for x in minmax_scale(bounding_Ye, feature_range=[0, config.max_bounding_box_size - 1])]
        bounding_We = [int(x) for x in minmax_scale(bounding_We, feature_range=[0, config.max_bounding_box_size - 1])]
        bounding_He = [int(x) for x in minmax_scale(bounding_He, feature_range=[0, config.max_bounding_box_size - 1])]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        isImage += [0]
        bounding_X += [0]
        bounding_Y += [0]
        bounding_W += [0]
        bounding_H += [0]
        color_A += [0]
        color_R += [0]
        color_G += [0]
        color_B += [0]
        bounding_box_IsSame += [0]
        FontWeight += [0]
        FontSize += [0]
        IsPrecededByWS += [0]
        IsPrecededByLineBreak += [0]
        IsClipped += [0]
        IsVisible += [0]

        bounding_Xe += [0]
        bounding_Ye += [0]
        bounding_We += [0]
        bounding_He += [0]
        isAnchor += [0]
        part += [0]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            isImage += [0]
            bounding_X += [0]
            bounding_Y += [0]
            bounding_W += [0]
            bounding_H += [0]
            color_A += [0]
            color_R += [0]
            color_G += [0]
            color_B += [0]
            bounding_box_IsSame += [0]
            FontWeight += [0]
            FontSize += [0]
            IsPrecededByWS += [0]
            IsPrecededByLineBreak += [0]
            IsClipped += [0]
            IsVisible += [0]

            bounding_Xe += [0]
            bounding_Ye += [0]
            bounding_We += [0]
            bounding_He += [0]
            isAnchor += [0]
            part += [0]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
            isImage += [0]
            bounding_X += [0]
            bounding_Y += [0]
            bounding_W += [0]
            bounding_H += [0]
            color_A += [0]
            color_R += [0]
            color_G += [0]
            color_B += [0]
            bounding_box_IsSame += [0]
            FontWeight += [0]
            FontSize += [0]
            IsPrecededByWS += [0]
            IsPrecededByLineBreak += [0]
            IsClipped += [0]
            IsVisible += [0]

            bounding_Xe += [0]
            bounding_Ye += [0]
            bounding_We += [0]
            bounding_He += [0]
            isAnchor += [0]
            part += [0]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids
            isImage = [0] + isImage
            bounding_X = [0] + bounding_X
            bounding_Y = [0] + bounding_Y
            bounding_W = [0] + bounding_W
            bounding_H = [0] + bounding_H
            color_A = [0] + color_A
            color_R = [0] + color_R
            color_G = [0] + color_G
            color_B = [0] + color_B
            bounding_box_IsSame = [0] + bounding_box_IsSame
            FontWeight = [0] + FontWeight
            FontSize = [0] + FontSize
            IsPrecededByWS = [0] + IsPrecededByWS
            IsPrecededByLineBreak = [0] + IsPrecededByLineBreak
            IsClipped = [0] + IsClipped
            IsVisible = [0] + IsVisible

            bounding_Xe = [0] + bounding_Xe
            bounding_Ye = [0] + bounding_Ye
            bounding_We = [0] + bounding_We
            bounding_He = [0] + bounding_He
            isAnchor = [0] + isAnchor
            part = [0] + part

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        if use_vocab_clip:
            assert len(vocab_clip_mapping) > 0
            input_ids = [vocab_clip_mapping[x] if vocab_clip_mapping.__contains__(x) else vocab_clip_mapping[3] for x in input_ids]
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([0] * padding_length) + label_ids
            isImage = ([0] * padding_length) + isImage
            bounding_X = ([0] * padding_length) + bounding_X
            bounding_Y = ([0] * padding_length) + bounding_Y
            bounding_W = ([0] * padding_length) + bounding_W
            bounding_H = ([0] * padding_length) + bounding_H
            color_A = ([0] * padding_length) + color_A
            color_R = ([0] * padding_length) + color_R
            color_G = ([0] * padding_length) + color_G
            color_B = ([0] * padding_length) + color_B
            bounding_box_IsSame = ([0] * padding_length) + bounding_box_IsSame
            FontWeight = ([0] * padding_length) + FontWeight
            FontSize = ([0] * padding_length) + FontSize
            IsPrecededByWS = ([0] * padding_length) + IsPrecededByWS
            IsPrecededByLineBreak = ([0] * padding_length) + IsPrecededByLineBreak
            IsClipped = ([0] * padding_length) + IsClipped
            IsVisible = ([0] * padding_length) + IsVisible

            bounding_Xe = ([0] * padding_length) + bounding_Xe
            bounding_Ye = ([0] * padding_length) + bounding_Ye
            bounding_We = ([0] * padding_length) + bounding_We
            bounding_He = ([0] * padding_length) + bounding_He
            isAnchor = ([0] * padding_length) + isAnchor
            part = ([0] * padding_length) + part
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length
            isImage += [0] * padding_length
            bounding_X += [0] * padding_length
            bounding_Y += [0] * padding_length
            bounding_W += [0] * padding_length
            bounding_H += [0] * padding_length
            color_A += [0] * padding_length
            color_R += [0] * padding_length
            color_G += [0] * padding_length
            color_B += [0] * padding_length
            bounding_box_IsSame += [0] * padding_length
            FontWeight += [0] * padding_length
            FontSize += [0] * padding_length
            IsPrecededByWS += [0] * padding_length
            IsPrecededByLineBreak += [0] * padding_length
            IsClipped += [0] * padding_length
            IsVisible += [0] * padding_length

            bounding_Xe += [0] * padding_length
            bounding_Ye += [0] * padding_length
            bounding_We += [0] * padding_length
            bounding_He += [0] * padding_length
            isAnchor += [0] * padding_length
            part += [0] * padding_length

            # bounding_X = bounding_Xe
            # bounding_Y = bounding_Ye
            # bounding_W = bounding_We
            # bounding_H = bounding_He

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(isImage) == max_seq_length
        assert len(bounding_X) == max_seq_length
        assert len(bounding_Y) == max_seq_length
        assert len(bounding_W) == max_seq_length
        assert len(bounding_H) == max_seq_length
        assert len(color_A) == max_seq_length
        assert len(color_R) == max_seq_length
        assert len(color_G) == max_seq_length
        assert len(color_B) == max_seq_length
        assert len(bounding_box_IsSame) == max_seq_length
        assert len(FontWeight) == max_seq_length
        assert len(FontSize) == max_seq_length
        assert len(IsPrecededByWS) == max_seq_length
        assert len(IsPrecededByLineBreak) == max_seq_length
        assert len(IsClipped) == max_seq_length
        assert len(IsVisible) == max_seq_length
        assert len(bounding_Xe) == max_seq_length
        assert len(bounding_Ye) == max_seq_length
        assert len(bounding_We) == max_seq_length
        assert len(bounding_He) == max_seq_length
        assert len(isAnchor) == max_seq_length
        assert len(part) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

            logger.info("isImage: %s", " ".join([str(x) for x in isImage]))
            logger.info("bounding_X: %s", " ".join([str(x) for x in bounding_X]))
            logger.info("bounding_Y: %s", " ".join([str(x) for x in bounding_Y]))
            logger.info("bounding_W: %s", " ".join([str(x) for x in bounding_W]))
            logger.info("bounding_H: %s", " ".join([str(x) for x in bounding_H]))
            logger.info("color_A: %s", " ".join([str(x) for x in color_A]))
            logger.info("color_R: %s", " ".join([str(x) for x in color_R]))
            logger.info("color_G: %s", " ".join([str(x) for x in color_G]))
            logger.info("color_B: %s", " ".join([str(x) for x in color_B]))
            logger.info("bounding_box_IsSame: %s", " ".join([str(x) for x in bounding_box_IsSame]))
            logger.info("FontWeight: %s", " ".join([str(x) for x in FontWeight]))
            logger.info("FontSize: %s", " ".join([str(x) for x in FontSize]))
            logger.info("IsPrecededByWS: %s", " ".join([str(x) for x in IsPrecededByWS]))
            logger.info("IsPrecededByLineBreak: %s", " ".join([str(x) for x in IsPrecededByLineBreak]))
            logger.info("IsClipped: %s", " ".join([str(x) for x in IsClipped]))
            logger.info("IsVisible: %s", " ".join([str(x) for x in IsVisible]))

            logger.info("bounding_Xe: %s", " ".join([str(x) for x in bounding_Xe]))
            logger.info("bounding_Ye: %s", " ".join([str(x) for x in bounding_Ye]))
            logger.info("bounding_We: %s", " ".join([str(x) for x in bounding_We]))
            logger.info("bounding_He: %s", " ".join([str(x) for x in bounding_He]))
            logger.info("isAnchor: %s", " ".join([str(x) for x in isAnchor]))
            logger.info("part: %s", " ".join([str(x) for x in part]))

        if "token_type_ids" not in tokenizer.model_input_names:
            segment_ids = None

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, label_ids=label_ids,
                isImage=isImage, bounding_X=bounding_X, bounding_Y=bounding_Y, bounding_W=bounding_W,
                bounding_H=bounding_H, color_A=color_A, color_R=color_R, color_G=color_G, color_B=color_B,
                bounding_box_IsSame=bounding_box_IsSame, FontWeight=FontWeight, FontSize=FontSize,
                IsPrecededByWS=IsPrecededByWS, IsPrecededByLineBreak=IsPrecededByLineBreak, IsClipped=IsClipped,
                IsVisible=IsVisible, bounding_Xe=bounding_Xe, bounding_Ye=bounding_Ye, bounding_We=bounding_We,
                bounding_He=bounding_He, isAnchor=isAnchor, part=part
            )
        )
    return features


def overlap_func(line_dict, overlap_size, max_seq_length):
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
        result_line.append(dict_tmp)
    return result_line


def convert_examples_to_features_sliding_window(
        examples: List[InputExample],
        label_list: List[str],
        max_seq_length: int,
        tokenizer,
        use_vocab_clip=False,
        vocab_clip_mapping={},
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        config=None,
        overlap_size=50
) -> List[InputFeatures]:
    """ Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # TODO clean up all this to leverage built-in features of tokenizers
    label_map = defaultdict(int)
    label_list = ['O', 'B-MainImage', 'I-MainImage', 'B-Manufacturer', 'I-Manufacturer', 'B-Name',
                  'I-Name', 'B-Price', 'I-Price', 'B-Rating', 'I-Rating', 'B-NumberofReviews',
                  'I-NumberofReviews', 'B-ProductCode', 'I-ProductCode', 'B-OutOfStock',
                  'I-OutOfStock', "B-Address", "I-Address"]
    for i, label in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        visual_features = []
        tokens = []
        label_ids = []
        for word, label, visual_features_tmp in zip(example.words, example.labels, example.visual_features):
            word_tokens = tokenizer.tokenize(word)
            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                label_ids_ori = []
                for j in range(len(label_ids)):
                    if label_ids[j] != pad_token_label_id:
                        label_ids_ori.append(label_ids[j])
                visual_features.extend([visual_features_tmp] * len(word_tokens))

            else:
                tokens.extend([''])
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                visual_features.append([0] * len(visual_features_tmp))
        example.words = example.words_ori
        bounding_features = minmax_scale([feature[1:5] for feature in visual_features], feature_range=[0, config.max_bounding_box_size - 1], axis=0).astype(int).tolist()
        color_features = minmax_scale([feature[5:9] for feature in visual_features], feature_range=[0, config.max_color_size - 1], axis=0).astype(int).tolist()
        FontWeight_features = minmax_scale([feature[10:11] for feature in visual_features], feature_range=[0, config.max_FontWeight_size - 1], axis=0).astype(int).tolist()
        FontSize_features = minmax_scale([feature[11:12] for feature in visual_features], feature_range=[0, config.max_FontSize_size - 1], axis=0).astype(int).tolist()
        bounding_e_features = minmax_scale([feature[16:20] for feature in visual_features], feature_range=[0, config.max_bounding_box_size - 1], axis=0).astype(int).tolist()
        for index_f, _ in enumerate(visual_features):
            _[1:5] = bounding_features[index_f]
            _[5:9] = color_features[index_f]
            _[10:11] = FontWeight_features[index_f]
            _[11:12] = FontSize_features[index_f]
            _[16:20] = bounding_e_features[index_f]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        if use_vocab_clip:
            assert len(vocab_clip_mapping) > 0
            input_ids = [vocab_clip_mapping[x] if vocab_clip_mapping.__contains__(x) else vocab_clip_mapping[3] for x in input_ids]
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        assert len(input_mask) == len(input_ids)
        assert len(visual_features) == len(input_ids)

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s", example.guid)
        #     logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
        #     logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))

        if "token_type_ids" not in tokenizer.model_input_names:
            segment_ids = None
        line_dict = {"input_ids": input_ids, "attention_mask": input_mask, "label": label_ids, "visual_features": visual_features}
        overlap_lines = overlap_func(line_dict, overlap_size, max_seq_length)
        for overlap_line in overlap_lines:
            example.xml_dic.append(len(features))
            features.append(
                InputFeatures_sliding_window(
                    input_ids=overlap_line["input_ids"], attention_mask=overlap_line["attention_mask"], token_type_ids=segment_ids, label_ids=overlap_line["label"], visual_features=overlap_line["visual_features"]
                )
            )

    return features




def convert_examples_to_features_mt(
        examples: List[InputExample],
        label_list: List[str],
        max_seq_length: int,
        tokenizer,
        use_vocab_clip=False,
        vocab_clip_mapping={},
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        config=None
) -> List[InputFeatures]:
    """ Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # TODO clean up all this to leverage built-in features of tokenizers
    label_map = defaultdict(int)
    # the label list should be ["O", "MainImage"] or ["O", "Price"] or ["O", "Name"] for this case, the label_map is {"O":0, "Price":1}
    for i, label in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        tokens = []
        label_ids = []
        isImage, bounding_X, bounding_Y, bounding_W, bounding_H, color_A, color_R, color_G, color_B, bounding_box_IsSame, FontWeight, FontSize, IsPrecededByWS, IsPrecededByLineBreak, IsClipped, IsVisible = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        bounding_Xe, bounding_Ye, bounding_We, bounding_He, isAnchor, part = [], [], [], [], [], []
        for word, label, isImage_tmp, bounding_X_tmp, bounding_Y_tmp, bounding_W_tmp, bounding_H_tmp, color_A_tmp, color_R_tmp, color_G_tmp, color_B_tmp, bounding_box_IsSame_tmp, FontWeight_tmp, FontSize_tmp, IsPrecededByWS_tmp, IsPrecededByLineBreak_tmp, IsClipped_tmp, IsVisible_tmp, bounding_Xe_tmp, bounding_Ye_tmp, bounding_We_tmp, bounding_He_tmp, isAnchor_tmp, part_tmp in zip(
                example.words, example.labels, example.isImage, example.bounding_X, example.bounding_Y,
                example.bounding_W, example.bounding_H, example.color_A, example.color_R, example.color_G,
                example.color_B, example.bounding_box_IsSame, example.FontWeight, example.FontSize,
                example.IsPrecededByWS, example.IsPrecededByLineBreak, example.IsClipped, example.IsVisible,
                example.bounding_Xe, example.bounding_Ye, example.bounding_We, example.bounding_He, example.isAnchor,
                example.part):
            word_tokens = tokenizer.tokenize(word)

            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                label_ids_ori = []
                for j in range(len(label_ids)):
                    if label_ids[j] != pad_token_label_id:
                        label_ids_ori.append(label_ids[j])

                isImage.extend([isImage_tmp] * len(word_tokens))
                bounding_X.extend([bounding_X_tmp] * len(word_tokens))
                bounding_Y.extend([bounding_Y_tmp] * len(word_tokens))
                bounding_W.extend([bounding_W_tmp] * len(word_tokens))
                bounding_H.extend([bounding_H_tmp] * len(word_tokens))
                color_A.extend([color_A_tmp] * len(word_tokens))
                color_R.extend([color_R_tmp] * len(word_tokens))
                color_G.extend([color_G_tmp] * len(word_tokens))
                color_B.extend([color_B_tmp] * len(word_tokens))
                bounding_box_IsSame.extend([bounding_box_IsSame_tmp] * len(word_tokens))
                FontWeight.extend([FontWeight_tmp] * len(word_tokens))
                FontSize.extend([FontSize_tmp] * len(word_tokens))
                IsPrecededByWS.extend([IsPrecededByWS_tmp] * len(word_tokens))
                IsPrecededByLineBreak.extend([IsPrecededByLineBreak_tmp] * len(word_tokens))
                IsClipped.extend([IsClipped_tmp] * len(word_tokens))
                IsVisible.extend([IsVisible_tmp] * len(word_tokens))

                bounding_Xe.extend([bounding_Xe_tmp] * len(word_tokens))
                bounding_Ye.extend([bounding_Ye_tmp] * len(word_tokens))
                bounding_We.extend([bounding_We_tmp] * len(word_tokens))
                bounding_He.extend([bounding_He_tmp] * len(word_tokens))
                isAnchor.extend([isAnchor_tmp] * len(word_tokens))
                part.extend([part_tmp] * len(word_tokens))

            else:
                tokens.extend([''])
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                isImage.extend([0])
                bounding_X.extend([0])
                bounding_Y.extend([0])
                bounding_W.extend([0])
                bounding_H.extend([0])
                color_A.extend([0])
                color_R.extend([0])
                color_G.extend([0])
                color_B.extend([0])
                bounding_box_IsSame.extend([0])
                FontWeight.extend([0])
                FontSize.extend([0])
                IsPrecededByWS.extend([0])
                IsPrecededByLineBreak.extend([0])
                IsClipped.extend([0])
                IsVisible.extend([0])
                bounding_Xe.extend([0])
                bounding_Ye.extend([0])
                bounding_We.extend([0])
                bounding_He.extend([0])
                isAnchor.extend([0])
                part.extend([0])

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            isImage = isImage[: (max_seq_length - special_tokens_count)]
            bounding_X = bounding_X[: (max_seq_length - special_tokens_count)]
            bounding_Y = bounding_Y[: (max_seq_length - special_tokens_count)]
            bounding_W = bounding_W[: (max_seq_length - special_tokens_count)]
            bounding_H = bounding_H[: (max_seq_length - special_tokens_count)]
            color_A = color_A[: (max_seq_length - special_tokens_count)]
            color_R = color_R[: (max_seq_length - special_tokens_count)]
            color_G = color_G[: (max_seq_length - special_tokens_count)]
            color_B = color_B[: (max_seq_length - special_tokens_count)]
            bounding_box_IsSame = bounding_box_IsSame[: (max_seq_length - special_tokens_count)]
            FontWeight = FontWeight[: (max_seq_length - special_tokens_count)]
            FontSize = FontSize[: (max_seq_length - special_tokens_count)]
            IsPrecededByWS = IsPrecededByWS[: (max_seq_length - special_tokens_count)]
            IsPrecededByLineBreak = IsPrecededByLineBreak[: (max_seq_length - special_tokens_count)]
            IsClipped = IsClipped[: (max_seq_length - special_tokens_count)]
            IsVisible = IsVisible[: (max_seq_length - special_tokens_count)]

            bounding_Xe = bounding_Xe[: (max_seq_length - special_tokens_count)]
            bounding_Ye = bounding_Ye[: (max_seq_length - special_tokens_count)]
            bounding_We = bounding_We[: (max_seq_length - special_tokens_count)]
            bounding_He = bounding_He[: (max_seq_length - special_tokens_count)]
            isAnchor = isAnchor[: (max_seq_length - special_tokens_count)]
            part = part[: (max_seq_length - special_tokens_count)]
        bounding_X = [int(x) for x in minmax_scale(bounding_X, feature_range=[0, config.max_bounding_box_size - 1])]
        bounding_Y = [int(x) for x in minmax_scale(bounding_Y, feature_range=[0, config.max_bounding_box_size - 1])]
        bounding_W = [int(x) for x in minmax_scale(bounding_W, feature_range=[0, config.max_bounding_box_size - 1])]
        bounding_H = [int(x) for x in minmax_scale(bounding_H, feature_range=[0, config.max_bounding_box_size - 1])]
        color_A = [int(x) for x in minmax_scale(color_A, feature_range=[0, config.max_color_size - 1])]
        color_R = [int(x) for x in minmax_scale(color_R, feature_range=[0, config.max_color_size - 1])]
        color_G = [int(x) for x in minmax_scale(color_G, feature_range=[0, config.max_color_size - 1])]
        color_B = [int(x) for x in minmax_scale(color_B, feature_range=[0, config.max_color_size - 1])]
        FontWeight = [int(x) for x in minmax_scale(FontWeight, feature_range=[0, config.max_FontWeight_size - 1])]
        FontSize = [int(x) for x in minmax_scale(FontSize, feature_range=[0, config.max_FontSize_size - 1])]
        bounding_Xe = [int(x) for x in minmax_scale(bounding_Xe, feature_range=[0, config.max_bounding_box_size - 1])]
        bounding_Ye = [int(x) for x in minmax_scale(bounding_Ye, feature_range=[0, config.max_bounding_box_size - 1])]
        bounding_We = [int(x) for x in minmax_scale(bounding_We, feature_range=[0, config.max_bounding_box_size - 1])]
        bounding_He = [int(x) for x in minmax_scale(bounding_He, feature_range=[0, config.max_bounding_box_size - 1])]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        isImage += [0]
        bounding_X += [0]
        bounding_Y += [0]
        bounding_W += [0]
        bounding_H += [0]
        color_A += [0]
        color_R += [0]
        color_G += [0]
        color_B += [0]
        bounding_box_IsSame += [0]
        FontWeight += [0]
        FontSize += [0]
        IsPrecededByWS += [0]
        IsPrecededByLineBreak += [0]
        IsClipped += [0]
        IsVisible += [0]

        bounding_Xe += [0]
        bounding_Ye += [0]
        bounding_We += [0]
        bounding_He += [0]
        isAnchor += [0]
        part += [0]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            isImage += [0]
            bounding_X += [0]
            bounding_Y += [0]
            bounding_W += [0]
            bounding_H += [0]
            color_A += [0]
            color_R += [0]
            color_G += [0]
            color_B += [0]
            bounding_box_IsSame += [0]
            FontWeight += [0]
            FontSize += [0]
            IsPrecededByWS += [0]
            IsPrecededByLineBreak += [0]
            IsClipped += [0]
            IsVisible += [0]

            bounding_Xe += [0]
            bounding_Ye += [0]
            bounding_We += [0]
            bounding_He += [0]
            isAnchor += [0]
            part += [0]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
            isImage += [0]
            bounding_X += [0]
            bounding_Y += [0]
            bounding_W += [0]
            bounding_H += [0]
            color_A += [0]
            color_R += [0]
            color_G += [0]
            color_B += [0]
            bounding_box_IsSame += [0]
            FontWeight += [0]
            FontSize += [0]
            IsPrecededByWS += [0]
            IsPrecededByLineBreak += [0]
            IsClipped += [0]
            IsVisible += [0]

            bounding_Xe += [0]
            bounding_Ye += [0]
            bounding_We += [0]
            bounding_He += [0]
            isAnchor += [0]
            part += [0]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids
            isImage = [0] + isImage
            bounding_X = [0] + bounding_X
            bounding_Y = [0] + bounding_Y
            bounding_W = [0] + bounding_W
            bounding_H = [0] + bounding_H
            color_A = [0] + color_A
            color_R = [0] + color_R
            color_G = [0] + color_G
            color_B = [0] + color_B
            bounding_box_IsSame = [0] + bounding_box_IsSame
            FontWeight = [0] + FontWeight
            FontSize = [0] + FontSize
            IsPrecededByWS = [0] + IsPrecededByWS
            IsPrecededByLineBreak = [0] + IsPrecededByLineBreak
            IsClipped = [0] + IsClipped
            IsVisible = [0] + IsVisible

            bounding_Xe = [0] + bounding_Xe
            bounding_Ye = [0] + bounding_Ye
            bounding_We = [0] + bounding_We
            bounding_He = [0] + bounding_He
            isAnchor = [0] + isAnchor
            part = [0] + part

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        if use_vocab_clip:
            assert len(vocab_clip_mapping) > 0
            input_ids = [vocab_clip_mapping[x] if vocab_clip_mapping.__contains__(x) else vocab_clip_mapping[3] for x in input_ids]
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([0] * padding_length) + label_ids
            isImage = ([0] * padding_length) + isImage
            bounding_X = ([0] * padding_length) + bounding_X
            bounding_Y = ([0] * padding_length) + bounding_Y
            bounding_W = ([0] * padding_length) + bounding_W
            bounding_H = ([0] * padding_length) + bounding_H
            color_A = ([0] * padding_length) + color_A
            color_R = ([0] * padding_length) + color_R
            color_G = ([0] * padding_length) + color_G
            color_B = ([0] * padding_length) + color_B
            bounding_box_IsSame = ([0] * padding_length) + bounding_box_IsSame
            FontWeight = ([0] * padding_length) + FontWeight
            FontSize = ([0] * padding_length) + FontSize
            IsPrecededByWS = ([0] * padding_length) + IsPrecededByWS
            IsPrecededByLineBreak = ([0] * padding_length) + IsPrecededByLineBreak
            IsClipped = ([0] * padding_length) + IsClipped
            IsVisible = ([0] * padding_length) + IsVisible

            bounding_Xe = ([0] * padding_length) + bounding_Xe
            bounding_Ye = ([0] * padding_length) + bounding_Ye
            bounding_We = ([0] * padding_length) + bounding_We
            bounding_He = ([0] * padding_length) + bounding_He
            isAnchor = ([0] * padding_length) + isAnchor
            part = ([0] * padding_length) + part
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length
            isImage += [0] * padding_length
            bounding_X += [0] * padding_length
            bounding_Y += [0] * padding_length
            bounding_W += [0] * padding_length
            bounding_H += [0] * padding_length
            color_A += [0] * padding_length
            color_R += [0] * padding_length
            color_G += [0] * padding_length
            color_B += [0] * padding_length
            bounding_box_IsSame += [0] * padding_length
            FontWeight += [0] * padding_length
            FontSize += [0] * padding_length
            IsPrecededByWS += [0] * padding_length
            IsPrecededByLineBreak += [0] * padding_length
            IsClipped += [0] * padding_length
            IsVisible += [0] * padding_length

            bounding_Xe += [0] * padding_length
            bounding_Ye += [0] * padding_length
            bounding_We += [0] * padding_length
            bounding_He += [0] * padding_length
            isAnchor += [0] * padding_length
            part += [0] * padding_length

            # bounding_X = bounding_Xe
            # bounding_Y = bounding_Ye
            # bounding_W = bounding_We
            # bounding_H = bounding_He

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(isImage) == max_seq_length
        assert len(bounding_X) == max_seq_length
        assert len(bounding_Y) == max_seq_length
        assert len(bounding_W) == max_seq_length
        assert len(bounding_H) == max_seq_length
        assert len(color_A) == max_seq_length
        assert len(color_R) == max_seq_length
        assert len(color_G) == max_seq_length
        assert len(color_B) == max_seq_length
        assert len(bounding_box_IsSame) == max_seq_length
        assert len(FontWeight) == max_seq_length
        assert len(FontSize) == max_seq_length
        assert len(IsPrecededByWS) == max_seq_length
        assert len(IsPrecededByLineBreak) == max_seq_length
        assert len(IsClipped) == max_seq_length
        assert len(IsVisible) == max_seq_length
        assert len(bounding_Xe) == max_seq_length
        assert len(bounding_Ye) == max_seq_length
        assert len(bounding_We) == max_seq_length
        assert len(bounding_He) == max_seq_length
        assert len(isAnchor) == max_seq_length
        assert len(part) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

            logger.info("isImage: %s", " ".join([str(x) for x in isImage]))
            logger.info("bounding_X: %s", " ".join([str(x) for x in bounding_X]))
            logger.info("bounding_Y: %s", " ".join([str(x) for x in bounding_Y]))
            logger.info("bounding_W: %s", " ".join([str(x) for x in bounding_W]))
            logger.info("bounding_H: %s", " ".join([str(x) for x in bounding_H]))
            logger.info("color_A: %s", " ".join([str(x) for x in color_A]))
            logger.info("color_R: %s", " ".join([str(x) for x in color_R]))
            logger.info("color_G: %s", " ".join([str(x) for x in color_G]))
            logger.info("color_B: %s", " ".join([str(x) for x in color_B]))
            logger.info("bounding_box_IsSame: %s", " ".join([str(x) for x in bounding_box_IsSame]))
            logger.info("FontWeight: %s", " ".join([str(x) for x in FontWeight]))
            logger.info("FontSize: %s", " ".join([str(x) for x in FontSize]))
            logger.info("IsPrecededByWS: %s", " ".join([str(x) for x in IsPrecededByWS]))
            logger.info("IsPrecededByLineBreak: %s", " ".join([str(x) for x in IsPrecededByLineBreak]))
            logger.info("IsClipped: %s", " ".join([str(x) for x in IsClipped]))
            logger.info("IsVisible: %s", " ".join([str(x) for x in IsVisible]))

            logger.info("bounding_Xe: %s", " ".join([str(x) for x in bounding_Xe]))
            logger.info("bounding_Ye: %s", " ".join([str(x) for x in bounding_Ye]))
            logger.info("bounding_We: %s", " ".join([str(x) for x in bounding_We]))
            logger.info("bounding_He: %s", " ".join([str(x) for x in bounding_He]))
            logger.info("isAnchor: %s", " ".join([str(x) for x in isAnchor]))
            logger.info("part: %s", " ".join([str(x) for x in part]))

        if "token_type_ids" not in tokenizer.model_input_names:
            segment_ids = None

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, label_ids=label_ids,
                isImage=isImage, bounding_X=bounding_X, bounding_Y=bounding_Y, bounding_W=bounding_W,
                bounding_H=bounding_H, color_A=color_A, color_R=color_R, color_G=color_G, color_B=color_B,
                bounding_box_IsSame=bounding_box_IsSame, FontWeight=FontWeight, FontSize=FontSize,
                IsPrecededByWS=IsPrecededByWS, IsPrecededByLineBreak=IsPrecededByLineBreak, IsClipped=IsClipped,
                IsVisible=IsVisible, bounding_Xe=bounding_Xe, bounding_Ye=bounding_Ye, bounding_We=bounding_We,
                bounding_He=bounding_He, isAnchor=isAnchor, part=part
            )
        )
    return features


class LabelingDataset(Dataset):
    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
    examples: List[InputExample]

    def __init__(self, data, tokenizer, labels, use_vocab_clip=False, vocab_clip_mapping={}, max_seq_length=512, config=None):
        model_type = 'xlm-roberta'
        self.examples = read_examples_from_file(data, config)
        self.features = convert_examples_to_features(
            self.examples,
            labels,
            max_seq_length,
            tokenizer,
            use_vocab_clip,
            vocab_clip_mapping,
            cls_token_at_end=bool(model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(tokenizer.padding_side == "left"),
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
            pad_token_label_id=-100,
            config=config
        )
        for feature in self.features:
            dict_tmp = {"input_ids": feature.input_ids, "attention_mask": feature.attention_mask,
                        "label": feature.label_ids, "visual_features": []}
            for isImage, bounding_X, bounding_Y, bounding_W, bounding_H, color_A, color_R, color_G, color_B, bounding_box_IsSame, FontWeight, FontSize, IsPrecededByWS, IsPrecededByLineBreak, IsClipped, IsVisible, bounding_Xe, bounding_Ye, bounding_We, bounding_He, isAnchor, part \
                    in zip(feature.isImage, feature.bounding_X, feature.bounding_Y, feature.bounding_W,
                           feature.bounding_H, feature.color_A, feature.color_R, feature.color_G,
                           feature.color_B, feature.bounding_box_IsSame, \
                           feature.FontWeight, feature.FontSize, feature.IsPrecededByWS,
                           feature.IsPrecededByLineBreak, feature.IsClipped, feature.IsVisible,
                           feature.bounding_Xe, feature.bounding_Ye, feature.bounding_We, feature.bounding_He,
                           feature.isAnchor, feature.part):
                dict_tmp["visual_features"].append(
                    [isImage, bounding_X, bounding_Y, bounding_W, bounding_H, color_A, color_R, color_G, color_B,
                     bounding_box_IsSame, FontWeight, FontSize, IsPrecededByWS, IsPrecededByLineBreak, IsClipped,
                     IsVisible, bounding_Xe, bounding_Ye, bounding_We, bounding_He, isAnchor, part])
            feature.visual_features = dict_tmp["visual_features"]
        d = 1
    def __len__(self):
        return len(self.features)

    def __get_examples__(self):
        return self.examples

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


class LabelingDataset_mt(Dataset):
    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
    examples: List[InputExample]

    def __init__(self, data, tokenizer, labels, use_vocab_clip=False, vocab_clip_mapping={}, max_seq_length=512, config=None):
        model_type = 'xlm-roberta'
        self.examples = read_examples_from_file(data, config)
        self.features = convert_examples_to_features_mt(
            self.examples,
            labels,
            max_seq_length,
            tokenizer,
            use_vocab_clip,
            vocab_clip_mapping,
            cls_token_at_end=bool(model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(tokenizer.padding_side == "left"),
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
            pad_token_label_id=-100,
            config=config
        )
        for feature in self.features:
            dict_tmp = {"input_ids": feature.input_ids, "attention_mask": feature.attention_mask,
                        "label": feature.label_ids, "visual_features": []}
            for isImage, bounding_X, bounding_Y, bounding_W, bounding_H, color_A, color_R, color_G, color_B, bounding_box_IsSame, FontWeight, FontSize, IsPrecededByWS, IsPrecededByLineBreak, IsClipped, IsVisible, bounding_Xe, bounding_Ye, bounding_We, bounding_He, isAnchor, part \
                    in zip(feature.isImage, feature.bounding_X, feature.bounding_Y, feature.bounding_W,
                           feature.bounding_H, feature.color_A, feature.color_R, feature.color_G,
                           feature.color_B, feature.bounding_box_IsSame, \
                           feature.FontWeight, feature.FontSize, feature.IsPrecededByWS,
                           feature.IsPrecededByLineBreak, feature.IsClipped, feature.IsVisible,
                           feature.bounding_Xe, feature.bounding_Ye, feature.bounding_We, feature.bounding_He,
                           feature.isAnchor, feature.part):
                dict_tmp["visual_features"].append(
                    [isImage, bounding_X, bounding_Y, bounding_W, bounding_H, color_A, color_R, color_G, color_B,
                     bounding_box_IsSame, FontWeight, FontSize, IsPrecededByWS, IsPrecededByLineBreak, IsClipped,
                     IsVisible, bounding_Xe, bounding_Ye, bounding_We, bounding_He, isAnchor, part])
            feature.visual_features = dict_tmp["visual_features"]
        d = 1
    def __len__(self):
        return len(self.features)

    def __get_examples__(self):
        return self.examples

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]



class LabelingDataset_sliding_window(Dataset):
    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
    examples: List[InputExample]

    def __init__(self, data, tokenizer, labels, use_vocab_clip=False, vocab_clip_mapping={}, max_seq_length=512, config=None, overlap_size=50):
        model_type = 'xlm-roberta'
        self.examples = read_examples_from_file_sliding_window(data, config)
        self.features = convert_examples_to_features_sliding_window(
            self.examples,
            labels,
            max_seq_length,
            tokenizer,
            use_vocab_clip,
            vocab_clip_mapping,
            cls_token_at_end=bool(model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(tokenizer.padding_side == "left"),
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
            pad_token_label_id=-100,
            config=config,
            overlap_size=overlap_size
        )
        print("start writing!")

    def __len__(self):
        return len(self.features)

    def __get_examples__(self):
        return self.examples

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]




def get_eval_dataloader(eval_dataset: Dataset, batch_size=20) -> DataLoader:
    data_loader = DataLoader(
        dataset=eval_dataset,
        sampler=None,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    return data_loader


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def inference(dataloader, model, args=None):
    # logger.info("***** Running %s *****")
    # logger.info("  Num examples = %d", len(dataloader.dataset))
    # logger.info("  Batch size = %d", dataloader.batch_size)
    preds: np.ndarray = None
    label_ids: np.ndarray = None
    session = None
    if args.test_onnx:
        session = rt.InferenceSession(model, providers=['CPUExecutionProvider'])
    else:
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model, args.device_ids)
        model.to(args.device)
        model.eval()
    for inputs in tqdm(dataloader, desc='TEST'):
        has_labels = any(inputs.get(k) is not None for k in ["labels", "masked_lm_labels"])
        for k in list(inputs.keys()):
            if k not in ['input_ids', 'attention_mask', 'visual_features', 'labels']:
                inputs.pop(k)
        for k, v in inputs.items():
            inputs[k] = v.to(args.device)
        if args.test_onnx:
            onnx_inputs = {session.get_inputs()[0].name: to_numpy(inputs['input_ids']),
                           session.get_inputs()[1].name: to_numpy(inputs['attention_mask']),
                           session.get_inputs()[2].name: to_numpy(inputs['visual_features'])}
            ort_outs = session.run(['output'], onnx_inputs)
            pred = ort_outs[0]
        else:
            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                else:
                    logits = outputs[0]
                pred = logits.detach().cpu().numpy()
        if preds is None:
            preds = pred
        else:
            preds = np.append(preds, pred, axis=0)
        if inputs.get("labels") is not None:
            if label_ids is None:
                label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                label_ids = np.append(label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
    return label_ids, preds


def overlapReduction(prediction_onnx, test_label_ids, overlap_size, examples):
    prediction_onnx_origin = []
    test_label_ids_origin = []
    for example in examples:
        prediction_onnx_tmp: np.ndarray = None
        test_label_ids_tmp: np.ndarray = None
        for index in example.xml_dic:
            if prediction_onnx_tmp is None:
                prediction_onnx_tmp = prediction_onnx[index]
            else:
                prediction_onnx_tmp = np.append(prediction_onnx_tmp[:-1], prediction_onnx[index][overlap_size + 1:], axis=0)
            if test_label_ids_tmp is None:
                test_label_ids_tmp = test_label_ids[index]
            else:
                assert all(test_label_ids_tmp[-overlap_size-1:-1] == test_label_ids[index][1:overlap_size + 1])
                test_label_ids_tmp = np.append(test_label_ids_tmp[:-1], test_label_ids[index][overlap_size + 1:], axis=0)
        # prediction_onnx_tmp = np.expand_dims(prediction_onnx_tmp, axis=0)
        # test_label_ids_tmp = np.expand_dims(test_label_ids_tmp, axis=0)
        prediction_onnx_origin.append(prediction_onnx_tmp)
        test_label_ids_origin.append(test_label_ids_tmp)
    return test_label_ids_origin, prediction_onnx_origin


def get_real_label(predictions: np.ndarray, label_ids: np.ndarray, label_map: Dict, label2id: Dict):
    predictions = softmax(predictions, axis=2)
    preds_image, preds_name, preds_price = predictions[:, :, label2id['B-MainImage']], predictions[:, :, label2id['B-Name']], predictions[:, :, label2id['B-Price']]
    tt = predictions[:,:,0]
    preds_max = np.max(predictions, axis=2)
    res = preds_max - tt
    preds_max, preds = np.max(predictions, axis=2) - predictions[:, :, 0], np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    out_label_list, preds_list, preds_max_list, preds_name_list, preds_price_list, preds_image_list = [[[] for _ in range(batch_size)] for i in range(6)]
    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])
                preds_max_list[i].append(preds_max[i][j])
                preds_image_list[i].append(preds_image[i][j])
                preds_name_list[i].append(preds_name[i][j])
                preds_price_list[i].append(preds_price[i][j])
    return preds_list, out_label_list, preds_max_list, preds_name_list, preds_price_list, preds_image_list


def get_real_label_sliding_window(predictions: np.ndarray, label_ids: np.ndarray, label_map: Dict, label2id: Dict):
    preds_list, label_list, preds_max_list, preds_name_list, preds_price_list, preds_image_list = [[[] for _ in range(len(predictions))] for i in range(6)]
    for index, pred in enumerate(predictions):
        pred = softmax(pred, axis=1)
        preds_image, preds_name, preds_price = pred[:, label2id['B-MainImage']], pred[:, label2id['B-Name']], pred[:, label2id['B-Price']]
        preds_max, pred = np.max(pred, axis=1), np.argmax(pred, axis=1)
        for offset in range(len(pred)):
            if label_ids[index][offset] != nn.CrossEntropyLoss().ignore_index:
                preds_list[index].append(label_map[pred[offset]])
                label_list[index].append(label_map[label_ids[index][offset]])
                preds_max_list[index].append(preds_max[offset])
                preds_image_list[index].append(preds_image[offset])
                preds_name_list[index].append(preds_name[offset])
                preds_price_list[index].append(preds_price[offset])
    return preds_list, label_list, preds_max_list, preds_name_list, preds_price_list, preds_image_list


def get_real_result_from_real_preds_add(real_preds, examples, preds_max_list, preds_name_list, preds_price_list, preds_image_list, real_label_ids):
    xml_dic = {}
    real_preds_ori, real_label_ids_ori, examples_ori, preds_max_ori, preds_name_ori, preds_price_ori, preds_image_ori, label_real_from_example_ori = [[] for i in range(8)]
    for index, example in enumerate(examples):
        if xml_dic.__contains__(example.xml_path) is False:
            xml_dic[example.xml_path] = [index]
        else:
            xml_dic[example.xml_path].append(index)
    for key in xml_dic:
        IsImage, preds_max, IsPrecededByWS, IsPrecededByLineBreak, IsSameBoundingBox, IsSameElement, preds_name, preds_price, preds_image = [[0 for i in range(250 * (len(xml_dic[key]) - 1))] for j in range(9)]
        words = ['' for i in range(250 * (len(xml_dic[key]) - 1))]
        label_pred, label_real_from_example, label_real = [['O' for i in range(250 * (len(xml_dic[key]) - 1))] for j in range(3)]
        for part, index in enumerate(xml_dic[key]):
            words[part * 250: part * 250 + len(examples[index].words_ori)] = examples[index].words_ori
            IsImage[part * 250: part * 250 + len(examples[index].isImage)] = examples[index].isImage
            label_real_from_example[part * 250: part * 250 + len(examples[index].labels)] = examples[index].labels
            label_real[part * 250: part * 250 + len(real_label_ids[index])] = real_label_ids[index]
            IsPrecededByWS[part * 250: part * 250 + len(examples[index].isImage)] = examples[index].IsPrecededByWS
            IsPrecededByLineBreak[part * 250: part * 250 + len(examples[index].isImage)] = examples[index].IsPrecededByLineBreak
            IsSameBoundingBox[part * 250: part * 250 + len(examples[index].isImage)] = examples[index].bounding_box_IsSame
            IsSameElement[part * 250: part * 250 + len(examples[index].isImage)] = examples[index].IsSameElement
            tmp, tmp_max, tmp_name, tmp_price, tmp_image = list(real_preds[index]), list(preds_max_list[index]), list(preds_name_list[index]), list(preds_price_list[index]), list(preds_image_list[index])
            for index_, lab in enumerate(label_pred[part * 250:]):
                if index_ == len(tmp):
                    break
                if tmp[index_] == 'O' and lab != 'O':
                    tmp[index_] = lab
                    tmp_max[index_] = preds_max[part * 250 + index_]
                    tmp_name[index_] = preds_name[part * 250 + index_]
                    tmp_price[index_] = preds_price[part * 250 + index_]
                    tmp_image[index_] = preds_image[part * 250 + index_]
            label_pred[part * 250: part * 250 + len(tmp)] = tmp
            preds_max[part * 250: part * 250 + len(tmp_max)] = tmp_max
            preds_name[part * 250: part * 250 + len(tmp_name)] = tmp_name
            preds_price[part * 250: part * 250 + len(tmp_price)] = tmp_price
            preds_image[part * 250: part * 250 + len(tmp_image)] = tmp_image
        examples_ori.append(InputExample_pred(xml_path=key, words=words, IsImage=IsImage, IsPrecededByWS=IsPrecededByWS, IsPrecededByLineBreak=IsPrecededByLineBreak, IsSameBoundingBox=IsSameBoundingBox, IsSameElement=IsSameElement))
        real_preds_ori.append(label_pred)
        preds_max_ori.append(preds_max)
        preds_name_ori.append(preds_name)
        preds_price_ori.append(preds_price)
        preds_image_ori.append(preds_image)
        label_real_from_example_ori.append(label_real_from_example)
        real_label_ids_ori.append(label_real)
    return real_preds_ori, examples_ori, preds_max_ori, preds_name_ori, preds_price_ori, preds_image_ori, label_real_from_example_ori, real_label_ids_ori


def get_real_result_from_real_preds_max(real_preds, real_label_ids, examples, preds_maxs, isprint=True):
    dict_info_label = {}
    dict_info_pred = {}
    # write_result = open("D:\\job\\NER\\RIKD\\datasets\\picl\\en\\result.tsv", 'w', encoding='utf-8')
    write_path_list_pred = False
    path_list, path_list_pred, huanyuan = '', '', {}
    if write_path_list_pred:
        # path_list = open("D:\\job\\NER\\RIKD\\datasets\\picl\\en\\path_list.tsv", 'r', encoding='utf-8').readlines()
        # path_list_pred = open("D:\\job\\NER\\RIKD\\datasets\\picl\\en\\path_list_pred.tsv", 'w', encoding='utf-8')
        huanyuan = {
            'MainImage': 'Main Image',
            'Name': 'Name',
            'ProductCode': 'ProductCode',
            'OutOfStock': 'OutOfStock',
            'Manufacturer': 'Manufacturer',
            'Price': 'Price',
            'Rating': 'Rating',
            'NumberofReviews': 'Number of Reviews'
        }

    if len(examples) != len(real_preds):
        print("len不一致!!!", len(examples), len(real_preds))
    write_res = False
    write_url = False
    real_preds_v = []
    real_label_v = []
    res = []
    real_results = {}
    res_label = []
    real_label_results = {}
    for index_i, real_pred in enumerate(real_preds):
        for index_j, label in enumerate(real_pred):
            if 'O' == label:
                continue
            key_v = label.split('-')
            if real_results.__contains__(key_v[1]) is False:
                real_results[key_v[1]] = []
            if 'B' == key_v[0]:
                real_results[key_v[1]].append(
                    [index_j, index_j, ' '.join(examples[index_i].words[index_j:index_j + 1])])
            if 'I' == key_v[0]:
                if len(real_results[key_v[1]]) == 0:
                    # real_results[key_v[1]].append([index_j, index_j, ' '.join(examples[index_i].words[index_j:index_j+1])])
                    continue
                else:
                    real_results[key_v[1]][-1][-2] = index_j
                    real_results[key_v[1]][-1][-1] = ' '.join(
                        examples[index_i].words[real_results[key_v[1]][-1][0]:index_j + 1])
        res.append(real_results)
        real_results = {}

        for index_j, label in enumerate(real_label_ids[index_i]):
            if 'O' == label:
                continue
            key_v = label.split('-')
            if real_label_results.__contains__(key_v[1]) is False:
                real_label_results[key_v[1]] = []
            if 'B' == key_v[0]:
                real_label_results[key_v[1]].append([index_j, index_j])
            if 'I' == key_v[0]:
                if not real_label_results[key_v[1]]:
                    real_label_results[key_v[1]].append([index_j, index_j])
                else:
                    real_label_results[key_v[1]][-1][-1] = index_j
        res_label.append(real_label_results)
        real_label_results = {}
    result_label = {}
    for index, label in enumerate(res_label):
        if isprint:
            print("************* LABEL " + str(index + 1) + " ***************")
        # print(test_url[index])
        # print(path_list[index])
        # print(examples[index].xml_path)
        path_de = "D:\\job\\rampup\\PICL\\EdgeProductModels2\\V1\\NewTest\\DE_hitapp\\test_xml_de\\".lower() + examples[index].xml_path
        path_fr = "D:\\job\\rampup\\PICL\\EdgeProductModels2\\V1\\NewTest\\FR_hitapp\\test_xml_fr\\".lower() + examples[index].xml_path
        path__ = examples[index].xml_path
        if os.path.exists(path_de):
            path__ = path_de
        elif os.path.exists(path_fr):
            path__ = path_fr
        if isprint:
            print(path__)
        url = ''
        real_path = examples[index].xml_path.replace('pickle', 'xml').replace('docs', 'xml')
        if os.path.exists(real_path):
            xml_file1 = open(real_path, 'r', encoding='utf-8')
            for linee in xml_file1:
                if 'property name=\"url\"' in linee:
                    url = linee[30:-5]
                    if isprint:
                        print(url)
                    break
        else:
            if os.path.exists(path__):
                xml_file1 = open(path__, 'r', encoding='utf-8')
                for linee in xml_file1:
                    if 'property name=\"url\"' in linee:
                        url = linee[30:-5]
                        if isprint:
                            print(url)
                        break
        dict_info_label[examples[index].xml_path] = {'url': url}
        dict_info_pred[examples[index].xml_path] = {'url': url}
        for key in label.keys():
            for start, end in label[key]:
                if dict_info_label[examples[index].xml_path].__contains__(key) is False:
                    dict_info_label[examples[index].xml_path][key] = [' '.join(examples[index].words[start: end + 1])]
                else:
                    dict_info_label[examples[index].xml_path][key].append(
                        ' '.join(examples[index].words[start: end + 1]))
                result_label[key] = ' '.join(examples[index].words[start: end + 1]) + " | " + str(start) + " " + str(
                    end)
                if '#IMAGE' == result_label[key]:
                    result_label[key] += str(start) + str(end)
                if isprint:
                    print(key, '|', result_label[key], '|', start, end)
        if isprint:
            print("*************  PRED " + str(index + 1) + " ***************")
        Name_offset = -1
        if 'Name' in res[index].keys() and len(res[index]['Name']) > 0:
            Name_offset = res[index]['Name'][0][0]
        for key in res[index].keys():
            for start, end, _ in res[index][key]:
                tmp_v = list(examples[index].words[start: end + 1])
                for index_tmp_, tmp in enumerate(tmp_v):
                    if len(tmp) == 1 and tmp.encode('utf-8') == b'\xe2\x80\x8f':
                        tmp_v[index_tmp_] = ''
                pred_tokens_text = ' '.join(tmp_v) + " | " + str(start) + " " + str(end)
                if dict_info_pred[examples[index].xml_path].__contains__(key) is False:
                    dict_info_pred[examples[index].xml_path][key] = [' '.join(tmp_v)]
                else:
                    dict_info_pred[examples[index].xml_path][key].append(' '.join(tmp_v))
                # if '#IMAGE' == pred_tokens_text:
                #     pred_tokens_text += str(start) + str(end)
                gap = (start - Name_offset) if Name_offset != -1 else -9999
                if gap > 200:
                    gap = str(gap) + (' !!!!!!!!!! ')
                if isprint:
                    print(key, '|', pred_tokens_text, '|',
                          (result_label.__contains__(key) and result_label[key] == pred_tokens_text), '|', gap, '|',
                          np.mean(preds_maxs[index][start: end + 1]))
    # if write_path_list_pred:
    #     for index, line in enumerate(res):
    #         # path_l = path_list[index].strip()
    #         path_l = examples[index].xml_path
    #         if '\\' not in path_l:
    #             path_de = "D:\\job\\rampup\\PICL\\EdgeProductModels2\\V1\\NewTest\\DE_hitapp\\test_xml_de\\".lower() + \
    #                       examples[index].xml_path
    #             path_fr = "D:\\job\\rampup\\PICL\\EdgeProductModels2\\V1\\NewTest\\FR_hitapp\\test_xml_fr\\".lower() + \
    #                       examples[index].xml_path
    #             if os.path.exists(path_de):
    #                 path_l = path_de
    #             elif os.path.exists(path_fr):
    #                 path_l = path_fr
    #         res__ = []
    #         for key in line.keys():
    #             for start, end, _ in line[key]:
    #                 res__.append('(' + huanyuan[key] + ':' + str(start) + '-' + str(end) + ')')
    #         path_list_pred.write(path_l + '\t' + ','.join(res__) + '\n')
    #     path_list_pred.close()

    if isprint:
        print(classification_report(real_label_ids, real_preds))
    ddd = {}
    for index__, xxxx in enumerate(res):
        ddd[examples[index__].xml_path] = xxxx
    return ddd


def get_real_result_from_real_preds(real_preds, examples, real_label):
    if len(examples) != len(real_preds):
        print("len(examples) != len(real_preds)", len(examples), len(real_preds))
    res, real_results, res_label, result_json = [], {}, [], {}
    for index_i, real_pred in enumerate(real_preds):
        for index_j, label in enumerate(real_pred):
            if 'O' == label:
                continue
            key_v = label.split('-')
            if real_results.__contains__(key_v[1]) is False:
                real_results[key_v[1]] = []
            if 'B' == key_v[0]:
                real_results[key_v[1]].append([index_j, index_j])
            if 'I' == key_v[0]:
                if len(real_results[key_v[1]]) == 0:
                    continue
                else:
                    real_results[key_v[1]][-1][-1] += 1
        real_results = {key: value for key, value in real_results.items() if value != []}
        res.append(real_results)
        real_results = {}
    for index in range(len(res)):
        print("*************  PRED " + str(index) + " ***************")
        print("File: " + examples[index].xml_path)
        result_json[examples[index].xml_path] = res[index]
        for key in res[index].keys():
            for index_ in range(len(res[index][key])):
                start, end = res[index][key][index_]
                pred_tokens_text = ' '.join(examples[index].words[start: end + 1])
                if '#IMAGE' == pred_tokens_text:
                    pred_tokens_text += str(start) + str(end)
                result_json[examples[index].xml_path][key][index_].append(pred_tokens_text)
                print(key, '|', pred_tokens_text, '|', start, end, '|')
    print(classification_report(real_label, real_preds))
    return result_json


def longestCommonSubstring(A, B):
    A = A.lower()
    B = B.lower()
    if not (A and B):
        return 0
    M, N = len(A), len(B)
    f = [[0 for i in range(N + 1)] for j in range(M + 1)]
    for i in range(M):
        for j in range(N):
            if A[i] == B[j]:
                f[i + 1][j + 1] = 1 + f[i][j]
    f = max(map(max, f))
    return f


def compare_number_for_price(location1, location2):
    number_a = [token for token in location1 if IsNumber(token)]
    number_b = [token for token in location2 if IsNumber(token)]
    return number_a == number_b


def compare(entity, location1, location2, words, isprint):
    if entity == 'MainImage' and abs(location2[0] - location1[0]) <= 5:
        if isprint:
            print("_iamge_", entity, words[location1[0]: location1[1] + 1], words[location2[0]: location2[1] + 1])
        return True
    if words[location1[0]: location1[1] + 1] == words[location2[0]: location2[1] + 1]:
        if isprint:
            print("_1_", entity, words[location1[0]: location1[1] + 1], words[location2[0]: location2[1] + 1])
        return True
    if (min(location1[1], location2[1]) - max(location1[0], location2[0]) + 1) / (
            location2[1] - location2[0] + 1) > 0.6:
        if isprint:
            print("_2_", entity, words[location1[0]: location1[1] + 1], words[location2[0]: location2[1] + 1])
        return True
    Longest = longestCommonSubstring(' '.join(words[location1[0]: location1[1] + 1]),
                                     ' '.join(words[location2[0]: location2[1] + 1]))
    if Longest / len(' '.join(words[location2[0]: location2[1] + 1])) > 0.6:
        if entity == 'Price' and compare_number_for_price(words[location1[0]: location1[1] + 1],
                                                          words[location2[0]: location2[1] + 1]) is False:
            if isprint:
                print("_4_", entity, words[location1[0]: location1[1] + 1], words[location2[0]: location2[1] + 1])
            return False
        if isprint:
            print("_3_", entity, words[location1[0]: location1[1] + 1], words[location2[0]: location2[1] + 1])
        return True
    if isprint:
        print("_4_", entity, words[location1[0]: location1[1] + 1], words[location2[0]: location2[1] + 1])
    return False


def get_metric(real_preds, real_label_ids, examples, isprint=False):
    real_results, real_label_results = {}, {}
    res_pred, res_label = [], []
    for index_i, real_pred in enumerate(real_preds):
        for index_j, label in enumerate(real_pred):
            if 'O' == label:
                continue
            key_v = label.split('-')
            if real_results.__contains__(key_v[1]) is False:
                real_results[key_v[1]] = []
            if 'B' == key_v[0]:
                real_results[key_v[1]].append([index_j, index_j])
            if 'I' == key_v[0]:
                if len(real_results[key_v[1]]) == 0:
                    # real_results[key_v[1]].append([index_j, index_j])
                    continue
                else:
                    real_results[key_v[1]][-1][-1] += 1
        res_pred.append(real_results)
        real_results = {}

        for index_j, label in enumerate(real_label_ids[index_i]):
            if 'O' == label:
                continue
            key_v = label.split('-')
            if real_label_results.__contains__(key_v[1]) is False:
                real_label_results[key_v[1]] = []
            if 'B' == key_v[0]:
                real_label_results[key_v[1]].append([index_j, index_j])
            if 'I' == key_v[0]:
                if not real_label_results[key_v[1]]:
                    real_label_results[key_v[1]].append([index_j, index_j])
                else:
                    real_label_results[key_v[1]][-1][-1] = index_j
        res_label.append(real_label_results)
        real_label_results = {}
    for index, pred in enumerate(res_pred):
        for key in pred:
            if res_label[index].__contains__(key) is False:
                continue
            labels = res_label[index][key][0]
            for number in range(len(pred[key])):
                if pred[key][number] != labels and compare(key, pred[key][number], labels, examples[index].words, isprint):
                    for offset in range(pred[key][number][0], pred[key][number][1] + 1):
                        real_preds[index][offset] = 'O'
                    for offset in range(labels[0], labels[1] + 1):
                        if offset == labels[0]:
                            real_preds[index][offset] = 'B-' + key
                        else:
                            real_preds[index][offset] = 'I-' + key
                    pred[key][number] = labels
    print(classification_report(real_label_ids, real_preds, digits=7))
    metrics_tsv = classification_report(real_label_ids, real_preds, digits=7)
    return real_preds, real_label_ids, metrics_tsv


def is_dot(s):
    dic = {
        ',': 1,
        '.': 1,
        '。': 1,
        '，': 1
    }
    return dic.__contains__(s)


def IsCurrency(s):
    dic = {
        '€': 1, 'CHF': 1, 'USD': 1, '$': 1, 'CNY': 1, '￥': 1, '¥': 1, '円': 1, 'Lei': 1, 'lei': 1,
        '₩': 1, '원': 1, '₣': 1, '৳': 1, '₡': 1, '£': 1, '₵': 1, '₫': 1, 'Ft': 1, 'kr': 1, 'zł': 1

    }
    return dic.__contains__(s)


def IsNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    if len(s) > 0:
        if s[-1] == '円':
            return IsNumber(s[:-1])

        if s[-4:] == 'Euro':
            return IsNumber(s[:-4])
        if s[-3:] == 'RON':
            return IsNumber(s[:-3])

    return False


def post_processing(real_preds, preds_maxs, examples_ori, preds_name_ori, preds_price_ori, preds_image_ori):
    entitys = ['Name', 'Price', 'MainImage']
    for index_i, preds in enumerate(real_preds):
        for start in range(len(preds) - 1):
            if preds[start] != 'O' and preds[start + 1] != 'O' and preds[start].split('-')[1] == preds[start + 1].split('-')[1] and preds[start + 1].split('-')[0] is 'B':
                # if preds[start + 1].split('-')[1] == 'Price':
                #     try:
                #         B_price_index = preds.index('B-Price')
                #     except:
                #         B_price_index = -1
                #     if preds_price_ori[index_i][B_price_index] < preds_price_ori[index_i][start + 1]: continue
                preds[start + 1] = 'I-' + preds[start + 1].split('-')[1]
        for entity in entitys:
            try:
                argindex = preds.index('B-' + entity)
                for index_j, pred in enumerate(preds):
                    if pred == 'I-' + entity and index_j <= argindex:
                        real_preds[index_i][index_j] = 'O'
                    if index_j > argindex and pred == 'I-' + entity and preds[index_j - 1] == 'O':
                        real_preds[index_i][index_j] = 'O'
            except:
                continue

    real_results = {}
    for index_i, real_pred in enumerate(real_preds):
        for index_j, label in enumerate(real_pred):
            if 'B-Manufacturer' == label and examples_ori[index_i].IsImage[index_j] == 1:
                real_preds[index_i][index_j] = 'O'
            if 'O' == label:
                continue
            key_v = label.split('-')
            if real_results.__contains__(key_v[1]) is False:
                real_results[key_v[1]] = []
            if 'B' == key_v[0]:
                real_results[key_v[1]].append([index_j, index_j])
            if 'I' == key_v[0]:
                if len(real_results[key_v[1]]) == 0:
                    continue
                else:
                    real_results[key_v[1]][-1][-1] += 1
        for key in real_results.keys():
            if key not in ['ProductCode'] and len(real_results[key]) > 1:
                for _index_ in range(len(real_results[key])):
                    start, end = real_results[key][_index_]
                    real_results[key][_index_].append(np.mean(preds_maxs[index_i][start: end + 1]))
                real_results[key] = sorted(real_results[key], key=lambda x: x[-1], reverse=False)
                for start, end, _ in real_results[key][:-1]:
                    for i in range(start, end + 1):
                        real_preds[index_i][i] = 'O'
        real_results = {}

    # coverage
    for index_i, real_pred in enumerate(real_preds):
        if 'B-MainImage' not in real_pred:
            argIndex = preds_image_ori[index_i].index(max(preds_image_ori[index_i][1:]))
            if preds_image_ori[index_i][argIndex] < 0.3 or examples_ori[index_i].IsImage[argIndex] != 1:
                continue
            real_pred[argIndex] = 'B-MainImage'
        if 'B-Name' not in real_pred:
            argIndex = preds_name_ori[index_i].index(max(preds_name_ori[index_i][1:]))
            if preds_name_ori[index_i][argIndex] < 0.3 or examples_ori[index_i].IsImage[argIndex] == 1:
                continue
            real_pred[argIndex] = 'B-Name'
            for index_name in range(argIndex + 1, len(real_pred)):
                if examples_ori[index_i].IsSameElement[index_name] == 1:
                    real_pred[index_name] = 'I-Name'
                    continue
                else:
                    break
        if 'B-Price' not in real_pred:
            try:
                B_name_index = real_pred.index('B-Name')
                dict_price = {index_t: _ for index_t, _ in enumerate(preds_price_ori[index_i]) if _ > 0.0001}
                for dict_price_key in dict_price.keys():
                    dict_price[dict_price_key] = 1 / (abs(dict_price_key - B_name_index) + 1) * dict_price[dict_price_key]
                argIndex = max(dict_price.keys(), key=(lambda k: dict_price[k]))
            except:
                argIndex = preds_price_ori[index_i].index(max(preds_price_ori[index_i][1:]))
            # argIndex = preds_price_ori[index_i].index(max(preds_price_ori[index_i][1:]))
            ddd = preds_price_ori[index_i][argIndex]
            # print(index_i, argIndex, "-----------", ddd)
            if preds_price_ori[index_i][argIndex] < 0.005 or examples_ori[index_i].IsImage[argIndex] == 1:
                continue
            real_pred[argIndex] = 'B-Price'
            for index_price in range(argIndex + 1, min(len(real_pred), len(examples_ori[index_i].IsSameElement))):
                if examples_ori[index_i].IsSameElement[index_price] == 1:
                    real_pred[index_price] = 'I-Price'
                    continue
                else:
                    break
    # optimize name_price_image
    for index_i, real_pred in enumerate(real_preds):
        real_preds[index_i] = ['O' if label in ['I-Price', 'I-MainImage'] else label for label in real_preds[index_i]]
        for entity in ['B-Price', 'B-Name']:
            try:
                argIndex = real_preds[index_i].index(entity)
            except:
                continue
            if argIndex > len(examples_ori[index_i].words) - 1:
                continue
            left, right = argIndex, argIndex + 1
            for left in range(argIndex, -1, -1):
                left_value, left_1_value = examples_ori[index_i].words[left], examples_ori[index_i].words[left - 1] if left > 0 else 0
                if 'Price' in entity:
                    if IsCurrency(left_value):
                        break
                    if left > 0 and IsCurrency(left_1_value):
                        continue
                    if left > 0 and IsNumber(left_value) and not IsCurrency(left_1_value):
                        break
                if examples_ori[index_i].IsSameElement[left] == 0:
                    break
            for right in range(argIndex + 1, min(len(real_pred), len(examples_ori[index_i].words))):
                right_value, right_1_value = examples_ori[index_i].words[right], examples_ori[index_i].words[right - 1]
                if 'Price' in entity:
                    if (right > argIndex + 1 and IsCurrency(right_1_value)) or (right - left > 1 and IsNumber(right_1_value) and IsNumber(right_value)):
                        break
                    if is_dot(right_value) or IsNumber(right_value) or IsCurrency(right_value):
                        continue
                    if not IsNumber(right_value) and not IsCurrency(right_value):
                        break
                if examples_ori[index_i].IsSameElement[right] == 0:
                    break
            if 'Price' in entity:
                if (right - 1 > left and IsCurrency(examples_ori[index_i].words[left]) and IsCurrency(examples_ori[index_i].words[right - 1])) or (right > left and is_dot(examples_ori[index_i].words[right - 1])):
                    right -= 1
            left, right = max(0, left), min(right, len(real_preds[index_i]))
            for index_new_range in range(left, right):
                real_preds[index_i][index_new_range] = entity if index_new_range == left else entity.replace('B-', 'I-')
            if 'Price' in entity:
                IsNumberForEachToken = [IsNumber(token_in) for token_in in examples_ori[index_i].words[left:right]]
                if any(IsNumberForEachToken) is False:
                    for index_new_range in range(left, right):
                        real_preds[index_i][index_new_range] = 'O'
    return real_preds


def coverage_calc(labels_vector):
    labels = ['O', 'B-MainImage', 'I-MainImage', 'B-Manufacturer', 'I-Manufacturer', 'B-Name', 'I-Name', 'B-Price',
              'I-Price', 'B-Rating', 'I-Rating', 'B-NumberofReviews', 'I-NumberofReviews', 'B-ProductCode',
              'I-ProductCode', 'B-OutOfStock', 'I-OutOfStock', "B-Address", "I-Address"]
    dic1 = {
        'MainImage': 0,
        'Manufacturer': 0,
        'Name': 0,
        'Price': 0,
        'Rating': 0,
        'NumberofReviews': 0,
        'ProductCode': 0,
        'OutOfStock': 0,
        'Address': 0
    }
    dic2 = {
        'MainImage': 0,
        'Manufacturer': 1,
        'Name': 2,
        'Price': 3,
        'Rating': 4,
        'NumberofReviews': 5,
        'ProductCode': 6,
        'OutOfStock': 7,
        "Address": 8
    }
    for line in labels_vector:
        true_v = [0 for i in range(len(dic1))]
        for n in line:
            if 'B-' in n:
                kkk = n.split('-')[1]
                if true_v[dic2[kkk]] == 0:
                    dic1[kkk] += 1
                    true_v[dic2[kkk]] = 1
    print(dic1)
    print(len(labels_vector))
    for key in dic1.keys():
        print(key, dic1[key], dic1[key] / len(labels_vector))