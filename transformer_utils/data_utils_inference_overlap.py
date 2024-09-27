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
import numpy as np
from scipy.special import softmax
import onnxruntime as rt
import json
from tqdm import tqdm

@dataclass
class InputExample_pred:
    xml_path: str
    words: List[str]
    isimages: List[str]



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
    visual_features: Optional[List[List[int]]] = None
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None
    logits: Optional[List[float]] = None


def trans_conll_file_to_overlap_file(conll_file_path):
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
                has_label = True
                for line_tmp in content:
                    if 'O' != line_tmp.split(' ')[1]:
                        has_label = True
                if has_label:
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
    if content and has_label:
        for part, start_index in enumerate(split_):
            overlap_conll_file.append(xml_path)
            for tmp_content in content[start_index: start_index + 400]:
                overlap_conll_file.append(tmp_content + ' ' + str(part) + '\n')
            if start_index + 400 >= len(content):
                break
    return overlap_conll_file


def read_examples_from_file(data, config):
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
                examples.append(InputExample(guid=f"{guid_index}", xml_path=xml_path, words=words, IsImage=IsImage, IsSameElement=IsSameElement, words_ori=words_ori, labels=labels, visual_features=visual_features, xml_dic=[]))
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
        examples.append(InputExample(guid=f"{guid_index}", xml_path=xml_path, IsImage=IsImage, IsSameElement=IsSameElement, words=words, words_ori=words_ori, labels=labels, visual_features=visual_features, xml_dic=[]))
    return examples


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
                  'I-OutOfStock']
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
        # if use_vocab_clip:
        #     assert len(vocab_clip_mapping) > 0
        #     input_ids = [vocab_clip_mapping[x] if vocab_clip_mapping.__contains__(x) else vocab_clip_mapping[3] for x in input_ids]
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        assert len(input_mask) == len(input_ids)
        assert len(visual_features) == len(input_ids)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))

        if "token_type_ids" not in tokenizer.model_input_names:
            segment_ids = None
        line_dict = {"input_ids": input_ids, "attention_mask": input_mask, "label": label_ids, "visual_features": visual_features}
        overlap_lines = overlap_func(line_dict, overlap_size, max_seq_length)
        for overlap_line in overlap_lines:
            example.xml_dic.append(len(features))
            features.append(
                InputFeatures(
                    input_ids=overlap_line["input_ids"], attention_mask=overlap_line["attention_mask"], token_type_ids=segment_ids, label_ids=overlap_line["label"], visual_features=overlap_line["visual_features"]
                )
            )

    return features


def overlap_reverse(prediction_onnx, test_label_ids, overlap_size, examples):
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


class LabelingDataset(Dataset):
    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
    examples: List[InputExample]

    def __init__(self, data, tokenizer, labels, use_vocab_clip=False, vocab_clip_mapping={}, max_seq_length=512, config=None, overlap_size=50):
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
            config=config,
            overlap_size=overlap_size
        )
        print("start writing!")
        # file_write = open(write_path, "w", encoding='utf-8')
        # for index_f, feature in enumerate(tqdm(self.features)):
        #     dict_tmp = {"input_ids": feature.input_ids, "attention_mask": feature.attention_mask,
        #                 "label": feature.label_ids, "visual_features": []}
        #     for isImage, bounding_X, bounding_Y, bounding_W, bounding_H, color_A, color_R, color_G, color_B, bounding_box_IsSame, FontWeight, FontSize, IsPrecededByWS, IsPrecededByLineBreak, IsClipped, IsVisible, bounding_Xe, bounding_Ye, bounding_We, bounding_He, isAnchor, part \
        #             in zip(feature.isImage, feature.bounding_X, feature.bounding_Y, feature.bounding_W,
        #                    feature.bounding_H, feature.color_A, feature.color_R, feature.color_G,
        #                    feature.color_B, feature.bounding_box_IsSame, \
        #                    feature.FontWeight, feature.FontSize, feature.IsPrecededByWS,
        #                    feature.IsPrecededByLineBreak, feature.IsClipped, feature.IsVisible,
        #                    feature.bounding_Xe, feature.bounding_Ye, feature.bounding_We, feature.bounding_He,
        #                    feature.isAnchor, feature.part):
        #         dict_tmp["visual_features"].append(
        #             [isImage, bounding_X, bounding_Y, bounding_W, bounding_H, color_A, color_R, color_G, color_B,
        #              bounding_box_IsSame, FontWeight, FontSize, IsPrecededByWS, IsPrecededByLineBreak, IsClipped,
        #              IsVisible, bounding_Xe, bounding_Ye, bounding_We, bounding_He, isAnchor, part])
        #     feature.visual_features = dict_tmp["visual_features"]
        #     if feature.visual_features != self.features2[index_f].visual_features:
        #         print("False!!!!!")
            # file_write.write(json.dumps(dict_tmp) + '\n')
        # file_write.close()

    def __len__(self):
        return len(self.features)

    def __get_examples__(self):
        return self.examples

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


def get_eval_dataloader(eval_dataset: Dataset) -> DataLoader:
    data_loader = DataLoader(
        dataset=eval_dataset,
        sampler=None,
        batch_size=4,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    return data_loader


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def inference(dataloader, device, onnx_path):
    logger.info("***** Running %s *****")
    logger.info("  Num examples = %d", len(dataloader.dataset))
    logger.info("  Batch size = %d", dataloader.batch_size)
    session = rt.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    preds_onnx: np.ndarray = None
    label_ids: np.ndarray = None
    for inputs in tqdm(dataloader, desc='TEST'):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        onnx_inputs = {session.get_inputs()[0].name: to_numpy(inputs['input_ids']),
                       session.get_inputs()[1].name: to_numpy(inputs['attention_mask']),
                       session.get_inputs()[2].name: to_numpy(inputs['visual_features'])}
        ort_outs = session.run(['output'], onnx_inputs)
        if preds_onnx is None:
            preds_onnx = ort_outs[0]
        else:
            preds_onnx = np.append(preds_onnx, ort_outs[0], axis=0)
        if inputs.get("labels") is not None:
            if label_ids is None:
                label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                label_ids = np.append(label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
    return label_ids, preds_onnx


def get_real_label(predictions: np.ndarray, label_ids: np.ndarray, label_map: Dict):
    predictions = softmax(predictions, axis=2)
    preds_max = np.max(predictions, axis=2)
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]
    preds_max_list = [[] for _ in range(batch_size)]
    for i in range(batch_size):
        k = 0
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                k += 1
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])
                preds_max_list[i].append(preds_max[i][j])
    return preds_list, out_label_list, preds_max_list


def get_real_label2(predictions: np.ndarray, label_ids: np.ndarray, label_map: Dict, label2id: Dict):
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


def get_real_result_from_real_preds_add(real_preds, examples, preds_max_list):
    xml_dic = {}
    real_preds_ori, real_label_ids_ori, examples_ori, preds_max_ori = [], [], [], []
    for index, example in enumerate(examples):
        if xml_dic.__contains__(example.xml_path) is False:
            xml_dic[example.xml_path] = [index]
        else:
            xml_dic[example.xml_path].append(index)
    for key in xml_dic:
        isimages, preds_max = [0 for i in range(250 * (len(xml_dic[key]) - 1))], [0.0 for i in range(250 * (len(xml_dic[key]) - 1))]
        words, label_pred, label_real_from_example = ['' for i in range(250 * (len(xml_dic[key]) - 1))], ['O' for i in range(250 * (len(xml_dic[key]) - 1))], ['O' for i in range(250 * (len(xml_dic[key]) - 1))]
        for part, index in enumerate(xml_dic[key]):
            words[part * 250: part * 250 + len(examples[index].words_ori)] = examples[index].words_ori
            isimages[part * 250: part * 250 + len(examples[index].isImage)] = examples[index].isImage
            label_real_from_example[part * 250: part * 250 + len(examples[index].labels)] = examples[index].labels
            tmp, tmp_max = list(real_preds[index]), list(preds_max_list[index])
            for index_, lab in enumerate(label_pred[part * 250:]):
                if index_ == len(tmp):
                    break
                if tmp[index_] == 'O' and lab != 'O':
                    tmp[index_] = lab
                    tmp_max[index_] = preds_max[part * 250 + index_]
            label_pred[part * 250: part * 250 + len(tmp)] = tmp
            preds_max[part * 250: part * 250 + len(tmp_max)] = tmp_max
        examples_ori.append(InputExample_pred(xml_path=key, words=words, isimages=isimages))
        real_preds_ori.append(label_pred)
        preds_max_ori.append(preds_max)
    return real_preds_ori, examples_ori, preds_max_ori


def get_real_result_from_real_preds(real_preds, examples):
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
    return result_json


def post_processing(real_preds, preds_maxs, examples_ori):
    for v in real_preds:
        for start in range(len(v) - 1):
            if v[start] != 'O' and v[start + 1] != 'O' and v[start].split('-')[1] == v[start + 1].split('-')[1] and v[start + 1].split('-')[0] is 'B':
                v[start + 1] = 'I-' + v[start + 1].split('-')[1]
    real_results = {}
    for index_i, real_pred in enumerate(real_preds):
        for index_j, label in enumerate(real_pred):
            if 'B-Manufacturer' == label and examples_ori[index_i].isimages[index_j] == 1:
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
                    real_results[key][_index_].append(mean(preds_maxs[index_i][start: end + 1]))
                real_results[key] = sorted(real_results[key], key=lambda x: x[-1], reverse=False)
                for start, end, _ in real_results[key][:-1]:
                    for i in range(start, end + 1):
                        real_preds[index_i][i] = 'O'
        real_results = {}
    return real_preds