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

import csv
import sys
import numpy as np
import logging
from collections import defaultdict
logger = logging.getLogger(__name__)

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score
    from sklearn.metrics import roc_auc_score
    from sklearn import metrics
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score

    _has_sklearn = True
except (AttributeError, ImportError) as e:
    logger.warning("To use data.metrics please install scikit-learn. See https://scikit-learn.org/stable/index.html")
    _has_sklearn = False

def is_sklearn_available():
    return _has_sklearn


def get_entities_bio(seq):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-PER']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return set([tuple(chunk) for chunk in chunks])


def f1_score_ee(true_entities, pred_entities):
    """Compute the F1 score for DeepEE."""
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return score


def precision_score_ee(true_entities, pred_entities):
    """Compute the precision for DeepEE."""
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)

    score = nb_correct / nb_pred if nb_pred > 0 else 0

    return score


def recall_score_ee(true_entities, pred_entities):
    """Compute the recall for DeepEE."""
    nb_correct = len(true_entities & pred_entities)
    nb_true = len(true_entities)

    score = nb_correct / nb_true if nb_true > 0 else 0

    return score

def classification_report(true_entities, pred_entities, digits=5):
    """Build a text report showing the main classification metrics."""
    name_width = 0
    d1 = defaultdict(set)
    d2 = defaultdict(set)
    for e in true_entities:
        d1[e[0]].add((e[1], e[2]))
        name_width = max(name_width, len(e[0]))
    for e in pred_entities:
        d2[e[0]].add((e[1], e[2]))

    last_line_heading = 'macro avg'
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

    ps, rs, f1s, s = [], [], [], []
    for type_name, type_true_entities in d1.items():
        type_pred_entities = d2[type_name]
        nb_correct = len(type_true_entities & type_pred_entities)
        nb_pred = len(type_pred_entities)
        nb_true = len(type_true_entities)

        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        report += row_fmt.format(*[type_name, p, r, f1, nb_true], width=width, digits=digits)

        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)

    report += u'\n'

    # compute averages
    report += row_fmt.format('micro avg',
                             precision_score_ee(true_entities, pred_entities),
                             recall_score_ee(true_entities, pred_entities),
                             f1_score_ee(true_entities, pred_entities),
                             np.sum(s),
                             width=width, digits=digits)
    report += row_fmt.format(last_line_heading,
                             np.average(ps, weights=s),
                             np.average(rs, weights=s),
                             np.average(f1s, weights=s),
                             np.sum(s),
                             width=width, digits=digits)

    return report


def topk_(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data,axis=axis)
        topk_data_sort = topk_data[topk_index_sort,row_index]
        topk_index_sort = topk_index[0:K,:][topk_index_sort,row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]
    return topk_data_sort, topk_index_sort
def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
    softmax = x_exp / x_exp_row_sum
    return softmax

if _has_sklearn:
    topn_list = [1,3,5]
    def simple_accuracy(preds, labels):
        prediction = softmax(preds)
        prediction = prediction[:, -1]
        try:
            auc = roc_auc_score(labels, prediction)
        except:
            auc = 0
        preds = np.argmax(preds, axis=1)
        precision = precision_score(labels, preds, average='binary')
        recall = recall_score(labels, preds, average='binary')
        acc = (preds == labels).mean()
        return [acc, precision, recall, auc]


    # def topic_model_metric(preds, labels):
    def simple_accuracy2(preds, labels):
        preds = np.argmax(preds, axis=1)

        return (preds == labels).mean()


    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }


    def calculate_AUC(prediction, label):
        prediction = softmax(prediction)
        prediction = prediction[:, -1]
        return roc_auc_score(label, prediction)


    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }


    def calculate_precision_recall(y_truth, y_pred):
        num_gt = len(y_truth)
        ret_dict = dict()
        # ret
        for topn in topn_list:
            y_topn = set(y_pred[0:topn])
            true_positive_tp = len(y_topn & set(y_truth))
            ret_dict["precision" + str(topn)] = true_positive_tp * 1.0 / float(topn)
            ret_dict["recall" + str(topn)] = true_positive_tp * 1.0 / float(num_gt)
        return ret_dict


    def topic_precision_recall(preds, labels):
        final_result_dict = dict()
        for topn in topn_list:
            final_result_dict["precision" + str(topn)] = 0.0
            final_result_dict["recall" + str(topn)] = 0.0

        top_tp_probs, pred_topic_list = topk_(preds, 5)

        label_topic_list = [np.argwhere(item == 1).reshape(-1).tolist() for item in labels]
        index = 0
        for y_pred, y_ground in zip(pred_topic_list, label_topic_list):
            try:
                pred_result = calculate_precision_recall(y_ground, y_pred.tolist())
                for topn in topn_list:
                    final_result_dict["precision" + str(topn)] = final_result_dict["precision" + str(topn)] + \
                                                                 pred_result["precision" + str(topn)]
                    final_result_dict["recall" + str(topn)] = final_result_dict["recall" + str(topn)] + pred_result[
                        "recall" + str(topn)]
                index += 1
            except:
                continue
        for key, value in final_result_dict.items():
            final_result_dict[key] = round(value*100/index, 2)
        return final_result_dict


    def macro_micro(preds, labels):
        preds = (preds > 0.5).astype('int')
        result_dict = dict()
        # print(preds)
        # print(labels)
        ma_p, ma_r, ma_f1, _ = metrics.precision_recall_fscore_support(labels, preds, average="macro",
                                                                       zero_division=True)
        mi_p, mi_r, mi_f1, _ = metrics.precision_recall_fscore_support(labels, preds, average="micro",
                                                                       zero_division=True)

        result_dict["macro_p"] = round(ma_p*100, 2)
        result_dict["macro_r"] = round(ma_r*100, 2)
        result_dict["macro_f1"] = round(ma_f1*100, 2)
        result_dict["micro_p"] = round(mi_p*100, 2)
        result_dict["micro_r"] = round(mi_r*100, 2)
        result_dict["micro_f1"] = round(mi_f1*100, 2)

        return result_dict

    def glue_compute_metrics(task_name, preds, labels, ids2label=None):
        assert len(preds) == len(labels)
        if task_name == "cola":
            return {"mcc": matthews_corrcoef(labels, preds)}
        elif task_name == "sst-2":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mrpc":
            return acc_and_f1(preds, labels)
        elif task_name == "sts-b":
            return pearson_and_spearman(preds, labels)
        elif task_name == "qqp":
            return acc_and_f1(preds, labels)
        elif task_name == "mnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mnli-mm":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "qnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "rte":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "wnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "quantus":
            return {"auc": calculate_AUC(preds, labels)}
        elif task_name == "pairwise":
            return {"auc": calculate_AUC(preds, labels)}
        elif task_name == "rankerbert":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "singlesentence":
            return {"acc": simple_accuracy(preds, labels)[0],"precision": simple_accuracy(preds, labels)[1],"recall": simple_accuracy(preds, labels)[2],"auc": simple_accuracy(preds, labels)[3]}
        elif "topic_model" in task_name:
            result_dict = dict()
            p_r = topic_precision_recall(preds, labels)
            result_dict.update(p_r)
            ma_mi_p_r = macro_micro(preds, labels)
            result_dict.update(ma_mi_p_r)
            return result_dict
        elif task_name == "language":
            return {"acc":simple_accuracy2(preds, labels)}
        elif task_name == "xml_ee_features_logits":
            pad_token_label_id = -100 # set to default, -100 is the padding label
            pred_ids = np.argmax(preds, axis=2)
            trues_list = [[] for _ in range(labels.shape[0])]
            preds_list = [[] for _ in range(pred_ids.shape[0])]
            for i in range(labels.shape[0]):
                for j in range(labels.shape[1]):
                    if labels[i, j] != pad_token_label_id:
                        trues_list[i].append(ids2label[labels[i][j]])
                        preds_list[i].append(ids2label[pred_ids[i][j]])
            print(trues_list[0])
            print(preds_list[0])
            true_entities = get_entities_bio(trues_list)
            pred_entities = get_entities_bio(preds_list)
            results = {
                "f1": f1_score_ee(true_entities, pred_entities),
                'report': classification_report(true_entities, pred_entities)
            }
            return results
        else:
            raise KeyError(task_name)


    def xnli_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name == "xnli":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)
