import json
import logging
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm
import collections
from scipy.stats import entropy

from transformers.file_utils import is_tf_available, is_torch_available
from transformers.models.bert.tokenization_bert import whitespace_tokenize
from transformer_utils.data_utils import DataProcessor
from transformers.data.metrics.squad_metrics import get_final_text
import hashlib
from scipy.stats import entropy
import collections

if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)

MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet"}


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def entropy_score(vec_s, vec_e):
    seqlen = np.where(vec_s == 0)[0]
    if seqlen.shape[0] == 0:
        seqlen = vec_s.shape[0]
    else:
        seqlen = seqlen[0]
    # start 1*150 vector [0, 0, 0, 1, 0, ...0] -> [0,1]
    # end 1*150 vector [0,0,0,0,0, 1, 0, ...,0 ] entropy() -> [0,1]
    return 1.0 - 0.5 * (entropy(vec_s, base=seqlen) + entropy(vec_e, base=seqlen))


def find_best_spans(scores, start, end, best_inds, best_scores,
                    continue_score_ratio=0.001, min_start_end_prod=0.005):
    # At least will find out one best_scores. No overlap.
    # best_score - largest outer production value in 150*150 (flat)
    best_score = np.partition(scores[start:end + 1, start:end + 1], -1, axis=None)[-1]
    if len(best_scores) > 0 \
            and best_score < max(best_scores[-1] * continue_score_ratio, min_start_end_prod):
        return
    # best_idx - position of largest outer production value in 150*150 (flat)
    best_idx = np.argpartition(scores[start:end + 1, start:end + 1], -1, axis=None)[-1]
    # s_idx, e_idx - start, end positions of largest outer production value in 150*150
    s_idx, e_idx = np.unravel_index(best_idx, scores[start:end + 1, start:end + 1].shape)
    best_scores.append(best_score)
    best_inds.append((start + s_idx, start + e_idx))  # s_index <= e_idx
    # before the best span
    if s_idx >= 1:  # avoid start > end
        find_best_spans(scores, start, start + s_idx - 1, best_inds, best_scores,
                        continue_score_ratio, min_start_end_prod)
    # after the best span
    if start + e_idx + 1 <= end:  # avoid start > end
        find_best_spans(scores, start + e_idx + 1, end, best_inds, best_scores,
                        continue_score_ratio, min_start_end_prod)

def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def write_msnet_one_predictions(example, features, results, n_best_size, max_answer_length, do_lower_case):
    continue_score_ratio = 0.001
    min_start_end_prod = 0.015  # minimum outer production of start probability and end probability
    prelim_predictions = []
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index_list", "end_index_list", "score_list", "pprod_list"])

    for feature_index, (feature, result) in enumerate(zip(features, results)):
        start_indexes = _get_best_indexes(result.start_logits, n_best_size)
        end_indexes = _get_best_indexes(result.end_logits, n_best_size)
        score_s = result.start_logits
        score_e = result.end_logits

        score_s = softmax(score_s)
        score_e = softmax(score_e)

        scores = np.outer(score_s,
                          score_e)  # Outer production, probability (start * end) of each span with a start-index and end-index
        scores = np.triu(scores)  # Upper triangle
        b_inds = []  # best scores' start and end position
        b_scores = []  # best scores, outer production
        find_best_spans(scores, 0, scores.shape[1] - 1, b_inds, b_scores,
                        continue_score_ratio=continue_score_ratio,
                        min_start_end_prod=min_start_end_prod)
        start_index_list = []
        end_index_list = []
        score_list = []
        pprod_list = []
        last_start = -1
        # last_end = -1
        for j in range(len(b_inds)):
            start_index, end_index = b_inds[j]
            # if last_end >= start_index:
            #     continue
            if start_index >= len(feature.tokens):
                continue
            if end_index >= len(feature.tokens):
                continue
            if start_index not in feature.token_to_orig_map:  # must in passage, not in query
                continue
            if end_index not in feature.token_to_orig_map:
                continue
            if not feature.token_is_max_context.get(start_index, False):
                continue
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > max_answer_length:  # too long answer
                continue

            start_index_list.append(start_index)
            end_index_list.append(end_index)
            score_list.append(entropy_score(score_s, score_e))  # entropy
            pprod_list.append(b_scores[j])  # probability

        if len(score_list) == 0:
            # if find no answers, directly use n_best start logits and end logits,
            # rather than their other production probability
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            score_s = softmax(result.start_logits)
            score_e = softmax(result.end_logits)
            start_end_logits = 0
            last_start = -1
            last_end = -1
            for start_index in start_indexes:
                if last_end >= start_index:
                    continue
                for end_index in end_indexes:
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    last_start = start_index
                    last_end = end_index
                    start_index_list.append(start_index)
                    end_index_list.append(end_index)
                    # score_list.append(entropy_score(score_s, score_e))
                    # pprod_list.append(score_s[start_index]*score_e[end_index])
                    score_list = [entropy_score(score_s, score_e)]
                    pprod_list = [score_s[start_index] * score_e[end_index]]
                    break
                if len(score_list) != 0:
                    break

        prelim_predictions.append(
            _PrelimPrediction(
                feature_index=feature_index,
                start_index_list=start_index_list,
                end_index_list=end_index_list,
                score_list=score_list,
                pprod_list=pprod_list))
        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["start_list", "length_list", "final_text_list", "score_list", "pprod_list"])
        # convert prediction results to raw texts
        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            start_list = []
            length_list = []
            final_text_list = []
            for start_index, end_index in zip(pred.start_index_list, pred.end_index_list):
                tok_tokens = feature.tokens[start_index:(end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[start_index]
                orig_doc_end = feature.token_to_orig_map[end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)
                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")
                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                # add by yz
                start = len(' '.join(example.doc_tokens[:orig_doc_start]))
                length = len(orig_text)
                final_text = get_final_text(tok_text, orig_text, do_lower_case)

                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True
                start_list.append(start)
                length_list.append(length)
                final_text_list.append(final_text)

            nbest.append(
                _NbestPrediction(
                    start_list=start_list,
                    length_list=length_list,
                    final_text_list=final_text_list,
                    score_list=pred.score_list,
                    pprod_list=pred.pprod_list
                ))
        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(start_list=[],
                                 length_list=[],
                                 final_text_list=[],
                                 score_list=[],
                                 pprod_list=[]))
        assert len(nbest) >= 1
        for (i, entry) in enumerate(nbest):
            hls = []
            for one_text, one_start, one_score, one_pprod in zip(entry.final_text_list, entry.start_list,
                                                                 entry.score_list, entry.pprod_list):
                hls.append({
                    'text': one_text,
                    'start': one_start,
                    'end': one_start + len(one_text),
                    'score': one_score,  # entropy score
                    'pprod': one_pprod  # probability, start prob * end prob
                })
            if len(hls) == 0:
                hls = []
                hls.append({
                    "text": "#",
                    "start": "",
                    "end": "",
                    "score": 0,
                    "pprod": 0
                })
            predict_result = {
                # 'query': example.line_data.split("\t")[0],
                # "url": example.line_data.split("\t")[1],
                # "passage": example.line_data.split("\t")[2],
                "answers": [{"highlights": hls}],
            }
            # result_line = "{}\t{}".format(example.line_data, json.dumps(predict_result))
            return predict_result


def highlight_convert_example_to_features(example, max_seq_length, doc_stride, max_query_length, is_training):
    features = []
    if is_training and not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
            return []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )

    spans = []

    truncated_query = tokenizer.encode(example.question_text, add_special_tokens=False, max_length=max_query_length)
    '''
    sequence_added_tokens = (
        tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
        if "roberta" in str(type(tokenizer))
        else tokenizer.model_max_length - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair
    '''
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (
        tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
        if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
        else tokenizer.model_max_length - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):

        encoded_dict = tokenizer.encode_plus(
            truncated_query if tokenizer.padding_side == "right" else span_doc_tokens,
            span_doc_tokens if tokenizer.padding_side == "right" else truncated_query,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            pad_to_max_length=True,
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            truncation_strategy="only_second" if tokenizer.padding_side == "right" else "only_first",
            return_token_type_ids = True
        )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                    len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1 :]

        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict:
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0) (not sure why...)
        p_mask = np.array(span["token_type_ids"])

        p_mask = np.minimum(p_mask, 1)

        if tokenizer.padding_side == "right":
            # Limit positive values to one
            p_mask = 1 - p_mask

        p_mask[np.where(np.array(span["input_ids"]) == tokenizer.sep_token_id)[0]] = 1

        # Set the CLS index to '0'
        p_mask[cls_index] = 0

        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        features.append(
            HighlightFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
            )
        )
    return features


def highlight_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def highlight_convert_examples_to_features(
    examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training, return_dataset=False, threads=1
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset,
            if 'tf': returns a tf.data.Dataset
        threads: multiple processing threadsa-smi
    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`
    Example::
        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)
        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """

    # Defining helper methods
    # features = []
    # print("~~~~~~~~~~~~~~~~~~~")
    # print(threads)
    # print(cpu_count())
    # threads = min(threads, cpu_count())
    # # print(threads)
    # with Pool(threads, initializer=highlight_convert_example_to_features_init, initargs=(tokenizer,)) as p:
    #     annotate_ = partial(
    #         highlight_convert_example_to_features,
    #         max_seq_length=max_seq_length,
    #         doc_stride=doc_stride,
    #         max_query_length=max_query_length,
    #         is_training=is_training,
    #     )
    #     features = list(
    #         tqdm(
    #             p.imap(annotate_, examples, chunksize=32),
    #             total=len(examples),
    #             desc="convert squad examples to features",
    #         )
    #     )
    highlight_convert_example_to_features_init(tokenizer)
    # Defining helper methods
    features = []
    for example in examples:
        part_features = highlight_convert_example_to_features(example, max_seq_length, doc_stride=doc_stride,
                                                             max_query_length=max_query_length, is_training=is_training)
        features.append(part_features)

    new_features = []
    #unique_id = 1000000000
    unique_id = 0
    example_index = 0
    for example_features in features:
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        return features
    elif return_dataset == "tf":
        if not is_tf_available():
            raise RuntimeError("TensorFlow must be installed to return a TensorFlow dataset.")

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    {
                        "start_position": ex.start_position,
                        "end_position": ex.end_position,
                        "cls_index": ex.cls_index,
                        "p_mask": ex.p_mask,
                    },
                )

        return tf.data.Dataset.from_generator(
            gen,
            (
                {"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32},
                {"start_position": tf.int64, "end_position": tf.int64, "cls_index": tf.int64, "p_mask": tf.int32},
            ),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                {
                    "start_position": tf.TensorShape([]),
                    "end_position": tf.TensorShape([]),
                    "cls_index": tf.TensorShape([]),
                    "p_mask": tf.TensorShape([None]),
                },
            ),
        )

    return features



class HighlightProcessor(DataProcessor):
    """
    Processor for the SQuAD data set.
    Overriden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and version 2.0 of SQuAD, respectively.
    """

    train_file = None
    dev_file = None

    def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
        if not evaluate:
            answer = tensor_dict["answers"]["text"][0].numpy().decode("utf-8")
            answer_start = tensor_dict["answers"]["answer_start"][0].numpy()
            answers = []
        else:
            answers = [
                {"answer_start": start.numpy(), "text": text.numpy().decode("utf-8")}
                for start, text in zip(tensor_dict["answers"]["answer_start"], tensor_dict["answers"]["text"])
            ]

            answer = None
            answer_start = None

        return HighlightExample(
            qas_id=tensor_dict["id"].numpy().decode("utf-8"),
            question_text=tensor_dict["question"].numpy().decode("utf-8"),
            context_text=tensor_dict["context"].numpy().decode("utf-8"),
            answer_text=answer,
            start_position_character=answer_start,
            title=tensor_dict["title"].numpy().decode("utf-8"),
            answers=answers,
        )
    def get_labels(self, label_file=None):
        """See base class."""
        return ["0", "1"]

    def get_examples_from_dataset(self, dataset, evaluate=False):
        """
        Creates a list of :class:`~transformers.data.processors.squad.SquadExample` using a TFDS dataset.
        Args:
            dataset: The tfds dataset loaded from `tensorflow_datasets.load("squad")`
            evaluate: boolean specifying if in evaluation mode or in training mode
        Returns:
            List of SquadExample
        Examples::
            import tensorflow_datasets as tfds
            dataset = tfds.load("squad")
            training_examples = get_examples_from_dataset(dataset, evaluate=False)
            evaluation_examples = get_examples_from_dataset(dataset, evaluate=True)
        """

        if evaluate:
            dataset = dataset["validation"]
        else:
            dataset = dataset["train"]

        examples = []
        for tensor_dict in tqdm(dataset, mininterval=100):
            examples.append(self._get_example_from_tensor_dict(tensor_dict, evaluate=evaluate))

        return examples

    def get_train_examples(self, data_dir, filename=None):
        """
        Returns the training examples from the data directory.
        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        if data_dir is None:
            data_dir = ""

        # if self.train_file is None:
        #     raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
            os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            return self._create_examples(reader, "train")


    def get_dev_examples(self, data_dir, filename=None):
        with open(os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8") as reader:
            data = reader.readlines()
        examples = []
        for line in data:
            eles = line.rstrip().split("\t")

            example = HighlightExample(
                qas_id=line.rstrip(),
                question_text=eles[self.args.query_index],
                context_text=eles[self.args.passage_index],
                answer_text=None,
                start_position_character=None,
                title=None,
                is_impossible=False,
                answers=[],
                line_data=line.rstrip()
            )

            examples.append(example)
        return examples

    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        for entry in tqdm(input_data, mininterval=100):
            split_line = entry.split("\t")
            question_text = split_line[self.args.query_index]
            context_text = split_line[self.args.passage_index]
            data = question_text + context_text
            qas_id = hashlib.md5(data.encode(encoding='UTF-8')).hexdigest()
            answer = json.loads(split_line[self.args.label_index])
            answers = []
            start_position_character = None
            answer_text = None

            if "is_impossible" in answer:
                is_impossible = answer["is_impossible"]
            else:
                is_impossible = False

            if not is_impossible:
                # if is_training:
                answer = answer["answers"][0]["highlights"][0]
                answer_text = answer["text"]
                start_position_character = answer["start"]



                # else:
                #     answers = answer["answers"]

            example = HighlightExample(
                qas_id=qas_id,
                question_text=question_text,
                context_text=context_text,
                answer_text=answer_text,
                start_position_character=start_position_character,
                title=question_text,
            )
            examples.append(example)
        return examples

    def parse_line(self, line, set_type="train"):
        line_dict = {}

        split_line = line.split("\t")
        text_a = split_line[self.args.query_index]
        text_b = split_line[self.args.passage_index]
        data = text_a + text_b
        qas_id = hashlib.md5(data.encode(encoding='UTF-8')).hexdigest()

        line_dict["qas_id"] = qas_id
        line_dict["question_text"] = text_a
        line_dict["context_text"] = text_b


        if set_type != "predict":
            label = split_line[self.args.label_index]
            answer = json.loads(label)
            answer = answer["answers"][0]["highlights"][0]
            answer_text = answer["text"]
            start_position_character = answer["start"]

            line_dict["answer_text"] = answer_text
            line_dict["start_position_character"] = start_position_character

        return line_dict

class HighlightExample(object):
    """
    A single training/test example for the Squad dataset, as loaded from disk.
    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text=None,
        start_position_character=None,
        title=None,
        answers=[],
        is_impossible=False,
        line_data=None
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers
        self.line_data = line_data

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start end end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]


class HighlightFeatures(object):
    """
    Single squad example features to be fed to a model.
    Those features are model-specific and can be crafted from :class:`~transformers.data.processors.squad.SquadExample`
    using the :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        start_position,
        end_position,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position


class HighlightResult(object):
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.
    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logits=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits



def get_msnet_predictions(all_examples, all_features, all_results, n_best_size,
                            max_answer_length, do_lower_case, hl_results):
    """Write final predictions to the json file."""
    # tf.logging.info("Writing predictions to: %s" % (output_prediction_file))
    # tf.logging.info("Writing nbest to: %s" % (output_nbest_file))
    no_answer_num = 0
    # f = open(predict_output_file, 'w', encoding='utf8')
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index_list", "end_index_list", "score_list", "pprod_list"])

    def entropy_score(vec_s, vec_e):
        seqlen = np.where(vec_s == 0)[0]
        if seqlen.shape[0] == 0:
            seqlen = vec_s.shape[0]
        else:
            seqlen = seqlen[0]
        # start 1*150 vector [0, 0, 0, 1, 0, ...0] -> [0,1]
        # end 1*150 vector [0,0,0,0,0, 1, 0, ...,0 ] entropy() -> [0,1]
        return 1.0 - 0.5 * (entropy(vec_s, base=seqlen) + entropy(vec_e, base=seqlen))

    def find_best_spans(scores, start, end, best_inds, best_scores,
                        continue_score_ratio=0.001, min_start_end_prod=0.005):
        # At least will find out one best_scores. No overlap.
        # best_score - largest outer production value in 150*150 (flat)
        best_score = np.partition(scores[start:end + 1, start:end + 1], -1, axis=None)[-1]
        if len(best_scores) > 0 \
                and best_score < max(best_scores[-1] * continue_score_ratio, min_start_end_prod):
            return
        # best_idx - position of largest outer production value in 150*150 (flat)
        best_idx = np.argpartition(scores[start:end + 1, start:end + 1], -1, axis=None)[-1]
        # s_idx, e_idx - start, end positions of largest outer production value in 150*150
        s_idx, e_idx = np.unravel_index(best_idx, scores[start:end + 1, start:end + 1].shape)
        best_scores.append(best_score)
        best_inds.append((start + s_idx, start + e_idx))  # s_index <= e_idx
        # before the best span
        if s_idx >= 1:  # avoid start > end
            find_best_spans(scores, start, start + s_idx - 1, best_inds, best_scores,
                            continue_score_ratio, min_start_end_prod)
        # after the best span
        if start + e_idx + 1 <= end: # avoid start > end
            find_best_spans(scores, start + e_idx + 1, end, best_inds, best_scores,
                            continue_score_ratio, min_start_end_prod)

    # for (example_index, example) in enumerate(tqdm(all_examples, desc="write result")):
    for (example_index, example) in enumerate(all_examples):

        continue_score_ratio = 0.001
        min_start_end_prod = 0.015 # minimum outer production of start probability and end probability

        features = example_index_to_features[example_index]
        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            score_s = result.start_logits
            score_e = result.end_logits

            score_s = softmax(score_s)
            score_e = softmax(score_e)

            scores = np.outer(score_s, score_e)  # Outer production, probability (start * end) of each span with a start-index and end-index
            scores = np.triu(scores)  # Upper triangle
            b_inds = []  # best scores' start and end position
            b_scores = []  # best scores, outer production
            find_best_spans(scores, 0, scores.shape[1] - 1, b_inds, b_scores,
                            continue_score_ratio=continue_score_ratio,
                            min_start_end_prod=min_start_end_prod)
            start_index_list = []
            end_index_list = []
            score_list = []
            pprod_list = []
            last_start = -1
            # last_end = -1

            for j in range(len(b_inds)):
                start_index, end_index = b_inds[j]
                # if last_end >= start_index:
                #     continue
                if start_index >= len(feature.tokens):
                    continue
                if end_index >= len(feature.tokens):
                    continue
                if start_index not in feature.token_to_orig_map:  # must in passage, not in query
                    continue
                if end_index not in feature.token_to_orig_map:
                    continue
                if not feature.token_is_max_context.get(start_index, False):
                    continue
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                if length > max_answer_length:  # too long answer
                    continue

                start_index_list.append(start_index)
                end_index_list.append(end_index)
                score_list.append(entropy_score(score_s, score_e))  # entropy
                pprod_list.append(b_scores[j])  # probability

            if len(score_list) == 0:
                # if find no answers, directly use n_best start logits and end logits,
                # rather than their other production probability
                start_indexes = _get_best_indexes(result.start_logits, n_best_size)
                end_indexes = _get_best_indexes(result.end_logits, n_best_size)
                score_s = softmax(result.start_logits)
                score_e = softmax(result.end_logits)
                start_end_logits = 0
                last_start = -1
                last_end = -1
                for start_index in start_indexes:
                    if last_end >= start_index:
                        continue
                    for end_index in end_indexes:
                        if start_index >= len(feature.tokens):
                            continue
                        if end_index >= len(feature.tokens):
                            continue
                        if start_index not in feature.token_to_orig_map:
                            continue
                        if end_index not in feature.token_to_orig_map:
                            continue
                        if not feature.token_is_max_context.get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > max_answer_length:
                            continue
                        last_start = start_index
                        last_end = end_index
                        start_index_list.append(start_index)
                        end_index_list.append(end_index)
                        # score_list.append(entropy_score(score_s, score_e))
                        # pprod_list.append(score_s[start_index]*score_e[end_index])
                        score_list = [entropy_score(score_s, score_e)]
                        pprod_list = [score_s[start_index] * score_e[end_index]]
                        break
                    if len(score_list) != 0:
                        break

            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=feature_index,
                    start_index_list=start_index_list,
                    end_index_list=end_index_list,
                    score_list=score_list,
                    pprod_list=pprod_list))

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["start_list", "length_list", "final_text_list", "score_list", "pprod_list"])

        # convert prediction results to raw texts
        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            start_list = []
            length_list = []
            final_text_list = []
            for start_index, end_index in zip(pred.start_index_list, pred.end_index_list):
                tok_tokens = feature.tokens[start_index:(end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[start_index]
                orig_doc_end = feature.token_to_orig_map[end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)
                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")
                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                # add by yz
                start = len(' '.join(example.doc_tokens[:orig_doc_start]))
                length = len(orig_text)
                final_text = get_final_text(tok_text, orig_text, do_lower_case)

                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True
                start_list.append(start)
                length_list.append(length)
                final_text_list.append(final_text)

            nbest.append(
                _NbestPrediction(
                    start_list=start_list,
                    length_list=length_list,
                    final_text_list=final_text_list,
                    score_list=pred.score_list,
                    pprod_list=pred.pprod_list
                ))
        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(start_list=[],
                                 length_list=[],
                                 final_text_list=[],
                                 score_list=[],
                                 pprod_list=[]))
        assert len(nbest) >= 1

        line_data = example.line_data

        # online schema
        # for (i, entry) in enumerate(nbest):
        #     predicitions = [{'Start': entry.start_list[idx],
        #                      'Score': entry.score_list[0],
        #                      'Length': entry.length_list[idx]} for idx in range(0, len(entry.start_list))]
        #     result_line = "{}\t{}\t{}".format(line_data, str(predicitions), str(entry.final_text_list))
        #     f.write(result_line + "\n")
        # highlight schema
        for (i, entry) in enumerate(nbest):
            hls = []
            for one_text, one_start, one_score, one_pprod in zip(entry.final_text_list, entry.start_list,
                                                                 entry.score_list, entry.pprod_list):
                hls.append({
                    'text': one_text,
                    'start': one_start,
                    'end': one_start + len(one_text),
                    'score': one_score,  # entropy score
                    'pprod': one_pprod  # probability, start prob * end prob
                })
            if len(hls) == 0:
                hls = []
                hls.append({
                    "text": "#",
                    "start": "",
                    "end": "",
                    "score": 0,
                    "pprod": 0
                })
                no_answer_num += 1

            predict_result = {
                # 'query': line_data.split("\t")[0],
                # "url": line_data.split("\t")[1],
                # "passage": line_data.split("\t")[2],
                "answers": [{"highlights": hls}],
            }
            # result_line = "{}\t{}".format(line_data, json.dumps(predict_result))
            result_line = "{}".format(json.dumps(predict_result))
            hl_results.append(result_line)
            break
    print("no answer num {}".format(no_answer_num))
    return hl_results