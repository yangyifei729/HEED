import transformers
from .traditional_modeling import *
from transformers import *
from torch.nn import CrossEntropyLoss, MSELoss, MultiLabelSoftMarginLoss, MarginRankingLoss, TripletMarginLoss, \
    BCEWithLogitsLoss
from transformers.modeling_utils import *
from transformers import (WEIGHTS_NAME, XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
                          DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer,
                          AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer,
                          XLMConfig, XLMTokenizer,
                          BertConfig, BertTokenizer,
                          RobertaConfig, RobertaTokenizer,
                          XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaTokenizerFast,
                          DebertaConfig, DebertaTokenizer, RobertaForQuestionAnswering)

import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np
from scipy.special import softmax

class BertForSequenceClassification(transformers.models.bert.BertForSequenceClassification):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config):
        super().__init__(config)
        self.finetuning_task = config.finetuning_task
        self.output_mode = config.output_mode
        self.use_embedding_head = config.use_embedding_head
        try:
            self.return_embeddings = config.return_embeddings
        except:
            self.return_embeddings = False
        if self.use_embedding_head:
            self.embeddingHead = nn.Linear(config.hidden_size, config.embedding_dim)

        # self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        if "mlm" in self.finetuning_task:
            self.classifier = transformers.models.bert.modeling_bert.BertOnlyMLMHead(config)
        elif "key_phrase" in self.finetuning_task:
            self.kernal_size = 5
            self.padding = int((self.kernal_size - 1) / 2)
            self.classifier = nn.Conv1d(config.hidden_size,
                                        config.num_labels,
                                        kernel_size=self.kernal_size,
                                        stride=1,
                                        padding=self.padding,
                                        dilation=1,
                                        groups=1,
                                        bias=True,
                                        padding_mode='zeros')
        elif "xml_ee" in self.finetuning_task:
            self.kernal_size = 5
            self.padding = int((self.kernal_size - 1) / 2)
            self.classifier = nn.Conv1d(config.hidden_size,
                                        config.num_labels,
                                        kernel_size=self.kernal_size,
                                        stride=1,
                                        padding=self.padding,
                                        dilation=1,
                                        groups=1,
                                        bias=True,
                                        padding_mode='zeros')
        else:
            if self.use_embedding_head:
                self.classifier = nn.Linear(config.embedding_dim, config.num_labels)
            else:
                self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def freeze_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
        if self.use_embedding_head:
            for param in self.embeddingHead.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        if self.use_embedding_head:
            pooled_output = self.embeddingHead(pooled_output)
        if self.return_embeddings:
            return pooled_output

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.output_mode == "regression":
                if self.finetuning_task == "topic_model":
                    loss_fct = MultiLabelSoftMarginLoss()

                    loss = loss_fct(logits, labels)
                elif self.finetuning_task == "topic_model1":
                    softmax_normalized_logits = F.softmax(logits, dim=-1)
                    loss = labels.detach().mul(torch.log(softmax_normalized_logits)).mul(-1).sum()
                elif self.finetuning_task == "topic_model2":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
                elif self.finetuning_task == "topic_model3":
                    softmax_normalized_logits = F.softmax(logits, dim=-1)
                    loss = labels.detach().mul(torch.log(softmax_normalized_logits)).mul(-1).sum()

                else:
                    loss_fct = MSELoss()
                    if self.num_labels == 1:
                        loss = loss_fct(torch.sigmoid(logits).view(-1), labels.view(-1))
                    else:
                        loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class XLMForSequenceClassification(transformers.models.xlm.XLMForSequenceClassification):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
        model = XLMForSequenceClassification.from_pretrained('xlm-mlm-en-2048')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """

    def __init__(self, config):
        super().__init__(config)
        self.finetuning_task = config.finetuning_task
        self.output_mode = config.output_mode

    def freeze_encoder(self):
        for param in self.transformer.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.transformer.parameters():
            param.requires_grad = True

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            langs=None,
            token_type_ids=None,
            position_ids=None,
            lengths=None,
            cache=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        transformer_outputs = self.transformer(input_ids,
                                               attention_mask=attention_mask,
                                               langs=langs,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               lengths=lengths,
                                               cache=cache,
                                               head_mask=head_mask,
                                               inputs_embeds=inputs_embeds)

        output = transformer_outputs[0]
        logits = self.sequence_summary(output)

        outputs = (logits,) + transformer_outputs[1:]  # Keep new_mems and attention/hidden states if they are here

        if labels is not None:
            if self.output_mode == "regression":
                if self.finetuning_task == "topic_model":
                    loss_fct = MultiLabelSoftMarginLoss()
                    loss = loss_fct(logits, labels)
                elif self.finetuning_task == "topic_model1":
                    softmax_normalized_logits = F.softmax(logits, dim=-1)
                    loss = labels.detach().mul(torch.log(softmax_normalized_logits)).mul(-1).sum()
                elif self.finetuning_task == "topic_model2":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
                elif self.finetuning_task == "topic_model3":
                    softmax_normalized_logits = F.softmax(logits, dim=-1)
                    loss = labels.detach().mul(torch.log(softmax_normalized_logits)).mul(-1).sum()

                else:
                    loss_fct = MSELoss()
                    if self.num_labels == 1:
                        loss = loss_fct(torch.sigmoid(logits).view(-1), labels.view(-1))
                    else:
                        loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs


# transformers.models.deberta.DebertaForSequenceClassification
class DebertaForSequenceClassification(transformers.models.deberta.DebertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.finetuning_task = config.finetuning_task
        self.output_mode = config.output_mode
        self.num_labels = config.num_labels
        output_dim = self.pooler.output_dim

        self.classifier = torch.nn.Linear(output_dim, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs[1:]

        # loss = None
        if labels is not None:
            if self.output_mode == "regression":
                if self.finetuning_task == "topic_model":
                    loss_fct = MultiLabelSoftMarginLoss()
                    loss = loss_fct(logits, labels)
                elif self.finetuning_task == "topic_model1":
                    softmax_normalized_logits = F.softmax(logits, dim=-1)
                    loss = labels.detach().mul(torch.log(softmax_normalized_logits)).mul(-1).sum()
                elif self.finetuning_task == "topic_model2":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
                elif self.finetuning_task == "topic_model3":
                    softmax_normalized_logits = F.softmax(logits, dim=-1)
                    loss = labels.detach().mul(torch.log(softmax_normalized_logits)).mul(-1).sum()

                else:
                    loss_fct = MSELoss()
                    if self.num_labels == 1:
                        loss = loss_fct(torch.sigmoid(logits).view(-1), labels.view(-1))
                    else:
                        loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs
        return outputs
        # else:
        #     return SequenceClassifierOutput(
        #         loss=loss,
        #         logits=logits,
        #         hidden_states=outputs.hidden_states,
        #         attentions=outputs.attentions,
        #     )


class RobertaForSequenceClassification(transformers.models.roberta.RobertaForSequenceClassification):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    config_class = RobertaConfig
    # pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.finetuning_task = config.finetuning_task
        self.output_mode = config.output_mode

        if "key_phrase" in self.finetuning_task:
            self.kernal_size = 5
            self.padding = int((self.kernal_size - 1) / 2)
            self.classifier = nn.Conv1d(config.hidden_size,
                                        self.num_labels,
                                        kernel_size=self.kernal_size,
                                        stride=1,
                                        padding=self.padding,
                                        dilation=1,
                                        groups=1,
                                        bias=True,
                                        padding_mode='zeros')
        elif "xml_ee" in self.finetuning_task:
            self.kernal_size = 5
            self.padding = int((self.kernal_size - 1) / 2)
            self.classifier = nn.Conv1d(config.hidden_size,
                                        config.num_labels,
                                        kernel_size=self.kernal_size,
                                        stride=1,
                                        padding=self.padding,
                                        dilation=1,
                                        groups=1,
                                        bias=True,
                                        padding_mode='zeros')
        # print(config.bert_trt)
        self.enable_trt = False
        try:

            self.enable_trt = config.enable_trt
            self.hidden_size = config.hidden_size
            self.max_seq_length = config.max_seq_length
            # logger.info("enable trt")
        except:
            logger.info("No enable trt")
        # print(self.hidden_size)
        if self.enable_trt:
            self.eval_batch_size = config.eval_batch_size
            # logger.info("enable trt")
            print("BERT-TRT is enabled")
            dir_path = os.path.dirname(os.path.realpath(__file__))
            os.environ['PATH'] = os.path.join(dir_path, '../../bert_trt') + ';' + os.environ['PATH']
            self.bert_lib = ctypes.CDLL('bert_trt.dll')

            self.dlis_bert_init = self.bert_lib.DLIS_BERT_INIT
            self.dlis_bert_init.argtypes = [ctypes.c_char_p]
            self.dlis_bert_init.restype = None

            self.dlis_bert = self.bert_lib.DLIS_BERT
            self.dlis_bert = self.bert_lib.DLIS_BERT
            self.dlis_bert.argtypes = [ndpointer(ctypes.c_int), ndpointer(ctypes.c_int), ndpointer(ctypes.c_float),
                                       ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_float), ctypes.c_int,
                                       ndpointer(ctypes.c_float), ctypes.c_int, ndpointer(ctypes.c_float), ctypes.c_int]
            self.dlis_bert.restype = None

            self.dlis_bert_init(ctypes.c_char_p(b"./bert_trt/bert_trt.ini"))

            self.output = np.zeros((self.eval_batch_size, self.max_seq_length, self.hidden_size)).astype("float32")

    def freeze_encoder(self):
        for param in self.roberta.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.roberta.parameters():
            param.requires_grad = True

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        if self.enable_trt:
            input_shape = input_ids.size()
            if attention_mask is None:
                attention_mask = torch.ones(input_shape, device=input_ids.device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

            inp_np = input_ids.cpu().numpy().astype("int32")
            segment_np = token_type_ids.cpu().numpy().astype("int32")
            mask_np = attention_mask.cpu().numpy().astype("float32")
            eval_batch_size = inp_np.shape[0]
            self.dlis_bert(inp_np, segment_np, mask_np, eval_batch_size, self.max_seq_length, self.output,
                           self.hidden_size, self.output, self.hidden_size, self.output, self.hidden_size)
            out_tensor = torch.from_numpy(self.output[:eval_batch_size]).to(input_ids.device)
            outputs = (out_tensor,)
        else:
            outputs = self.roberta(input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask)
        sequence_output = outputs[0]
        if "key_phrase" in self.finetuning_task or "xml_ee" in self.finetuning_task:
            logits = self.classifier(sequence_output.transpose(1, 2)).transpose(1, 2)  # bz *length * label number
        else:
            logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.output_mode == "regression":
                if self.finetuning_task == "topic_model":
                    loss_fct = MultiLabelSoftMarginLoss()
                    loss = loss_fct(logits, labels)
                elif self.finetuning_task == "topic_model1":
                    softmax_normalized_logits = F.softmax(logits, dim=-1)
                    loss = labels.detach().mul(torch.log(softmax_normalized_logits)).mul(-1).sum()
                elif self.finetuning_task == "topic_model2":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
                elif self.finetuning_task == "topic_model3":
                    softmax_normalized_logits = F.softmax(logits, dim=-1)
                    loss = labels.detach().mul(torch.log(softmax_normalized_logits)).mul(-1).sum()
                else:
                    loss_fct = MSELoss()
                    if self.num_labels == 1:
                        loss = loss_fct(torch.sigmoid(logits).view(-1), labels.view(-1))
                    else:
                        loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                if "key_phrase" in self.finetuning_task:
                    loss_fct = CrossEntropyLoss()
                    if attention_mask is not None:
                        active_loss = attention_mask.view(-1) == 1
                        active_logits = logits.contiguous().view(-1, self.num_labels)
                        active_labels = torch.where(
                            active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                        )
                        loss = loss_fct(active_logits, active_labels)
                    else:
                        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif "xml_ee" in self.finetuning_task:
                    loss_fct = CrossEntropyLoss(ignore_index=1)
                    loss = loss_fct(logits.reshape(-1, self.num_labels), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class XLMRobertaForSequenceClassification(RobertaForSequenceClassification):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-large')
        input_ids = torch.tensor(tokenizer.encode("Schloß Nymphenburg ist sehr schön .")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    config_class = XLMRobertaConfig
    # pretrained_model_archive_map = transformers.modeling_xlm_roberta.XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


class RobertaForQuestionAnswering(transformers.models.bert.BertPreTrainedModel):
    r"""
        **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        **start_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        **end_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        model = RobertaForQuestionAnswering.from_pretrained('roberta-large')
        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_ids = tokenizer.encode(question, text)
        start_scores, end_scores = model(torch.tensor([input_ids]))
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
    """
    config_class = RobertaConfig
    # pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.enable_trt = False
        try:

            self.enable_trt = config.enable_trt
            self.hidden_size = config.hidden_size
            self.max_seq_length = config.max_seq_length
            # logger.info("enable trt")

        except:
            logger.info("No enable trt")
        # print(self.hidden_size)
        if self.enable_trt:
            self.eval_batch_size = config.eval_batch_size
            # logger.info("enable trt")
            print("BERT-TRT is enabled")
            dir_path = os.path.dirname(os.path.realpath(__file__))
            os.environ['PATH'] = os.path.join(dir_path, '../../bert_trt') + ';' + os.environ['PATH']
            self.bert_lib = ctypes.CDLL('bert_trt.dll')

            self.dlis_bert_init = self.bert_lib.DLIS_BERT_INIT
            self.dlis_bert_init.argtypes = [ctypes.c_char_p]
            self.dlis_bert_init.restype = None

            self.dlis_bert = self.bert_lib.DLIS_BERT
            self.dlis_bert = self.bert_lib.DLIS_BERT
            self.dlis_bert.argtypes = [ndpointer(ctypes.c_int), ndpointer(ctypes.c_int), ndpointer(ctypes.c_float),
                                       ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_float), ctypes.c_int,
                                       ndpointer(ctypes.c_float), ctypes.c_int, ndpointer(ctypes.c_float), ctypes.c_int]
            self.dlis_bert.restype = None

            self.dlis_bert_init(ctypes.c_char_p(b"./bert_trt/bert_trt.ini"))

            self.output = np.zeros((self.eval_batch_size, self.max_seq_length, self.hidden_size)).astype("float32")

        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            start_positions=None,
            end_positions=None,
    ):

        if self.enable_trt:
            input_shape = input_ids.size()
            if attention_mask is None:
                attention_mask = torch.ones(input_shape, device=input_ids.device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

            inp_np = input_ids.cpu().numpy().astype("int32")
            segment_np = token_type_ids.cpu().numpy().astype("int32")
            mask_np = attention_mask.cpu().numpy().astype("float32")
            eval_batch_size = inp_np.shape[0]
            self.dlis_bert(inp_np, segment_np, mask_np, eval_batch_size, input_ids.numel(), self.output,
                           self.hidden_size, self.output, self.hidden_size, self.output, self.hidden_size)
            out_tensor = torch.from_numpy(self.output).to(input_ids.device)
            outputs = (out_tensor,)
        else:
            outputs = self.roberta(input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask)

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


class XLMRobertaForQuestionAnswering(RobertaForQuestionAnswering):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-large')
        input_ids = torch.tensor(tokenizer.encode("Schloß Nymphenburg ist sehr schön .")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    config_class = XLMRobertaConfig
    # pretrained_model_archive_map = transformers.modeling_xlm_roberta.XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


################################################### BEGIN Multi-task #############################################################


class BertForSequenceClassificationMT(transformers.models.bert.BertForSequenceClassification):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        # self.roberta = RobertaModel(config)
        self.finetuning_task = config.finetuning_task
        self.output_mode = config.output_mode
        self.finetuning_task_list = config.finetuning_task_list
        self.output_mode_list = config.output_mode_list
        self.num_labels_list = config.num_labels_list
        self.use_embedding_head = config.use_embedding_head
        try:
            self.return_embeddings = config.return_embeddings
        except:
            self.return_embeddings = False
        if self.use_embedding_head:
            self.embeddingHead = nn.Linear(config.hidden_size, config.embedding_dim)

        self.classifier_heads = list()
        for task_index, task_name in enumerate(self.finetuning_task_list):

            num_labels = self.num_labels_list[task_index]
            config.num_labels = num_labels
            # if "mrc" in task_name:
            #     self.classifier_heads.append(nn.Linear(config.hidden_size, num_labels))
            # elif "mlm" in task_name:
            #     self.classifier_heads.append(nn.Linear(config.hidden_size, self.config.vocab_size))
            # elif "key_phrase" in task_name:
            #     self.classifier_heads.append(nn.Linear(config.hidden_size, num_labels))
            # else:
            #     self.classifier_heads.append(nn.Linear(config.hidden_size, num_labels))
            if "mlm" in task_name:
                self.classifier_heads.append(transformers.models.bert.modeling_bert.BertOnlyMLMHead(config))
            elif "key_phrase" in task_name:
                self.kernal_size = 5
                self.padding = int((self.kernal_size - 1) / 2)
                self.classifier_heads.append(nn.Conv1d(config.hidden_size,
                                                       num_labels,
                                                       kernel_size=self.kernal_size,
                                                       stride=1,
                                                       padding=self.padding,
                                                       dilation=1,
                                                       groups=1,
                                                       bias=True,
                                                       padding_mode='zeros'))
            else:
                if self.use_embedding_head:
                    self.classifier_heads.append(nn.Linear(config.embedding_dim, num_labels))
                else:
                    self.classifier_heads.append(nn.Linear(config.hidden_size, num_labels))
        self.classifier_heads = nn.ModuleList(self.classifier_heads)
        config.num_labels = 1

        self.init_weights()

    def freeze_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
        if self.use_embedding_head:
            for param in self.embeddingHead.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            task_index=None
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        if self.use_embedding_head:
            pooled_output = self.embeddingHead(pooled_output)
        if self.return_embeddings:
            return pooled_output
        pooled_output = self.dropout(pooled_output)

        if task_index == None or isinstance(task_index, list):
            logits = list()
            for task_index, task_name in enumerate(self.finetuning_task_list):
                if "mrc" in task_name or "mlm" in task_name:
                    logits.append(self.classifier_heads[task_index](outputs[0]))
                elif "key_phrase" in task_name:
                    logits.append(self.classifier_heads[task_index](outputs[0].transpose(1, 2)).transpose(1, 2))
                else:
                    logits.append(self.classifier_heads[task_index](pooled_output))
            outputs = (logits,) + outputs[2:]
            return outputs

        output_mode = self.output_mode_list[task_index]
        num_labels = self.num_labels_list[task_index]
        finetuning_task = self.finetuning_task_list[task_index]
        if "mrc" in finetuning_task or "mlm" in finetuning_task:
            logits = self.classifier_heads[task_index](outputs[0])
        elif "key_phrase" in finetuning_task:
            # logits = self.classifier(sequence_output.transpose(1, 2)).transpose(1, 2)
            logits = self.classifier_heads[task_index](outputs[0].transpose(1, 2)).transpose(1, 2)
        else:
            logits = self.classifier_heads[task_index](pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if output_mode == "regression":
                if finetuning_task == "topic_model":
                    loss_fct = MultiLabelSoftMarginLoss()
                    loss = loss_fct(logits, labels)
                elif finetuning_task == "topic_model1":
                    softmax_normalized_logits = F.softmax(logits, dim=-1)
                    loss = labels.detach().mul(torch.log(softmax_normalized_logits)).mul(-1).sum()
                elif finetuning_task == "topic_model2":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits.view(-1, num_labels), labels.view(-1, num_labels))
                elif finetuning_task == "topic_model3":
                    softmax_normalized_logits = F.softmax(logits, dim=-1)
                    loss = labels.detach().mul(torch.log(softmax_normalized_logits)).mul(-1).sum()
                else:
                    loss_fct = MSELoss()
                    if num_labels == 1:
                        loss = loss_fct(torch.sigmoid(logits).view(-1), labels.view(-1))
                    else:
                        loss = loss_fct(logits.view(-1), labels.view(-1))
            elif output_mode == "classification":
                if "mrc" in finetuning_task:
                    # for multiple label classification, like question answering
                    start_logits, end_logits = logits
                    start_positions, end_positions = labels
                    if start_positions is not None and end_positions is not None:
                        # If we are on multi-GPU, split add a dimension
                        if len(start_positions.size()) > 1:
                            start_positions = start_positions.squeeze(-1)
                        if len(end_positions.size()) > 1:
                            end_positions = end_positions.squeeze(-1)
                        # sometimes the start/end positions are outside our model inputs, we ignore these terms
                        ignored_index = start_logits.size(1)
                        start_positions.clamp_(0, ignored_index)
                        end_positions.clamp_(0, ignored_index)
                        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

                        start_loss = loss_fct(start_logits, start_positions)
                        end_loss = loss_fct(end_logits, end_positions)
                        loss = (start_loss + end_loss) / 2
                elif "mlm" in finetuning_task:
                    loss_fct = CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
                elif "key_phrase" in finetuning_task:
                    loss_fct = CrossEntropyLoss()
                    if attention_mask is not None:
                        active_loss = attention_mask.view(-1) == 1
                        active_logits = logits.contiguous().view(-1, num_labels)
                        active_labels = torch.where(
                            active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                        )
                        loss = loss_fct(active_logits, active_labels)
                    else:
                        loss = loss_fct(logits.contiguous().view(-1, num_labels), labels.contiguous().view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class XLMForSequenceClassificationMT(transformers.models.xlm.XLMForSequenceClassification):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
        model = XLMForSequenceClassification.from_pretrained('xlm-mlm-en-2048')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config):
        super().__init__(config)
        self.finetuning_task = config.finetuning_task
        self.output_mode = config.output_mode
        self.finetuning_task_list = config.finetuning_task_list
        self.output_mode_list = config.output_mode_list
        self.num_labels_list = config.num_labels_list

        self.classifier_heads = list()
        for task_index, task_name in enumerate(self.finetuning_task_list):
            num_labels = self.num_labels_list[task_index]
            config.num_labels = num_labels
            if "mrc" in task_name:
                self.classifier_heads.append(transformers.models.xlm.modeling_xlm.SQuADHead(config))
            elif "mlm" in task_name:
                self.classifier_heads.append(nn.Linear(config.hidden_size, self.config.vocab_size))
            else:
                self.classifier_heads.append(SequenceSummary(config))
        self.classifier_heads = nn.ModuleList(self.classifier_heads)
        config.num_labels = 1

    def freeze_encoder(self):
        for param in self.transformer.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.transformer.parameters():
            param.requires_grad = True

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            langs=None,
            token_type_ids=None,
            position_ids=None,
            lengths=None,
            cache=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None, task_index=0
    ):

        transformer_outputs = self.transformer(input_ids,
                                               attention_mask=attention_mask,
                                               langs=langs,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               lengths=lengths,
                                               cache=cache,
                                               head_mask=head_mask,
                                               inputs_embeds=inputs_embeds)

        output = transformer_outputs[0]

        if task_index == None or isinstance(task_index, list):
            logits = list()
            for task_index, task_name in enumerate(self.finetuning_task_list):
                logits.append(self.classifier_heads[task_index](output))
            outputs = (logits,) + transformer_outputs[2:]
            return outputs

        output_mode = self.output_mode_list[task_index]
        num_labels = self.num_labels_list[task_index]
        finetuning_task = self.finetuning_task_list[task_index]
        logits = self.classifier_heads[task_index](output)

        outputs = (logits,) + transformer_outputs[1:]  # Keep new_mems and attention/hidden states if they are here

        if labels is not None:
            if output_mode == "regression":
                if finetuning_task == "topic_model":
                    loss_fct = MultiLabelSoftMarginLoss()
                    loss = loss_fct(logits, labels)
                elif self.finetuning_task == "topic_model1":
                    softmax_normalized_logits = F.softmax(logits, dim=-1)
                    loss = labels.detach().mul(torch.log(softmax_normalized_logits)).mul(-1).sum()

                else:
                    loss_fct = MSELoss()
                    if num_labels == 1:
                        loss = loss_fct(torch.sigmoid(logits).view(-1), labels.view(-1))
                    else:
                        loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


class RobertaMRCHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaMRCHead, self).__init__()
        self.config = config
        # if self.config.dense_mrc_head:
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, self.config.num_labels)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        if self.config.dense_mrc_head:
            features = self.dropout(features)
            features = self.dense(features)
            features = torch.tanh(features)
            features = self.dropout(features)

        logits = self.out_proj(features)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return (start_logits, end_logits)


class RobertaClassificationHeadWithPooler(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        self.config = config
        self.pooler = getattr(config, "pooler", "first")

        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.config.use_embedding_head:
            self.embeddingHead = nn.Linear(config.hidden_size, config.embedding_dim)
            self.out_proj = nn.Linear(config.embedding_dim, config.num_labels)
        else:
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        if self.pooler == "mask":
            self.rep_self = nn.Linear(self.config.hidden_size, 1)

    def weightsum_pooling(self, features, mask=None):
        rep_self_score = self.rep_self(features).squeeze(-1)
        rep_self_score += (1 - mask) * (-1e5)
        rep_self_score = torch.clamp(F.softmax(rep_self_score, dim=-1), min=0, max=1)
        rep = torch.einsum("bl,blh->bh", rep_self_score, features)
        return rep

    def forward(self, features, mask=None, **kwargs):
        # if self.pooler == "mean":
        #     x = torch.mean(features, dim=1)
        # else:
        #     x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        if self.pooler == "last":
            x = features[:, -1, :]
        elif self.pooler == "mean":
            x = torch.mean(features, dim=1)
        elif self.pooler == "mask":
            x = self.weightsum_pooling(features, mask)
        else:
            x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        if self.config.use_embedding_head:
            x = self.embeddingHead(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaRetrivalHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, rep1, rep2):
        x = self.dropout(rep1)
        y = self.dropout(rep2)
        logits = torch.matmul(x, y.transpose(1, 0))
        # logits = torch.tanh(logits)
        # logits = self.dropout(logits)
        return logits


class RobertaForSequenceClassificationMT(transformers.models.bert.BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    config_class = RobertaConfig
    # pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, encoder_config=None):
        super().__init__(config)
        # self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.drop_hard_label = config.drop_hard_label
        # self.visual_feature_encoder = RobertaModel(encoder_config)
        self.finetuning_task = config.finetuning_task
        self.output_mode = config.output_mode
        self.finetuning_task_list = config.finetuning_task_list
        self.output_mode_list = config.output_mode_list
        self.num_labels_list = config.num_labels_list
        self.pooler = getattr(config, "pooler", "first")
        self.inference_mt_all_tasks = config.inference_mt_all_tasks
        self.experts_per_modal = config.experts_per_modal
        # self.use_embedding_head = config.use_embedding_head

        # print(config.bert_trt)
        self.enable_trt = False
        try:
            self.return_embeddings = config.return_embeddings
        except:
            self.return_embeddings = False

        # if self.use_embedding_head:
        #     self.embedding_head = nn.Linear(config.hidden_size, config.embedding_dim)

        # This is used to add R-dropout
        self.R_dropout = config.R_dropout
        self.R_dropout_alpha = config.R_dropout_alpha
        if self.R_dropout:
            logger.info("Activate R-dropout")

        try:

            self.enable_trt = config.enable_trt
            self.hidden_size = config.hidden_size
            self.max_seq_length = config.max_seq_length
            # logger.info("enable trt")
        except:
            logger.info("No enable trt")
        # print(self.hidden_size)
        if self.enable_trt:
            self.eval_batch_size = config.eval_batch_size

            # logger.info("enable trt")
            print("BERT-TRT is enabled")
            dir_path = os.path.dirname(os.path.realpath(__file__))
            os.environ['PATH'] = os.path.join(dir_path, '../../bert_trt') + ';' + os.environ['PATH']

            self.bert_lib = ctypes.CDLL('bert_trt.dll')
            self.dlis_bert_init = self.bert_lib.DLIS_BERT_INIT
            self.dlis_bert_init.argtypes = [ctypes.c_char_p]
            self.dlis_bert_init.restype = None

            self.dlis_bert = self.bert_lib.DLIS_BERT
            self.dlis_bert.argtypes = [ndpointer(ctypes.c_int), ndpointer(ctypes.c_int), ndpointer(ctypes.c_float),
                                       ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_float), ctypes.c_int,
                                       ndpointer(ctypes.c_float), ctypes.c_int, ndpointer(ctypes.c_float), ctypes.c_int]
            self.dlis_bert.restype = None

            self.dlis_bert_init(ctypes.c_char_p(b"./bert_trt/bert_trt.ini"))

            self.output = np.zeros((self.eval_batch_size, self.max_seq_length, self.hidden_size)).astype("float32")

        self.classifier_heads = list()      # list of ModuleList For output with text + visual
        self.classifier_heads_2 = list()    #  For output only with text features
        self.classifier_heads_3 = list()    #  For output only with visual features
        self.classifier_routers = list()    #  For compine multiple experts
        self.project_layers = list()    #  For projecting the output feature (text + visual) to the new features specific for classifiers
        self.project_layers_2 = list()    #  For projecting the output feature (text) to the new features specific for classifiers
        self.project_layers_3 = list()    #  For projecting the output feature (visual) to the new features specific for classifiers


        for task_index, task_name in enumerate(self.finetuning_task_list):
            num_labels = self.num_labels_list[task_index]
            config.num_labels = num_labels
            self.num_labels = num_labels
            if "mrc" in task_name:
                self.classifier_heads.append(RobertaMRCHead(config))
            elif "mlm" in task_name:
                self.classifier_heads.append(nn.Linear(config.hidden_size, self.config.vocab_size))
            elif "topic_matching" in task_name:
                self.classifier_heads.append(RobertaRetrivalHead(config))
            elif "binary_classification" in task_name:  #  Add binary classification head for every task
                # add a layer with 0/1 probability prediction 
                # consider add dropout
                self.classifier_heads.append(nn.ModuleList([nn.Linear(config.hidden_size, 2) for _ in range(self.experts_per_modal)]))  
                self.classifier_heads_2.append(nn.ModuleList([nn.Linear(config.hidden_size, 2) for _ in range(self.experts_per_modal)]))
                self.classifier_heads_3.append(nn.ModuleList([nn.Linear(config.hidden_size, 2) for _ in range(self.experts_per_modal)]))
                self.project_layers.append(nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(self.experts_per_modal)]))  
                self.project_layers_2.append(nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(self.experts_per_modal)]))
                self.project_layers_3.append(nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(self.experts_per_modal)]))
                self.classifier_routers.append(nn.Linear(config.hidden_size*3, 3*self.experts_per_modal))   
                

            elif "multiclass_classification" in task_name:
                self.classifier_heads.append(nn.Linear(config.hidden_size, num_labels))
            elif "key_phrase" in task_name:
                # self.classifier_heads.append(nn.Linear(config.hidden_size, num_labels))
                self.kernal_size = 5
                self.padding = int((self.kernal_size - 1) / 2)
                self.classifier_heads.append(nn.Conv1d(config.hidden_size,
                                                       num_labels,
                                                       kernel_size=self.kernal_size,
                                                       stride=1,
                                                       padding=self.padding,
                                                       dilation=1,
                                                       groups=1,
                                                       bias=True,
                                                       padding_mode='zeros'))
            # elif "xml_ee_features_logits" in task_name:
            #     self.dropout = nn.Dropout(config.hidden_dropout_prob)
            #     self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            # elif "xml_ee_features" in task_name:
            #     self.dropout = nn.Dropout(config.hidden_dropout_prob)
            #     self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            # elif "xml_ee" in task_name:
            #     if task_name == "xml_ee" or task_name == "xml_ee_features":
            #         # CNN
            #         self.kernal_size = 15
            #         self.padding = int((self.kernal_size - 1) / 2)
            #         self.classifier_heads.append(nn.Conv1d(config.hidden_size,
            #                                                num_labels,
            #                                                kernel_size=self.kernal_size,
            #                                                stride=1,
            #                                                padding=self.padding,
            #                                                dilation=1,
            #                                                groups=1,
            #                                                bias=True,
            #                                                padding_mode='zeros'))
            #         self.encoder_head = nn.Conv1d(config.hidden_size,
            #                                       num_labels,
            #                                       kernel_size=self.kernal_size,
            #                                       stride=1,
            #                                       padding=self.padding,
            #                                       dilation=1,
            #                                       groups=1,
            #                                       bias=True,
            #                                       padding_mode='zeros')
            #         # Linear
            #         # self.classifier_heads.append(nn.Linear(config.hidden_size, num_labels))
            #         # self.encoder_head = nn.Linear(config.hidden_size, num_labels)
                # elif task_name == "xml_ee2":
                #     self.classifier_heads.append(nn.Linear(config.hidden_size, config.num_labels))

            else:
                self.classifier_heads.append(RobertaClassificationHeadWithPooler(config))

        self.classifier_heads = nn.ModuleList(self.classifier_heads)    # 
        self.classifier_heads_2 = nn.ModuleList(self.classifier_heads_2)
        self.classifier_heads_3 = nn.ModuleList(self.classifier_heads_3)
        self.project_layers = nn.ModuleList(self.project_layers)
        self.project_layers_2 = nn.ModuleList(self.project_layers_2)
        self.project_layers_3 = nn.ModuleList(self.project_layers_3)

        self.classifier_routers = nn.ModuleList(self.classifier_routers)

        self.temperature_ratio = config.temperature_ratio
        self.mse_loss_ratio =  config.mse_loss_ratio
        self.crossentropy_loss_ratio = config.crossentropy_loss_ratio
        self.kl_loss_ratio = config.kl_loss_ratio

    def set_label_config(self, label_feature, label_list):
        self.label_feature = label_feature
        self.label_index_mapping = dict([(key, i) for i, key in enumerate(label_list)])

    def freeze_encoder(self):
        for param in self.roberta.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.roberta.parameters():
            param.requires_grad = True

    def forward(self, input_ids=None, attention_mask=None, visual_features=None, labels=None, logits=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, task_index=0, config=None, sample_idx=0):
        teacher_logits = logits if logits is not None else None
        if self.enable_trt: #  False
            input_shape = input_ids.size()
            if attention_mask is None:
                attention_mask = torch.ones(input_shape, device=input_ids.device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

            inp_np = input_ids.cpu().numpy().astype("int32")
            segment_np = token_type_ids.cpu().numpy().astype("int32")
            mask_np = attention_mask.cpu().numpy().astype("float32")
            eval_batch_size = inp_np.shape[0]
            self.dlis_bert(inp_np, segment_np, mask_np, eval_batch_size, input_ids.numel(), self.output,
                           self.hidden_size, self.output, self.hidden_size, self.output, self.hidden_size)
            out_tensor = torch.from_numpy(self.output[:eval_batch_size]).to(input_ids.device)
            outputs = (out_tensor,)
        elif self.inference_mt_all_tasks:   #  False
            outputs = self.roberta(input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   visual_features=visual_features,
                                   head_mask=head_mask,
                                   inputs_embeds=inputs_embeds)
            sequence_output = outputs[0]
            real_outputs = []
            if self.return_embeddings:
                if self.pooler == "last":
                    pooling_output = sequence_output[:, -1, :]
                elif self.pooler == "mean":
                    pooling_output = torch.mean(sequence_output, dim=1)
                elif self.pooler == "mask":
                    pooling_output = self.weightsum_pooling(sequence_output, attention_mask)
                else:
                    pooling_output = sequence_output[:, 0, :]
                return pooling_output
            
            for one_task_index in range(len(self.output_mode_list)):
                output_mode = self.output_mode_list[one_task_index]
                num_labels = self.num_labels_list[one_task_index]
                finetuning_task = self.finetuning_task_list[one_task_index]
                if "xml_ee_binary_classification" in finetuning_task:
                    logits = self.classifier_heads[one_task_index](sequence_output)
                elif "xml_ee_multiclass_classification" in finetuning_task:
                    logits = self.classifier_heads[one_task_index](sequence_output)
                else:
                    logits = self.classifier_heads[one_task_index](sequence_output, mask=attention_mask)
                
                real_outputs.append(logits)
            return real_outputs
        else:
            # input_ids = input_ids.to(self.device)
            # attention_mask = attention_mask.to(self.device)
            # token_type_ids = token_type_ids.to(self.device)
            outputs = self.roberta(input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   visual_features=visual_features,
                                   head_mask=head_mask,
                                   inputs_embeds=inputs_embeds)

            outputs_wo_visfeat = self.roberta(input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   visual_features=None,
                                   head_mask=head_mask,
                                   inputs_embeds=inputs_embeds) # only consider text feature, set visual_features to None
            
            

            inp_emb = torch.zeros_like(input_ids.unsqueeze(-1).expand(input_ids.shape[0],\
                                                                      input_ids.shape[1], self.roberta.config.hidden_size),device=self.device, dtype=torch.float)
            outputs_wo_textfeat = self.roberta(input_ids=None,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   visual_features=visual_features,
                                   head_mask=head_mask,
                                   inputs_embeds=inp_emb) # only consider visual feature,
            

            # encoder_outputs = self.visual_feature_encoder(input_ids,
            #                                               attention_mask=attention_mask,
            #                                               token_type_ids=token_type_ids,
            #                                               visual_features=visual_features,
            #                                               position_ids=position_ids,
            #                                               head_mask=head_mask,
            #                                               inputs_embeds=inputs_embeds)


        sequence_output = outputs[0]    #  [bs, seq, hidden]
        sequence_output_wo_visfeat = outputs_wo_visfeat[0]
        sequence_output_wo_textfeat = outputs_wo_textfeat[0]

        # visual_feature_sequence_output = encoder_outputs[0]

        if self.return_embeddings:
            if self.pooler == "last":
                pooling_output = sequence_output[:, -1, :]
            elif self.pooler == "mean":
                pooling_output = torch.mean(sequence_output, dim=1)
            elif self.pooler == "mask":
                pooling_output = self.weightsum_pooling(sequence_output, attention_mask)
            else:
                pooling_output = sequence_output[:, 0, :]
            return pooling_output

            outputs = (logits,) + outputs[2:]
            return outputs

        output_mode = self.output_mode_list[task_index]
        num_labels = self.num_labels_list[task_index]
        finetuning_task = self.finetuning_task_list[task_index]

        if "topic_matching" == finetuning_task:
            batch_label_list = torch.nonzero(torch.sum(labels, dim=0)).view(-1)
            topic_embedding = self.roberta(
                input_ids=self.label_feature["input_ids"][batch_label_list, :],
                attention_mask=self.label_feature['attention_mask'][batch_label_list, :])[0]
        elif "xml_ee_binary_classification" in finetuning_task: #  Here

            representation_list = [prj(sequence_output) for prj in self.project_layers[task_index]]
            representation_wo_visfeat_list = [prj(sequence_output_wo_visfeat) for prj in self.project_layers_2[task_index]]
            representation_wo_textfeat_list = [prj(sequence_output_wo_visfeat) for prj in self.project_layers_3[task_index]]

            logits_list = [cls(sequence_output) for (cls, sequence_output) in zip(self.classifier_heads[task_index], representation_list) ]
            logits_wo_visfeat_list = [cls(sequence_output_wo_visfeat) for (cls, sequence_output_wo_visfeat) in zip(self.classifier_heads_2[task_index], representation_wo_visfeat_list)]
            logits_wo_textfeat_list = [cls(sequence_output_wo_textfeat) for (cls, sequence_output_wo_textfeat) in zip(self.classifier_heads_3[task_index], representation_wo_textfeat_list)]
            
            total_logits_softmax = [F.softmax(logits, dim=-1) for logits in logits_list] + \
                [F.softmax(logits_wo_visfeat, dim=-1) for logits_wo_visfeat in logits_wo_visfeat_list] + \
                [F.softmax(logits_wo_textfeat, dim=-1) for logits_wo_textfeat in logits_wo_textfeat_list]

            sequence_output_for_router = torch.cat([sequence_output, sequence_output_wo_visfeat, sequence_output_wo_textfeat], dim=-1)    # (bs, seq, hidden * 3)
            route_prob = F.softmax(self.classifier_routers[task_index](sequence_output_for_router), dim=-1) # (bs, seq, 3*experts_per_modal)
            combined_logits = sum([route_prob[:,:,i].unsqueeze(-1) * total_logits_softmax[i] for i in range(3 * self.experts_per_modal)])
            
            # route_prob[:,:,0].unsqueeze(-1) * logits_softmax + \
            #       route_prob[:,:,1].unsqueeze(-1) * logits_wo_visfeat_softmax + route_prob[:,:,2].unsqueeze(-1) * logits_wo_textfeat_softmax
            
            
            
            # 最终的输出logits （已经过了softmax，后续计算损失的时候不用交叉熵，用log+NLLLoss）



        elif "xml_ee_multiclass_classification" in finetuning_task:
            logits = self.classifier_heads[task_index](sequence_output)
        elif "key_phrase_mlm" in finetuning_task:
            logits = self.classifier_heads[task_index](sequence_output)
        elif "key_phrase" in finetuning_task:
            logits = self.classifier_heads[task_index](sequence_output.transpose(1, 2)).transpose(1, 2)
        elif "xml_ee_features_logits" in finetuning_task:
            logits = self.classifier(self.dropout(sequence_output))
            if self.training and self.R_dropout:
                # this will be used to calculate the kl divergence between two passes of the logits
                logits2 = self.classifier(self.dropout(sequence_output))
        elif "xml_ee_features" in finetuning_task:
            logits = self.classifier(self.dropout(sequence_output))
        elif "xml_ee2" in finetuning_task:
            logits = self.classifier_heads[task_index](sequence_output)
        elif "xml_ee" in finetuning_task:
            # CNN
            print(sequence_output, sequence_output.size())
            logits = self.classifier_heads[task_index](sequence_output.transpose(1, 2)).transpose(1, 2)
            visual_feature_logits = self.encoder_head(visual_feature_sequence_output.transpose(1, 2)).transpose(1, 2)
            # Linear
            # logits = self.classifier_heads[task_index](sequence_output)
            # visual_feature_logits = self.encoder_head(visual_feature_sequence_output)
        elif "topic_matching" in finetuning_task:
            sequence_output = sequence_output[:, 0, :]
            topic_embedding = topic_embedding[:, 0, :]
            logits = self.classifier_heads[task_index](sequence_output, topic_embedding)
        else:
            logits = self.classifier_heads[task_index](sequence_output, mask=attention_mask)

        outputs = (combined_logits,) + outputs[2:]   #  Here


        if labels is not None:
            if output_mode == "regression":
                if finetuning_task == "topic_matching":
                    loss_label = labels[:, batch_label_list]
                    loss_fct = MultiLabelSoftMarginLoss()
                    loss = loss_fct(logits, loss_label)
                elif finetuning_task == "topic_model":
                    loss_fct = MultiLabelSoftMarginLoss()
                    loss = loss_fct(logits, labels)
                elif finetuning_task == "topic_model1":
                    softmax_normalized_logits = F.softmax(logits, dim=-1)
                    loss = labels.detach().mul(torch.log(softmax_normalized_logits)).mul(-1).sum()
                elif finetuning_task == "topic_model2":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits.view(-1, num_labels), labels.view(-1, num_labels))
                elif finetuning_task == "topic_model3":
                    softmax_normalized_logits = F.softmax(logits, dim=-1)
                    loss = labels.detach().mul(torch.log(softmax_normalized_logits)).mul(-1).sum()
                else:
                    loss_fct = MSELoss()
                    if num_labels == 1:
                        loss = loss_fct(torch.sigmoid(logits).view(-1), labels.view(-1))
                    else:
                        loss = loss_fct(logits.view(-1), labels.view(-1))
            elif output_mode == "classification":
                if "mrc" in finetuning_task:
                    # for multiple label classification, like question answering
                    start_logits, end_logits = logits
                    start_positions, end_positions = labels
                    if start_positions is not None and end_positions is not None:
                        # If we are on multi-GPU, split add a dimension
                        if len(start_positions.size()) > 1:
                            start_positions = start_positions.squeeze(-1)
                        if len(end_positions.size()) > 1:
                            end_positions = end_positions.squeeze(-1)
                        # sometimes the start/end positions are outside our model inputs, we ignore these terms
                        ignored_index = start_logits.size(1)
                        start_positions.clamp_(0, ignored_index)
                        end_positions.clamp_(0, ignored_index)
                        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

                        start_loss = loss_fct(start_logits, start_positions)
                        end_loss = loss_fct(end_logits, end_positions)
                        loss = (start_loss + end_loss) / 2
                elif "mlm" in finetuning_task:
                    loss_fct = CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
                elif "key_phrase" in finetuning_task:
                    loss_fct = CrossEntropyLoss()
                    if attention_mask is not None:
                        active_loss = attention_mask.view(-1) == 1
                        active_logits = logits.contiguous().view(-1, num_labels)
                        active_labels = torch.where(
                            active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                        )
                        loss = loss_fct(active_logits, active_labels)
                    else:
                        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
                elif "xml_ee_binary_classification" in finetuning_task: 
                    if teacher_logits is not None:
                        # print("Activate knowledge distillition for binary classification")
                        # print("logits of teacher to student:", teacher_logits)
                        tempture = self.temperature_ratio
                        kl_loss = nn.KLDivLoss(size_average=True, reduce=True)
                        mse_loss = nn.MSELoss()
                        loss_mse = mse_loss(teacher_logits, logits)

                        teacher_logits_softmax = torch.softmax(teacher_logits / tempture, dim=-1)
                        student_logits = torch.log_softmax(logits / tempture, dim=-1)
                        loss_kl = kl_loss(student_logits, teacher_logits_softmax) * (tempture ** 2)

                        loss_fct = CrossEntropyLoss()
                        if attention_mask is not None:
                            active_loss = attention_mask.view(-1) == 1
                            active_logits = logits.view(-1, self.num_labels)
                            active_labels = torch.where(
                                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                            )
                            loss_softmax = loss_fct(active_logits, active_labels)
                        else:
                            loss_softmax = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                        loss = self.mse_loss_ratio * loss_mse + self.crossentropy_loss_ratio * loss_softmax + self.kl_loss_ratio * loss_kl
                    else:   
                        # loss_fct = CrossEntropyLoss()
                        loss_fct = nn.NLLLoss()
                        # Only keep active parts of the loss
                        if attention_mask is not None:  #  Here
                            active_loss = attention_mask.view(-1) == 1
                            # active_logits = torch.log(logits_softmax).view(-1, num_labels)
                            # active_logits_wo_visfeat = torch.log(logits_wo_visfeat_softmax).view(-1, num_labels)
                            # active_logits_wo_textfeat = torch.log(logits_wo_textfeat_softmax).view(-1, num_labels)

                            active_combined_logits = torch.log(combined_logits).view(-1, num_labels)

                            active_labels = torch.where(
                                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                                )
                            
                            loss_list = [loss_fct(torch.log(logits_softmax).view(-1, num_labels), active_labels) for logits_softmax in total_logits_softmax ]
                            loss_list.append(loss_fct(active_combined_logits, active_labels))


            # representation_list = [prj(sequence_output) for prj in self.project_layers[task_index]]
            # representation_wo_visfeat_list = [prj(sequence_output_wo_visfeat) for prj in self.project_layers_2[task_index]]
            # representation_wo_textfeat_list = [prj(sequence_output_wo_visfeat) for prj in self.project_layers_3[task_index]]
                                # for balance, it weight should be about  

                            # loss_with_visfeat = loss_fct(active_logits, active_labels)
                            # loss_wo_visfeat = loss_fct(active_logits_wo_visfeat, active_labels)
                            # loss_wo_textfeat = loss_fct(active_logits_wo_textfeat, active_labels)
                            # loss_combined = loss_fct(active_combined_logits, active_labels)
                            
                            # logits_list, logits_wo_visfeat_list, logits_wo_textfeat_list 三个logits各计算一个OrthogonalRegularizationLoss，然后求平均
                            
                            # orthogonal_loss will converge to about 3*78 for input (16, 32, 512)
                            # the final loss is about 2e-4, so scale the orthogonal_loss to about 2e-4
                            loss = 1.0/len(loss_list) * sum(loss_list)


                            # loss = 1.0/4 * loss_with_visfeat + 1.0/4 * loss_wo_visfeat + 1.0/4 * loss_wo_textfeat + 1.0/4 * loss_combined

                        else:
                            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
                elif "xml_ee_multiclass_classification" in finetuning_task:
                    if teacher_logits is not None:
                        # print("Activate knowledge distillition for multiclass")
                        tempture = self.temperature_ratio
                        kl_loss = nn.KLDivLoss(size_average=True, reduce=True)
                        mse_loss = nn.MSELoss()
                        loss_mse = mse_loss(teacher_logits, logits)

                        teacher_logits_softmax = torch.softmax(teacher_logits / tempture, dim=-1)
                        student_logits = torch.log_softmax(logits / tempture, dim=-1)
                        loss_kl = kl_loss(student_logits, teacher_logits_softmax) * (tempture ** 2)

                        loss_fct = CrossEntropyLoss()
                        if attention_mask is not None:
                            active_loss = attention_mask.view(-1) == 1
                            active_logits = logits.view(-1, self.num_labels)
                            active_labels = torch.where(
                                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                            )
                            loss_softmax = loss_fct(active_logits, active_labels)
                        else:
                            loss_softmax = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                        loss = self.mse_loss_ratio * loss_mse + self.crossentropy_loss_ratio * loss_softmax + self.kl_loss_ratio * loss_kl
                    else:
                        loss_fct = CrossEntropyLoss()
                        # Only keep active parts of the loss
                        if attention_mask is not None:
                            active_loss = attention_mask.view(-1) == 1
                            active_logits = logits.view(-1, num_labels)
                            active_labels = torch.where(
                                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                                )
                            loss = loss_fct(active_logits, active_labels)
                        else:
                            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
                elif "xml_ee_features_logits" in finetuning_task:
                    if teacher_logits is not None:
                        tempture = config.temperature_ratio
                        kl_loss = nn.KLDivLoss(size_average=True, reduce=True)
                        mse_loss = nn.MSELoss()
                        loss_mse = mse_loss(teacher_logits, logits)

                        teacher_logits_softmax = torch.softmax(teacher_logits / tempture, dim=-1)
                        student_logits = torch.log_softmax(logits / tempture, dim=-1)
                        loss_kl = kl_loss(student_logits, teacher_logits_softmax) * tempture * tempture

                        loss_fct = CrossEntropyLoss()
                        # Only keep active parts of the loss
                        if attention_mask is not None:
                            if (config.only_full_UHRS and sample_idx == 1) or config.only_full_UHRS is False:
                                logits_numpy = teacher_logits.detach().cpu().numpy()
                                predictions = softmax(logits_numpy, axis=2)
                                preds = np.argmax(predictions, axis=2)
                                predictions = torch.nn.functional.softmax(teacher_logits, dim=2)
                                preds = torch.argmax(predictions, dim=2)
                                multi_entity = config.incomplete_entity.split(',')
                                use_teacher_entity = config.use_teacher_entity.split(',')
                                multi_entity.extend(use_teacher_entity)
                                for index_i, _k in enumerate(preds):
                                    for entity in multi_entity:
                                        try:
                                            entity_index = config.label_list_list[0].index('B-' + entity)
                                        except:
                                            continue
                                        if entity not in use_teacher_entity:
                                            if bool((labels[index_i] == entity_index).any()) is False:
                                                labels[index_i] = torch.where((labels[index_i] != loss_fct.ignore_index) & ((preds[index_i] == entity_index) | (preds[index_i] == (entity_index + 1))), preds[index_i], labels[index_i])
                                        else:
                                            labels[index_i] = torch.where((labels[index_i] != loss_fct.ignore_index) & ((preds[index_i] == entity_index) | (preds[index_i] == (entity_index + 1)) | (labels[index_i] == entity_index) | (labels[index_i] == (entity_index + 1))), preds[index_i], labels[index_i])

                            active_loss = attention_mask.view(-1) == 1
                            active_logits = logits.view(-1, self.num_labels)
                            active_labels = torch.where(
                                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                            )
                            loss_softmax = loss_fct(active_logits, active_labels)
                        else:
                            loss_softmax = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                        loss = config.mse_loss_ratio * loss_mse + config.crossentropy_loss_ratio * loss_softmax + config.kl_loss_ratio * loss_kl
                    elif self.R_dropout and self.training:
                        # print("Start R-dropout !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        # Activate R_dropout to enrich the model robustness, only activate when model training, does not calculate during model.eval
                        loss_fct = CrossEntropyLoss()
                        def compute_kl_loss(p, q, pad_mask=None):
                            # p in dimension: (bs * seq_len) * num_labels
                            # q in dimension: (bs * seq_len) * num_labels
                            p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
                            q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
                            # pad_mask is for seq-level tasks
                            if pad_mask is not None:
                                pad_mask = pad_mask.unsqueeze(1).expand(-1, self.num_labels)
                                p_loss.masked_fill_(pad_mask, 0.)
                                q_loss.masked_fill_(pad_mask, 0.)
                                num_no_padding = pad_mask.eq(0).sum()
                            # You can choose whether to use function "sum" and "mean" depending on your task
                            # P_loss in dimension: (bs * seq_len) * num_labels
                            # q_loss in dimension: (bs * seq_len) * num_labels
                            # mean on dim=-1 which is the KL divergence equation, then sum over batch and divid by the valid position (where label != -100)
                            p_loss = p_loss.mean(dim=-1).sum() / num_no_padding
                            q_loss = q_loss.mean(dim=-1).sum()/ num_no_padding

                            loss = (p_loss + q_loss) / 2
                            return loss
                        if attention_mask is not None:
                            active_loss = attention_mask.view(-1) == 1
                            active_logits = logits.view(-1, self.num_labels)
                            active_labels = torch.where(
                                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                            )
                            loss = loss_fct(active_logits, active_labels)

                            active_logits2 = logits2.view(-1, self.num_labels)
                            loss2 = loss_fct(active_logits2, active_labels)
                            ce_loss = 0.5 * (loss + loss2)

                            kl_loss = compute_kl_loss(active_logits, active_logits2, active_labels.eq(loss_fct.ignore_index))
                            loss = ce_loss + self.R_dropout_alpha * kl_loss
                    else:
                        loss_fct = CrossEntropyLoss()
                        # Only keep active parts of the loss
                        if attention_mask is not None:
                            active_loss = attention_mask.view(-1) == 1
                            active_logits = logits.view(-1, self.num_labels)
                            active_labels = torch.where(
                                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                            )
                            loss = loss_fct(active_logits, active_labels)
                        else:
                            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif "xml_ee_features" in finetuning_task:
                    loss_fct = CrossEntropyLoss()
                    # Only keep active parts of the loss
                    if attention_mask is not None:
                        active_loss = attention_mask.view(-1) == 1
                        active_logits = logits.view(-1, self.num_labels)
                        active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels))
                        loss = loss_fct(active_logits, active_labels)
                    else:
                        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif "xml_ee" in finetuning_task:
                    # ignore no-start piece of a word, which 1 means label"X"
                    loss_fct = CrossEntropyLoss(ignore_index=1)
                    text_loss = loss_fct(logits.reshape(-1, self.num_labels), labels.view(-1))
                    visual_feature_loss = loss_fct(visual_feature_logits.reshape(-1, self.num_labels), labels.view(-1))
                    loss = text_loss + 0.4 * visual_feature_loss
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            else:
                raise NotImplementedError
            outputs = (loss,) + outputs #  Here
        return outputs  # (loss), combined_logits, (hidden_states), (attentions)     #  Here   

from transformers.models.deberta.modeling_deberta import *


class DebertaForSequenceClassificationMT(transformers.models.deberta.DebertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.finetuning_task = config.finetuning_task
        self.output_mode = config.output_mode
        self.finetuning_task_list = config.finetuning_task_list
        self.output_mode_list = config.output_mode_list
        self.num_labels_list = config.num_labels_list
        output_dim = self.pooler.output_dim

        self.classifier_heads = list()
        for task_index, task_name in enumerate(self.finetuning_task_list):
            num_labels = self.num_labels_list[task_index]
            config.num_labels = num_labels
            if "mrc" in task_name:
                self.classifier_heads.append(nn.Linear(output_dim, config.num_labels))
            elif "mlm" in task_name:
                self.classifier_heads.append(DebertaOnlyMLMHead(config))
            elif "key_phrase" in task_name:
                # self.classifier_heads.append(nn.Linear(config.hidden_size, num_labels))
                self.kernal_size = 5
                self.padding = int((self.kernal_size - 1) / 2)
                self.classifier_heads.append(nn.Conv1d(output_dim,
                                                       num_labels,
                                                       kernel_size=self.kernal_size,
                                                       stride=1,
                                                       padding=self.padding,
                                                       dilation=1,
                                                       groups=1,
                                                       bias=True,
                                                       padding_mode='zeros'))

            else:
                self.classifier_heads.append(nn.Linear(output_dim, config.num_labels))

        self.classifier_heads = nn.ModuleList(self.classifier_heads)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            task_index=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        if task_index == None or isinstance(task_index, list):
            logits = list()
            for task_index, task_name in enumerate(self.finetuning_task_list):
                if "mrc" in task_name or "mlm" in task_name:
                    logits.append(self.classifier_heads[task_index](outputs[0]))
                elif "key_phrase" in task_name:
                    logits.append(self.classifier_heads[task_index](outputs[0].transpose(1, 2)).transpose(1, 2))
                else:
                    logits.append(self.classifier_heads[task_index](pooled_output))
            outputs = (logits,) + outputs[1:]
            return outputs
        output_mode = self.output_mode_list[task_index]
        num_labels = self.num_labels_list[task_index]
        finetuning_task = self.finetuning_task_list[task_index]
        if "mrc" in finetuning_task or "mlm" in finetuning_task:
            logits = self.classifier_heads[task_index](outputs[0])
        elif "key_phrase" in finetuning_task:
            # logits = self.classifier(sequence_output.transpose(1, 2)).transpose(1, 2)
            logits = self.classifier_heads[task_index](outputs[0].transpose(1, 2)).transpose(1, 2)
        else:
            logits = self.classifier_heads[task_index](pooled_output)
        outputs = (logits,) + outputs[1:]

        # loss = None
        if labels is not None:
            if output_mode == "regression":
                if finetuning_task == "topic_model":
                    loss_fct = MultiLabelSoftMarginLoss()
                    loss = loss_fct(logits, labels)
                elif finetuning_task == "topic_model1":
                    softmax_normalized_logits = F.softmax(logits, dim=-1)
                    loss = labels.detach().mul(torch.log(softmax_normalized_logits)).mul(-1).sum()
                elif finetuning_task == "topic_model2":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits.view(-1, num_labels), labels.view(-1, num_labels))
                elif finetuning_task == "topic_model3":
                    softmax_normalized_logits = F.softmax(logits, dim=-1)
                    loss = labels.detach().mul(torch.log(softmax_normalized_logits)).mul(-1).sum()
                else:
                    loss_fct = MSELoss()
                    if num_labels == 1:
                        loss = loss_fct(torch.sigmoid(logits).view(-1), labels.view(-1))
                    else:
                        loss = loss_fct(logits.view(-1), labels.view(-1))
            elif output_mode == "classification":
                if "mrc" in finetuning_task:
                    # for multiple label classification, like question answering
                    start_logits, end_logits = logits
                    start_positions, end_positions = labels
                    if start_positions is not None and end_positions is not None:
                        # If we are on multi-GPU, split add a dimension
                        if len(start_positions.size()) > 1:
                            start_positions = start_positions.squeeze(-1)
                        if len(end_positions.size()) > 1:
                            end_positions = end_positions.squeeze(-1)
                        # sometimes the start/end positions are outside our model inputs, we ignore these terms
                        ignored_index = start_logits.size(1)
                        start_positions.clamp_(0, ignored_index)
                        end_positions.clamp_(0, ignored_index)
                        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

                        start_loss = loss_fct(start_logits, start_positions)
                        end_loss = loss_fct(end_logits, end_positions)
                        loss = (start_loss + end_loss) / 2
                elif "mlm" in finetuning_task:
                    loss_fct = CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
                elif "key_phrase" in finetuning_task:
                    loss_fct = CrossEntropyLoss()
                    if attention_mask is not None:
                        active_loss = attention_mask.view(-1) == 1
                        active_logits = logits.contiguous().view(-1, num_labels)
                        active_labels = torch.where(
                            active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                        )
                        loss = loss_fct(active_logits, active_labels)
                    else:
                        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs


class XLMRobertaForSequenceClassificationMT(RobertaForSequenceClassificationMT):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-large')
        input_ids = torch.tensor(tokenizer.encode("Schloß Nymphenburg ist sehr schön .")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    config_class = XLMRobertaConfig
    # pretrained_model_archive_map = transformers.modeling_xlm_roberta.XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


class RobertaTextCNNForSequenceClassificationMT(transformers.models.bert.BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    config_class = RobertaConfig
    # pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.enable_transformer_layer = self.config.enable_transformer_layer
        if self.enable_transformer_layer:
            self.roberta = RobertaModel(config)
        else:
            self.roberta = RobertaEmbeddingLayer(config)
        self.encoder = TextCNN(config)

        self.finetuning_task = config.finetuning_task
        self.output_mode = config.output_mode
        self.finetuning_task_list = config.finetuning_task_list
        self.output_mode_list = config.output_mode_list
        self.num_labels_list = config.num_labels_list
        # print(config.bert_trt)
        self.enable_trt = False
        try:

            self.enable_trt = config.enable_trt
            self.hidden_size = config.hidden_size
            self.max_seq_length = config.max_seq_length
            # logger.info("enable trt")

        except:
            logger.info("No enable trt")
        # print(self.hidden_size)

        self.classifier_heads = list()
        for task_index, task_name in enumerate(self.finetuning_task_list):
            num_labels = self.num_labels_list[task_index]
            config.num_labels = num_labels
            if "mrc" in task_name:
                self.classifier_heads.append(RobertaMRCHead(config))
            elif "mlm" in task_name:
                self.classifier_heads.append(nn.Linear(config.hidden_size, self.config.vocab_size))
            elif "key_phrase" in task_name:
                self.classifier_heads.append(nn.Linear(config.hidden_size, num_labels))
            else:
                self.classifier_heads.append(MyRobertaClassificationHead(config))
        self.classifier_heads = nn.ModuleList(self.classifier_heads)

    def freeze_encoder(self):
        if self.enable_transformer_layer:
            for param in self.roberta.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self):
        if self.enable_transformer_layer:
            for param in self.roberta.parameters():
                param.requires_grad = True

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None,
                labels=None, task_index=None):

        embedding_output = self.roberta(input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        head_mask=head_mask,
                                        inputs_embeds=inputs_embeds, )
        if self.enable_transformer_layer:
            embedding_output = embedding_output[0]

        sequence_output = self.encoder(embedding_output)

        if task_index == None or isinstance(task_index, list):
            logits = list()
            for task_index, task_name in enumerate(self.finetuning_task_list):
                logits.append(self.classifier_heads[task_index](sequence_output))
            outputs = (logits,)  # + outputs[2:]
            return outputs

        output_mode = self.output_mode_list[task_index]
        num_labels = self.num_labels_list[task_index]
        finetuning_task = self.finetuning_task_list[task_index]

        logits = self.classifier_heads[task_index](sequence_output)

        outputs = (logits,)  # + outputs[2:]
        if labels is not None:
            if output_mode == "regression":
                if finetuning_task == "topic_model":
                    loss_fct = MultiLabelSoftMarginLoss()
                    loss = loss_fct(logits, labels)
                elif finetuning_task == "topic_model1":
                    softmax_normalized_logits = F.softmax(logits, dim=-1)
                    loss = labels.detach().mul(torch.log(softmax_normalized_logits)).mul(-1).sum()
                elif finetuning_task == "topic_model2":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits.view(-1, num_labels), labels.view(-1, num_labels))
                elif finetuning_task == "topic_model3":
                    softmax_normalized_logits = F.softmax(logits, dim=-1)
                    loss = labels.detach().mul(torch.log(softmax_normalized_logits)).mul(-1).sum()
                else:
                    loss_fct = MSELoss()
                    if num_labels == 1:
                        loss = loss_fct(torch.sigmoid(logits).view(-1), labels.view(-1))
                    else:
                        loss = loss_fct(logits.view(-1), labels.view(-1))
            elif output_mode == "classification":
                if "mrc" in finetuning_task:
                    # for multiple label classification, like question answering
                    start_logits, end_logits = logits
                    start_positions, end_positions = labels
                    if start_positions is not None and end_positions is not None:
                        # If we are on multi-GPU, split add a dimension
                        if len(start_positions.size()) > 1:
                            start_positions = start_positions.squeeze(-1)
                        if len(end_positions.size()) > 1:
                            end_positions = end_positions.squeeze(-1)
                        # sometimes the start/end positions are outside our model inputs, we ignore these terms
                        ignored_index = start_logits.size(1)
                        start_positions.clamp_(0, ignored_index)
                        end_positions.clamp_(0, ignored_index)
                        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

                        start_loss = loss_fct(start_logits, start_positions)
                        end_loss = loss_fct(end_logits, end_positions)
                        loss = (start_loss + end_loss) / 2
                elif "key_phrase" in finetuning_task:
                    loss_fct = CrossEntropyLoss()
                    if attention_mask is not None:
                        active_loss = attention_mask.view(-1) == 1
                        active_logits = logits.view(-1, self.num_labels)
                        active_labels = torch.where(
                            active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                        )
                        loss = loss_fct(active_logits, active_labels)
                    else:
                        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif "mlm" in finetuning_task:
                    loss_fct = CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            else:
                raise NotImplementedError
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class RobertaLSTMForSequenceClassificationMT(RobertaTextCNNForSequenceClassificationMT):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BiLSTM(config)


class RobertaDPCNNForSequenceClassificationMT(RobertaTextCNNForSequenceClassificationMT):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = DPTextCNN(config)


class XLMRobertaTextCNNForSequenceClassificationMT(RobertaTextCNNForSequenceClassificationMT):
    config_class = XLMRobertaConfig
    # pretrained_model_archive_map = transformers.modeling_xlm_roberta.XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


class XLMRobertaLSTMForSequenceClassificationMT(RobertaLSTMForSequenceClassificationMT):
    config_class = XLMRobertaConfig
    # pretrained_model_archive_map = transformers.modeling_xlm_roberta.XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


class XLMRobertaDPCNNForSequenceClassificationMT(RobertaDPCNNForSequenceClassificationMT):
    config_class = XLMRobertaConfig
    # pretrained_model_archive_map = transformers.modeling_xlm_roberta.XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


################################################## END Multi-task #############################################################


################################################## BEGIN traditional models #############################################################
from transformers.models.bert.modeling_bert import BertEmbeddings


class BertEmbeddingLayer(transformers.models.bert.BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        return embedding_output


class BertTextCNNForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.enable_transformer_layer = self.config.enable_transformer_layer
        if self.enable_transformer_layer:
            self.bert = BertModel(config)
        else:
            self.bert = BertEmbeddingLayer(config)

        self.encoder = TextCNN(config)

        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        embedding_output = self.bert(input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     inputs_embeds=inputs_embeds, )

        if self.enable_transformer_layer:
            embedding_output = embedding_output[1]

        pooled_output = self.encoder(embedding_output)

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        outputs = (logits,)  # + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.output_mode == "regression":
                if self.finetuning_task == "topic_model":
                    loss_fct = MultiLabelSoftMarginLoss()
                    loss = loss_fct(logits, labels)
                elif self.finetuning_task == "topic_model1":
                    softmax_normalized_logits = F.softmax(logits, dim=-1)
                    loss = labels.detach().mul(torch.log(softmax_normalized_logits)).mul(-1).sum()

                else:
                    loss_fct = MSELoss()
                    if self.num_labels == 1:
                        loss = loss_fct(torch.sigmoid(logits).view(-1), labels.view(-1))
                    else:
                        loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertLSTMForSequenceClassification(BertTextCNNForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BiLSTM(config)


from transformers.models.roberta.modeling_roberta import RobertaEmbeddings


class RobertaEmbeddingLayer(BertEmbeddingLayer):
    config_class = RobertaConfig
    # pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


class MyRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaTextCNNForSequenceClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    # pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.finetuning_task = config.finetuning_task
        self.output_mode = config.output_mode
        self.enable_transformer_layer = self.config.enable_transformer_layer
        if self.enable_transformer_layer:
            self.roberta = RobertaModel(config)
        else:
            self.roberta = RobertaEmbeddingLayer(config)
        self.encoder = TextCNN(config)
        self.classifier = MyRobertaClassificationHead(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):

        embedding_output = self.roberta(input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        head_mask=head_mask,
                                        inputs_embeds=inputs_embeds, )
        if self.enable_transformer_layer:
            embedding_output = embedding_output[0]

        sequence_output = self.encoder(embedding_output)
        # print("sequence_output")
        # print(sequence_output.size())
        logits = self.classifier(sequence_output)
        outputs = (logits,)  # + outputs[2:]
        if labels is not None:
            if self.output_mode == "regression":
                if self.finetuning_task == "topic_model":
                    loss_fct = MultiLabelSoftMarginLoss()
                    loss = loss_fct(logits, labels)
                elif self.finetuning_task == "topic_model1":
                    softmax_normalized_logits = F.softmax(logits, dim=-1)
                    loss = labels.detach().mul(torch.log(softmax_normalized_logits)).mul(-1).sum()

                else:
                    loss_fct = MSELoss()
                    if self.num_labels == 1:
                        loss = loss_fct(torch.sigmoid(logits).view(-1), labels.view(-1))
                    else:
                        loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


from transformers.models.xlm.modeling_xlm import create_sinusoidal_embeddings, get_masks


class XLMEmbeddingLayer(transformers.models.xlm.XLMPreTrainedModel):

    def __init__(self, config):
        super(XLMEmbeddingLayer, self).__init__(config)
        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        # encoder / decoder, output layer
        self.is_encoder = config.is_encoder
        self.is_decoder = not config.is_encoder
        if self.is_decoder:
            raise NotImplementedError("Currently XLM can only be used as an encoder")
        # self.with_output = with_output
        self.causal = config.causal

        # dictionary / languages
        self.n_langs = config.n_langs
        self.use_lang_emb = config.use_lang_emb
        self.n_words = config.n_words
        self.eos_index = config.eos_index
        self.pad_index = config.pad_index
        # self.dico = dico
        # self.id2lang = config.id2lang
        # self.lang2id = config.lang2id
        # assert len(self.dico) == self.n_words
        # assert len(self.id2lang) == len(self.lang2id) == self.n_langs

        # model parameters
        self.dim = config.emb_dim  # 512 by default
        self.hidden_dim = self.dim * 4  # 2048 by default
        self.n_heads = config.n_heads  # 8 by default
        self.n_layers = config.n_layers
        self.dropout = config.dropout
        self.attention_dropout = config.attention_dropout
        assert self.dim % self.n_heads == 0, 'transformer dim must be a multiple of n_heads'

        # embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.dim)
        if config.sinusoidal_embeddings:
            create_sinusoidal_embeddings(config.max_position_embeddings, self.dim, out=self.position_embeddings.weight)
        if config.n_langs > 1 and config.use_lang_emb:
            self.lang_embeddings = nn.Embedding(self.n_langs, self.dim)
        self.embeddings = nn.Embedding(self.n_words, self.dim, padding_idx=self.pad_index)
        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=config.layer_norm_eps)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def forward(self, input_ids=None, attention_mask=None, langs=None, token_type_ids=None, position_ids=None,
                lengths=None, cache=None, head_mask=None, inputs_embeds=None):  # removed: src_enc=None, src_len=None
        if input_ids is not None:
            bs, slen = input_ids.size()
        else:
            bs, slen = inputs_embeds.size()[:-1]

        if lengths is None:
            if input_ids is not None:
                lengths = (input_ids != self.pad_index).sum(dim=1).long()
            else:
                lengths = torch.LongTensor([slen] * bs)
        # mask = input_ids != self.pad_index

        # check inputs
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # position_ids
        if position_ids is None:
            position_ids = torch.arange(slen, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand((bs, slen))
        else:
            assert position_ids.size() == (bs, slen)  # (slen, bs)
            # position_ids = position_ids.transpose(0, 1)

        # langs
        if langs is not None:
            assert langs.size() == (bs, slen)  # (slen, bs)
            # langs = langs.transpose(0, 1)

        # embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        tensor = inputs_embeds + self.position_embeddings(position_ids).expand_as(inputs_embeds)
        if langs is not None and self.use_lang_emb:
            tensor = tensor + self.lang_embeddings(langs)
        if token_type_ids is not None:
            tensor = tensor + self.embeddings(token_type_ids)
        tensor = self.layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)

        return tensor


class XLMTextCNNForSequenceClassification(transformers.models.xlm.XLMPreTrainedModel):
    def __init__(self, config):
        super(XLMTextCNNForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.finetuning_task = config.finetuning_task
        self.output_mode = config.output_mode
        self.enable_transformer_layer = self.config.enable_transformer_layer

        if self.enable_transformer_layer:
            self.transformer = XLMModel(config)
        else:
            self.transformer = XLMEmbeddingLayer(config)
        self.encoder = TextCNN(config)
        # self.classifier = MyRobertaClassificationHead(config)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, langs=None, token_type_ids=None, position_ids=None,
                lengths=None, cache=None, head_mask=None, inputs_embeds=None, labels=None):
        transformer_outputs = self.transformer(input_ids,
                                               attention_mask=attention_mask,
                                               langs=langs,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               lengths=lengths,
                                               cache=cache,
                                               head_mask=head_mask,
                                               inputs_embeds=inputs_embeds)

        output = transformer_outputs[0]
        logits = self.classifier(output)
        outputs = (logits,)  # + outputs[2:]
        if labels is not None:
            if self.output_mode == "regression":
                if self.finetuning_task == "topic_model":
                    loss_fct = MultiLabelSoftMarginLoss()
                    loss = loss_fct(logits, labels)
                elif self.finetuning_task == "topic_model1":
                    softmax_normalized_logits = F.softmax(logits, dim=-1)
                    loss = labels.detach().mul(torch.log(softmax_normalized_logits)).mul(-1).sum()

                else:
                    loss_fct = MSELoss()
                    if self.num_labels == 1:
                        loss = loss_fct(torch.sigmoid(logits).view(-1), labels.view(-1))
                    else:
                        loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class XLMLSTMForSequenceClassification(XLMTextCNNForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BiLSTM(config)


class XLMDPCNNForSequenceClassification(XLMTextCNNForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = DPTextCNN(config)


class RobertaLSTMForSequenceClassification(RobertaTextCNNForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BiLSTM(config)


class RobertaDPCNNForSequenceClassification(RobertaTextCNNForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = DPTextCNN(config)


class XLMRobertaTextCNNForSequenceClassification(RobertaTextCNNForSequenceClassification):
    config_class = XLMRobertaConfig
    # pretrained_model_archive_map = transformers.modeling_xlm_roberta.XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


class XLMRobertaLSTMForSequenceClassification(RobertaLSTMForSequenceClassification):
    config_class = XLMRobertaConfig
    # pretrained_model_archive_map = transformers.modeling_xlm_roberta.XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


class XLMRobertaDPCNNForSequenceClassification(RobertaDPCNNForSequenceClassification):
    config_class = XLMRobertaConfig
    # pretrained_model_archive_map = transformers.modeling_xlm_roberta.XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


################################################## END traditional models #############################################################


################################################## BEGIN Pair wise models #############################################################

class BertPairWiseClassifcationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output = nn.Linear(self.config.hidden_size, self.config.num_labels)

    def forward(self, feature_a, feature_b, **kwargs):
        x = feature_a - feature_b
        x = self.output(x)
        return x


class BertForPairWiseRanking(transformers.models.bert.BertForSequenceClassification):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config):
        super().__init__(config)
        self.finetuning_task = config.finetuning_task
        self.output_mode = config.output_mode
        self.classifier = BertPairWiseClassifcationHead(config)

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                input_ids_b=None, attention_mask_b=None,
                token_type_ids_b=None, position_ids_b=None, head_mask_b=None,
                inputs_embeds=None, labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        outputs_b = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids_b,
            position_ids=position_ids_b,
            head_mask=head_mask_b
        )

        pooled_output_a = outputs[1]
        pooled_output_a = self.dropout(pooled_output_a)

        pooled_output_b = outputs_b[1]
        pooled_output_b = self.dropout(pooled_output_b)

        logits = self.classifier(pooled_output_a, pooled_output_b)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class RobertaPairWiseClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features_a, features_b, **kwargs):
        x_a = features_a[:, 0, :]  # take <s> token (equiv. to [CLS])
        x_b = features_b[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x_a - x_b
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForPairWiseRanking(transformers.models.roberta.RobertaForSequenceClassification):
    # class RobertaForSequenceClassification(transformers.RobertaForSequenceClassification):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    config_class = RobertaConfig
    # pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.finetuning_task = config.finetuning_task
        self.output_mode = config.output_mode

        # print(config.bert_trt)
        self.enable_trt = False
        try:

            self.enable_trt = config.enable_trt
            self.hidden_size = config.hidden_size
            self.max_seq_length = config.max_seq_length
            # logger.info("enable trt")

        except:
            logger.info("No enable trt")
        # print(self.hidden_size)
        if self.enable_trt:
            self.eval_batch_size = config.eval_batch_size

            # logger.info("enable trt")
            print("BERT-TRT is enabled")
            dir_path = os.path.dirname(os.path.realpath(__file__))
            os.environ['PATH'] = os.path.join(dir_path, '../../bert_trt') + ';' + os.environ['PATH']
            self.bert_lib = ctypes.CDLL('bert_trt.dll')

            self.dlis_bert_init = self.bert_lib.DLIS_BERT_INIT
            self.dlis_bert_init.argtypes = [ctypes.c_char_p]
            self.dlis_bert_init.restype = None

            self.dlis_bert = self.bert_lib.DLIS_BERT
            self.dlis_bert = self.bert_lib.DLIS_BERT
            self.dlis_bert.argtypes = [ndpointer(ctypes.c_int), ndpointer(ctypes.c_int), ndpointer(ctypes.c_float),
                                       ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_float), ctypes.c_int,
                                       ndpointer(ctypes.c_float), ctypes.c_int, ndpointer(ctypes.c_float),
                                       ctypes.c_int]
            self.dlis_bert.restype = None

            self.dlis_bert_init(ctypes.c_char_p(b"./bert_trt/bert_trt.ini"))

            self.output = np.zeros((self.eval_batch_size, self.max_seq_length, self.hidden_size)).astype("float32")
            self.output_b = np.zeros((self.eval_batch_size, self.max_seq_length, self.hidden_size)).astype("float32")
        self.classifier = RobertaPairWiseClassificationHead(config)

    def forward(self,
                input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                input_ids_b=None, attention_mask_b=None, token_type_ids_b=None, position_ids_b=None, head_mask_b=None,
                inputs_embeds=None, labels=None):

        if self.enable_trt:
            input_shape = input_ids.size()
            if attention_mask is None:
                attention_mask = torch.ones(input_shape, device=input_ids.device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

            inp_np = input_ids.cpu().numpy().astype("int32")
            segment_np = token_type_ids.cpu().numpy().astype("int32")
            mask_np = attention_mask.cpu().numpy().astype("float32")
            eval_batch_size = inp_np.shape[0]
            self.dlis_bert(inp_np, segment_np, mask_np, eval_batch_size, input_ids.numel(), self.output,
                           self.hidden_size, self.output, self.hidden_size, self.output, self.hidden_size)
            out_tensor = torch.from_numpy(self.output).to(input_ids.device)
            outputs = (out_tensor,)

            input_shape = input_ids.size()
            if attention_mask_b is None:
                attention_mask_b = torch.ones(input_shape, device=input_ids.device)
            if token_type_ids_b is None:
                token_type_ids_b = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

            inp_np_b = input_ids_b.cpu().numpy().astype("int32")
            segment_np_b = token_type_ids_b.cpu().numpy().astype("int32")
            mask_np_b = attention_mask_b.cpu().numpy().astype("float32")
            self.dlis_bert(inp_np_b, segment_np_b, mask_np_b, 1, input_ids_b.numel(), self.output_b, self.hidden_size,
                           self.output_b, self.hidden_size, self.output_b, self.hidden_size)
            out_tensor_b = torch.from_numpy(self.output_b).to(input_ids.device)
            outputs_b = (out_tensor_b,)


        else:
            outputs = self.roberta(input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask)
            outputs_b = self.roberta(input_ids_b,
                                     attention_mask=attention_mask_b,
                                     token_type_ids=token_type_ids_b,
                                     position_ids=position_ids_b,
                                     head_mask=head_mask_b)
        sequence_output_a = outputs[0]
        sequence_output_b = outputs_b[0]
        logits = self.classifier(sequence_output_a, sequence_output_b)

        outputs = (logits,) + outputs[2:] + outputs_b[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class XLMRobertaForPairWiseRanking(RobertaForPairWiseRanking):
    config_class = XLMRobertaConfig
    # pretrained_model_archive_map = transformers.modeling_xlm_roberta.XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


################################################## END Pair wise models #############################################


################################################## BEGIN Generation #################################################
def _reorder_buffer(attn_cache, new_order):
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None:
            attn_cache[k] = input_buffer_k.index_select(0, new_order)
    return attn_cache


def _make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


class BartForConditionalGeneration(transformers.models.bart.modeling_bart.BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        base_model = BartModel(config)
        self.model = base_model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        r"""
        masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
            with labels
            in ``[0, ..., config.vocab_size]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

            # Mask filling only works for bart-large
            from transformers import BartTokenizer, BartForConditionalGeneration
            tokenizer = BartTokenizer.from_pretrained('bart-large')
            TXT = "My friends are <mask> but they eat too many carbs."
            model = BartForConditionalGeneration.from_pretrained('bart-large')
            input_ids = tokenizer.batch_encode_plus([TXT], return_tensors='pt')['input_ids']
            logits = model(input_ids)[0]
            masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
            probs = logits[0, masked_index].softmax(dim=0)
            values, predictions = probs.topk(5)
            tokenizer.decode(predictions).split()
            # ['good', 'great', 'all', 'really', 'very']
        """
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            # decoder_cached_states=decoder_cached_states,
            # generation_mode=generation_mode,
        )
        lm_logits = F.linear(outputs[0], self.model.shared.weight)
        outputs = (lm_logits,) + outputs[1:]  # Add hidden states and attention if they are here
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # TODO(SS): do we need to ignore pad tokens in lm_labels?
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs

    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        # first step, decoder_cached_states are empty
        if not past[1]:
            encoder_outputs, decoder_cached_states = past, None
        else:
            encoder_outputs, decoder_cached_states = past
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "decoder_cached_states": decoder_cached_states,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "generation_mode": True,
        }

    def prepare_scores_for_generation(self, scores, cur_len, max_length):
        if cur_len == 1:
            self._force_token_ids_generation(scores, self.config.bos_token_id)
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(scores, self.config.eos_token_id)
        return scores

    @staticmethod
    def _reorder_cache(past, beam_idx):
        ((enc_out, enc_mask), decoder_cached_states) = past
        reordered_past = []
        for layer_past in decoder_cached_states:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)

        new_enc_out = enc_out if enc_out is None else enc_out.index_select(0, beam_idx)
        new_enc_mask = enc_mask if enc_mask is None else enc_mask.index_select(0, beam_idx)

        past = ((new_enc_out, new_enc_mask), reordered_past)
        return past

    def get_encoder(self):
        return self.model.encoder

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.model.shared)  # make it on the fly


################################################## END Generation #################################################


################################################## BEGIN Key Phrase #################################################

class RobertaForKeyPhrase(transformers.models.roberta.RobertaForTokenClassification):
    config_class = RobertaConfig
    # pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.finetuning_task = config.finetuning_task
        self.output_mode = config.output_mode
        self.num_labels = config.num_labels

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.enable_trt = False

        try:

            self.enable_trt = config.enable_trt
            self.hidden_size = config.hidden_size
            self.max_seq_length = config.max_seq_length
            # logger.info("enable trt")
        except:
            logger.info("No enable trt")
        # print(self.hidden_size)
        if self.enable_trt:
            self.eval_batch_size = config.eval_batch_size

            # logger.info("enable trt")
            print("BERT-TRT is enabled")
            dir_path = os.path.dirname(os.path.realpath(__file__))
            os.environ['PATH'] = os.path.join(dir_path, '../../bert_trt') + ';' + os.environ['PATH']
            self.bert_lib = ctypes.CDLL('bert_trt.dll')

            self.dlis_bert_init = self.bert_lib.DLIS_BERT_INIT
            self.dlis_bert_init.argtypes = [ctypes.c_char_p]
            self.dlis_bert_init.restype = None

            self.dlis_bert = self.bert_lib.DLIS_BERT
            self.dlis_bert = self.bert_lib.DLIS_BERT
            self.dlis_bert.argtypes = [ndpointer(ctypes.c_int), ndpointer(ctypes.c_int), ndpointer(ctypes.c_float),
                                       ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_float), ctypes.c_int,
                                       ndpointer(ctypes.c_float), ctypes.c_int, ndpointer(ctypes.c_float), ctypes.c_int]
            self.dlis_bert.restype = None

            self.dlis_bert_init(ctypes.c_char_p(b"./bert_trt/bert_trt.ini"))

            self.output = np.zeros((self.eval_batch_size, self.max_seq_length, self.hidden_size)).astype("float32")

        self.init_weights()

    def freeze_encoder(self):
        for param in self.roberta.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.roberta.parameters():
            param.requires_grad = True

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        if self.enable_trt:
            input_shape = input_ids.size()
            if attention_mask is None:
                attention_mask = torch.ones(input_shape, device=input_ids.device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

            inp_np = input_ids.cpu().numpy().astype("int32")
            segment_np = token_type_ids.cpu().numpy().astype("int32")
            mask_np = attention_mask.cpu().numpy().astype("float32")
            eval_batch_size = inp_np.shape[0]
            self.dlis_bert(inp_np, segment_np, mask_np, eval_batch_size, input_ids.numel(), self.output,
                           self.hidden_size, self.output, self.hidden_size, self.output, self.hidden_size)
            out_tensor = torch.from_numpy(self.output).to(input_ids.device)
            outputs = (out_tensor,)
        else:
            outputs = self.roberta(input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


class XLMRobertaForKeyPhrase(RobertaForKeyPhrase):
    config_class = XLMRobertaConfig
    # pretrained_model_archive_map = transformers.models.xlm_roberta.modeling_xlm_roberta.XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST


################################################## END Key Phrase #################################################


################################################## Begin Topic Matching ###########################################
class RobertaForSimMatching(transformers.models.roberta.RobertaForSequenceClassification):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.embedding_dim = config.embedding_dim
        self.temperature = config.temperature
        self.distance = config.distance
        self.embeddingHead = nn.Linear(config.hidden_size, self.embedding_dim)
        self.norm = nn.LayerNorm(self.embedding_dim)
        self.init_weights()

    def get_cls_embeddings(self,
                           input_ids=None,
                           attention_mask=None,
                           token_type_ids=None,
                           position_ids=None,
                           head_mask=None,
                           ):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        sequence_output = outputs[0]
        x = sequence_output[:, 0, :]  # take <s> token (equiv. to [CLS])

        x = self.norm(self.embeddingHead(x))

        return x

    def forward(self, a, b=None, c=None):
        q_embs = self.get_cls_embeddings(**a)

        outputs = (q_embs,)
        if b != None:
            p_embs = self.get_cls_embeddings(**b)
            if self.distance == "cosine":
                d1 = torch.cosine_similarity(q_embs, p_embs).view(q_embs.shape[0], 1)
            else:
                d1 = (q_embs * p_embs).sum(-1).unsqueeze(1)
            outputs = (d1,)
        if c != None:
            g_embs = self.get_cls_embeddings(**c)
            if self.distance == "cosine":
                d2 = torch.cosine_similarity(q_embs, g_embs).view(q_embs.shape[0], 1)
            else:
                d2 = (q_embs * g_embs).sum(-1).unsqueeze(1)

            logit_matrix = torch.cat([d1,
                                      d2], dim=1)  # [B, 2]
            logit_matrix = logit_matrix / self.temperature
            lsm = F.log_softmax(logit_matrix, dim=1)
            loss = -1.0 * lsm[:, 0]
            outputs = (loss.mean(),)

        return outputs


class XLMRobertaForSimMatching(RobertaForSimMatching):
    config_class = XLMRobertaConfig
