"""
MORES on BERT base.
"""
import os
import warnings

import torch
import torch.functional as F
import copy
from transformers import BertModel, BertConfig, AutoModel
from transformers.models.bert.modeling_bert import BertPooler, BertPreTrainingHeads
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from torch import nn

from bert_mores import MORES_BertLayer
from arguments import ModelArguments, DataArguments, \
    MORESTrainingArguments as TrainingArguments
import logging

logger = logging.getLogger(__name__)


class MORESSym(nn.Module):
    def __init__(self, bert: BertModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments):
        super().__init__()
        self.bert = bert
        config_m: BertConfig = copy.deepcopy(bert.config)
        config_m.is_decoder = True
        config_m.add_cross_attention = True
        self.interaction_module = nn.ModuleList(
            [MORES_BertLayer(config_m) for _ in range(model_args.n_ibs)]
        )
        self.interaction_module.apply(bert._init_weights)

        if model_args.use_pooler:
            self.pooler = BertPooler(config_m)

        if model_args.copy_weight_to_ib:
            for i in range(model_args.n_ibs):
                self.interaction_module[i].attention =\
                    copy.deepcopy(self.bert.encoder.layer[-1].attention)
                self.interaction_module[i].crossattention = \
                    copy.deepcopy(self.bert.encoder.layer[-1].attention)
                self.interaction_module[i].intermediate = \
                    copy.deepcopy(self.bert.encoder.layer[-1].intermediate)
                self.interaction_module[i].output = \
                    copy.deepcopy(self.bert.encoder.layer[-1].output)

        self.proj = nn.Linear(config_m.hidden_size, 2)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

    def forward(self, qry, doc, labels=None):
        qry_out: BaseModelOutputWithPoolingAndCrossAttentions = self._encode_query(qry)
        doc_out: BaseModelOutputWithPoolingAndCrossAttentions = self._encode_document(doc)

        self_mask = self.bert.get_extended_attention_mask(
            qry['attention_mask'], qry['attention_mask'].shape, qry['attention_mask'].device)
        cross_mask = self.bert.get_extended_attention_mask(
            doc['attention_mask'], doc['attention_mask'].shape, doc['attention_mask'].device)

        hidden_states = qry_out.last_hidden_state
        interaction_self_attention = ()
        interaction_cross_attention = ()
        for i, ib_layer in enumerate(self.interaction_module):
            layer_outputs = ib_layer(
                hidden_states,
                attention_mask=self_mask,
                encoder_hidden_states=doc_out.last_hidden_state,
                encoder_attention_mask=cross_mask,
                output_attentions=self.bert.config.output_attentions,
            )
            hidden_states = layer_outputs[0]
            if self.bert.config.output_attentions:
                interaction_self_attention = interaction_self_attention + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    interaction_cross_attention = interaction_cross_attention + (layer_outputs[2],)

        if self.model_args.use_pooler:
            cls_reps = self.pooler(hidden_states)  # use bert pooler to pool the hiddens
        else:
            cls_reps = hidden_states[:, 0]
        scores = self.proj(cls_reps)

        if self.training:
            loss = self.cross_entropy(scores, labels)
        else:
            loss = None

        all_attentions = (
            doc_out.attentions,
            qry_out.attentions,
            interaction_self_attention,
            interaction_cross_attention
        ) if self.bert.config.output_attentions else None
        return SequenceClassifierOutput(
            loss=loss,
            logits=scores,
            attentions=all_attentions,
        )

    def _encode_document(self, doc):
        return self.bert(**doc, return_dict=True)

    def _encode_query(self, qry):
        return self.bert(**qry, return_dict=True)

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModel.from_pretrained(*args, **kwargs)
        mores = cls(hf_model, model_args, data_args, train_args)
        path = args[0]
        if os.path.exists(os.path.join(path, 'model.pt')):
            logger.info('loading extra weights from local files')
            model_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
            load_result = mores.load_state_dict(model_dict, strict=False)
            print(load_result, flush=True)
        return mores

    def save_pretrained(self, output_dir: str):
        self.bert.save_pretrained(output_dir)
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith('bert')]
        warnings.warn(f'omiting {len(hf_weight_keys)} transformer weights')
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        torch.save([self.data_args, self.model_args, self.train_args], os.path.join(output_dir, 'args.pt'))


class MORES(MORESSym):
    def __init__(self, bert: BertModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments):
        super().__init__(bert, model_args, data_args, train_args)
        self.bert = bert
        self.q_bert = copy.deepcopy(bert)
        config_m: BertConfig = copy.deepcopy(bert.config)
        config_m.is_decoder = True
        config_m.add_cross_attention = True
        self.interaction_module = nn.ModuleList(
            [MORES_BertLayer(config_m) for _ in range(model_args.n_ibs)]
        )
        self.interaction_module.apply(bert._init_weights)

        if model_args.copy_weight_to_ib:
            for i in range(model_args.n_ibs):
                self.interaction_module[i].attention =\
                    copy.deepcopy(self.bert.encoder.layer[12 - model_args.n_ibs + i].attention)
                self.interaction_module[i].crossattention = \
                    copy.deepcopy(self.bert.encoder.layer[12 - model_args.n_ibs + i].attention)
                self.interaction_module[i].intermediate = \
                    copy.deepcopy(self.bert.encoder.layer[12 - model_args.n_ibs + i].intermediate)
                self.interaction_module[i].output = \
                    copy.deepcopy(self.bert.encoder.layer[12 - model_args.n_ibs + i].output)

        self.q_bert.encoder.layer = nn.ModuleList(
            [self.bert.encoder.layer[i] for i in range(12 - model_args.n_ibs)])

        self.proj = nn.Linear(config_m.hidden_size, 2)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

    def _encode_query(self, qry):
        return self.q_bert(**qry, return_dict=True)

    def _encode_document(self, doc):
        return self.bert(**doc, return_dict=True)


