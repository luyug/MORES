from torch import nn
from transformers.models.bert.modeling_bert import  apply_chunking_to_forward, BertLayer


class MORES_BertLayer(BertLayer):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        cross_attention_outputs = self.crossattention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        outputs = cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        self_attention_outputs = self.attention(
            attention_output,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = outputs + self_attention_outputs[1:]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        return outputs