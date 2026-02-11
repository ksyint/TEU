import math
import torch
from torch import nn
from transformers.models.bert.configuration_bert import BertConfig

try:
    from transformers.modeling_utils import (
        PreTrainedModel,
        apply_chunking_to_forward,
        find_pruneable_heads_and_indices,
        prune_linear_layer,
    )
except ImportError:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.pytorch_utils import apply_chunking_to_forward
    from transformers.pytorch_utils import find_pruneable_heads_and_indices
    from transformers.pytorch_utils import prune_linear_layer

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
)


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.config = config

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            self.key = nn.Linear(config.encoder_width, self.all_head_size)
            self.value = nn.Linear(config.encoder_width, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        past_key_value = (key_layer, value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs_dropped = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask
        context_layer = torch.matmul(attention_probs_dropped, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.self = BertSelfAttention(config, is_cross_attention)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, output_attentions=False):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask,
            encoder_hidden_states, encoder_attention_mask,
            past_key_value, output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config, layer_num):
        super().__init__()
        self.config = config
        self.attention = BertAttention(config)
        if layer_num % config.cross_attention_freq == 0:
            self.crossattention = BertAttention(config, is_cross_attention=True)
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, output_attentions=False, mode=None):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]

        if mode == "multimodal" and self.has_cross_attention:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask,
                encoder_hidden_states, encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        outputs = outputs + (present_key_value,)
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [BertLayer(config, i) for i in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_values=None, use_cache=None, output_attentions=False,
                output_hidden_states=False, return_dict=True, mode="multimodal"):
        all_hidden_states = () if output_hidden_states else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            layer_outputs = layer_module(
                hidden_states, attention_mask, layer_head_mask,
                encoder_hidden_states, encoder_attention_mask,
                past_key_value, output_attentions, mode=mode,
            )
            hidden_states = layer_outputs[0]
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(PreTrainedModel):
    config_class = BertConfig

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

    def get_extended_attention_mask(self, attention_mask, input_shape, device, is_decoder=False):
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None, past_key_values=None, use_cache=None,
                output_attentions=False, output_hidden_states=False, return_dict=True,
                is_decoder=False, mode="multimodal"):
        if input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device, is_decoder)

        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, list):
                encoder_batch_size, encoder_sequence_length = encoder_hidden_states[0].size()[:2]
            else:
                encoder_batch_size, encoder_sequence_length = encoder_hidden_states.size()[:2]
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    (encoder_batch_size, encoder_sequence_length), device=device
                )
            encoder_extended_attention_mask = self.get_extended_attention_mask(
                encoder_attention_mask, (encoder_batch_size, encoder_sequence_length), device
            )
        else:
            encoder_extended_attention_mask = None

        head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mode=mode,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )
