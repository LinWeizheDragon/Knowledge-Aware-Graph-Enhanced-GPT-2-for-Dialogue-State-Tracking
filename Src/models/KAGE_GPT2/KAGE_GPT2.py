"""
KAGE_GPT2.py: Model Class for the paper:
Lin, W., Tseng, B. H., & Byrne, B. (2021). Knowledge-Aware Graph-Enhanced GPT-2 for Dialogue State Tracking. EMNLP 2021.
https://arxiv.org/abs/2104.04466v3
"""

__author__ = "Weizhe Lin"
__copyright__ = "Copyright 2021, Weizhe Lin"
__version__ = "1.0.0"
__email__ = "wl356@cam.ac.uk"
__status__ = "Published for Github"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from transformers import BertTokenizer, BertPreTrainedModel, BertModel, GPT2PreTrainedModel, GPT2Model
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.modeling_bert import BertPredictionHeadTransform
from transformers.modeling_outputs import CausalLMOutputWithPastAndCrossAttentions
from typing import Any, Dict, Iterable, List, Optional, Tuple
from transformers.generation_logits_process import (
    LogitsProcessorList,
)
import warnings

from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import Conv1D


class CustomCausalLMOutputWithPastAndCrossAttentions(ModelOutput):
    """
    Updated class for causal language model (or autoregressive) outputs.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`, with each tensor of shape :obj:`(2,
            batch_size, num_heads, sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            :obj:`past_key_values` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Cross attentions weights after the attention softmax, used to compute the weighted average in the
            cross-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    output_hidden_states: Optional[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


from .graph_model import GraphModel


class KAGEModel(GPT2PreTrainedModel):
    """
    The Model Class for KAGE
    """
    authorized_missing_keys = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config, sys_config=None):
        super().__init__(config)
        self.config = config
        self.sys_config = sys_config
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd * 2, config.vocab_size, bias=False)
        self.reset_output_size = False
        self.init_weights()
        # self.decode_layer = nn.Linear(config.n_embd * 2, config.vocab_size, bias=False)
        self.graph_model = GraphModel(self.sys_config)
        self.dropout = nn.Dropout(0.1)
        # Get embedding layer of GPT2
        self.embedding_layer = self.transformer.wte
        self.cls_loss_linear = nn.Linear(config.n_embd, config.n_embd, bias=True)

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights depending of whether we are using TorchScript or not"""
        if self.config.torchscript:
            output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        else:
            output_embeddings.weight = input_embeddings.weight

        if not self.reset_output_size:
            output_embeddings.weight = nn.Parameter(torch.zeros((self.config.vocab_size, self.config.n_embd * 2)))
            output_embeddings.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            print(output_embeddings.weight.shape)

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = torch.nn.functional.pad(
                output_embeddings.bias.data,
                (
                    0,
                    output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                ),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            print('skip reset!')
            if self.reset_output_size:
                output_embeddings.out_features = input_embeddings.num_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    def pre_forward(self,
                    pre_input_ids=None,
                    pre_attention_mask=None,
                    pre_ignore_len=None,
                    pre_ds_indice=None,
                    pre_past_key_values=None,
                    ):
        """Pre-extraction Forward Function

        Args:
            pre_input_ids (Tensor, optional): Pre-extraction sentences. Defaults to None.
            pre_attention_mask (Tensor, optional): Pre-extraction attention masks. Defaults to None.
            pre_ignore_len (Tensor, optional): Pre-extraction ignore length. Defaults to None.
            pre_ds_indice (Tensor, optional): Domain-slot special token position in pre-extraction. Defaults to None.
            pre_past_key_values (Tensor, optional): The same as huggingface transformer past_key_values. Defaults to None.

        Returns:
            Dict: dict(
                ds_embeddings=ds_embeddings, # Domain-slot embeddings of all domain-slots
                pre_past_key_values=pre_past_key_values,  # Not used in most cases
            )
        """
        pre_transformer_outputs = self.transformer(
            input_ids=pre_input_ids,
            past_key_values=pre_past_key_values,
            attention_mask=pre_attention_mask,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            return_dict=True,
        )
        pre_hidden_states = pre_transformer_outputs[0]
        pre_past_key_values = pre_transformer_outputs.past_key_values
        # print(pre_hidden_states.shape) # B x L x hidden_size

        ds_embeddings = []
        pre_ds_indice = torch.LongTensor(pre_ds_indice)
        batch_size, len_ds_pairs = pre_ds_indice.size()
        for ds_index in range(len_ds_pairs):
            # print(pre_ds_indice[:, ds_index].shape)
            sub_ds_embedding = pre_hidden_states[range(batch_size), pre_ds_indice[:, ds_index]]
            ds_embeddings.append(sub_ds_embedding)

        return dict(
            ds_embeddings=ds_embeddings,
            pre_past_key_values=pre_past_key_values,
        )

    def graph_forward(self, ds_embeddings, cls_labels=None):
        """Forward Function of graph modules

        Args:
            ds_embeddings (Tensor): Domain-slot embeddings
            cls_labels ([type], optional): [description]. Defaults to None.

        Returns:
            Dict: dict( 
                ds_embeddings=ds_embeddings_output, # Domain-slot embeddings after GATs
                loss=loss,
                logits=logits,
            )
        """
        # return ds_embeddings
        ts_ds_embeddings = torch.stack(ds_embeddings)  # N x B x F
        ts_ds_embeddings = ts_ds_embeddings.permute(1, 2, 0)  # B x F x N
        E = 1
        B, _, N_ds = ts_ds_embeddings.size()

        if self.sys_config.model_config.graph_mode == 'full':
            # use pre-defined ontology
            S = self.S.repeat(B, E, 1, 1).to(ts_ds_embeddings.device)
            self.graph_model.add_GSO(S)

            ontology_value_embeddings = self.ontology_value_embeddings.permute(1, 0).repeat(B, 1, 1).to(ts_ds_embeddings.device)
            # print(ontology_value_embeddings.shape)
            merged_embeddings = torch.cat([ts_ds_embeddings, ontology_value_embeddings], dim=-1)
            # print(merged_embeddings.shape)
            ts_merged_embeddings_output = self.graph_model(merged_embeddings)

            if self.sys_config.model_config.residual == True:
                # Residual Connection
                ts_merged_embeddings_output += merged_embeddings

            # print(ts_merged_embeddings_output.shape)
            ds_embeddings_output = [ts_merged_embeddings_output[:, :, i] for i in range(N_ds)]
            # print(ds_embeddings_output.shape)
            ts_ds_embeddings_output = ts_merged_embeddings_output[:, :, :N_ds]

        else:
            # create new ontology - connect domain-slots
            S = torch.ones((N_ds, N_ds)) - torch.eye(N_ds)
            S = S.repeat(B, E, 1, 1).to(ts_ds_embeddings.device)
            self.graph_model.add_GSO(S)

            ts_ds_embeddings_output = self.graph_model(ts_ds_embeddings)
            if self.sys_config.model_config.residual == True:
                # Residual Connection
                ts_ds_embeddings_output += ts_ds_embeddings
            ds_embeddings_output = [ts_ds_embeddings_output[:, :, i] for i in range(N_ds)]

        loss = None
        logits = []

        if cls_labels is not None:
            if self.sys_config.model_config.cls_loss:
                loss_fn = nn.NLLLoss()#(reduction='none')
                # self.mapping_class_indices_to_ontology[ds_index]

                # v1 after GATs
                ds_emb = ts_ds_embeddings_output # B x hidden_size x N_ds
                # v2 before GATs
                ds_emb = ts_ds_embeddings # B x hidden_size x N_ds

                ds_emb = self.cls_loss_linear(ds_emb.permute(0, 2, 1)).permute(0, 2, 1)
                # print(ds_emb.shape)
                value_emb = self.ontology_value_embeddings # N_v x hidden_size
                # print(value_emb.shape)
                dot_results = torch.matmul(ds_emb.permute(0, 2, 1), value_emb.T)  # B x N_ds x N_v
                # print(dot_results.shape)
                for ds_index in range(len(ds_embeddings_output)):
                    cls_label = cls_labels[:, ds_index]  # B
                    value_indices = self.mapping_class_indices_to_ontology[ds_index]
                    sub_dot_results = dot_results[:, ds_index, value_indices]
                    # print(sub_dot_results.shape)
                    sub_dot_results = F.log_softmax(sub_dot_results, dim=-1)
                    sub_loss = loss_fn(sub_dot_results, cls_label)

                    if loss is None:
                        loss = sub_loss
                    else:
                        loss += sub_loss
                    logits.append(sub_dot_results)
            else:
                loss = 0

        # Get attentions
        attentions = self.graph_model.get_GSO()
        self.graph_attentions = attentions
        return dict(
            ds_embeddings=ds_embeddings_output,
            loss=loss,
            logits=logits,
        )

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            ds_indice=None,
            ds_embeddings=None,
            **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        # embedding_layer = self.transformer.get_input_embeddings()
        if "past" in kwargs:
            warnings.warn(
                "The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("past")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if ds_indice:
        #     # This is teacher force, we have ds indice input
        #     inputs_embeds = embedding_layer(input_ids)  # B x L x hidden_size
        #     #
        #     # print(inputs_embeds.shape)
        #     ds_indice = torch.LongTensor(ds_indice)
        #     # print(ds_indice.shape) # B x 30 x 2
        #     batch_size, len_ds_pairs, _ = ds_indice.size()
        #     for ds_index in range(len_ds_pairs):
        #         # extract_embeds = inputs_embeds[range(ds_indice.shape[0]), ds_indice[:, ds_index, 1]]
        #         # print(extract_embeds.shape)
        #         for batch_id in range(batch_size):
        #             # print(inputs_embeds[batch_id, ds_indice[batch_id, ds_index, 1]].shape)
        #             # print(ds_embeddings[ds_index][batch_id].shape)
        #             inputs_embeds[batch_id, ds_indice[batch_id, ds_index, 1]] += ds_embeddings[ds_index][batch_id]
        #             # print(extract_embeds.shape)
        #     # print(attention_mask.shape)
        #     # print(inputs_embeds.shape)
        # else:
        #     # This is generation, we use pre-calculated embeddings from input
        #     assert inputs_embeds is not None

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]  # B x L x hidden_size
        if ds_indice is not None and ds_embeddings is not None:
            ds_indice = torch.LongTensor(ds_indice)
            cat_hidden_states = torch.zeros_like(hidden_states)  # B x L x hidden_size
            # print(ds_indice.shape) # B x 30 x 2
            batch_size, len_ds_pairs, _ = ds_indice.size()
            for ds_index in range(len_ds_pairs):
                for batch_id in range(batch_size):
                    ##### v1 ######
                    start_index = ds_indice[batch_id, ds_index, 0]
                    if ds_index < ds_indice.shape[1] - 1:
                        end_index = ds_indice[batch_id, ds_index + 1, 0] - 1
                    else:
                        end_index = hidden_states.shape[1]
                    ###############
                    ##### v2 ######
                    # start_index = ds_indice[batch_id, ds_index, 1] - 1 # last token of ds pair
                    # if ds_index < ds_indice.shape[1] - 1:
                    #     end_index = ds_indice[batch_id, ds_index + 1, 0] - 1
                    # else:
                    #     end_index = hidden_states.shape[1]
                    ###############
                    # print(start_index, end_index)
                    # print(cat_hidden_states[batch_id, start_index:end_index].shape)
                    # print(ds_embeddings[ds_index][batch_id].shape)
                    cat_hidden_states[batch_id, start_index:end_index] += ds_embeddings[ds_index][batch_id]
                    # print()
                    # input()
            # print(cat_hidden_states.shape)
            hidden_states = torch.cat([hidden_states, cat_hidden_states], dim=-1)
            hidden_states = self.dropout(hidden_states)
            # print(hidden_states.shape)
            # input()
            # # This is teacher force, we have ds indice input
            # ds_indice = torch.LongTensor(ds_indice)
            # # print(ds_indice.shape) # B x 30 x 2
            # batch_size, len_ds_pairs, _ = ds_indice.size()
            # for ds_index in range(len_ds_pairs):
            #     # for each ds
            #     for batch_id in range(batch_size):
            #         # print(hidden_states[batch_id, ds_indice[batch_id, ds_index, 1]].shape)
            #         # print(ds_embeddings[ds_index][batch_id].shape)
            #         hidden_states[batch_id, ds_indice[batch_id, ds_index, 1]] += ds_embeddings[ds_index][batch_id]
            #         # pass

        # input()
        # print(hidden_states.shape)
        # print(self.decode_layer.state_dict()['weight'].shape)
        if ds_embeddings is not None:
            lm_logits = self.lm_head(hidden_states)
        else:
            lm_logits = None
        # print(lm_logits.shape)
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CustomCausalLMOutputWithPastAndCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            output_hidden_states=hidden_states,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def greedy_search(
            self,
            input_ids: torch.LongTensor,
            logits_processor: Optional[LogitsProcessorList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            **model_kwargs
    ):
        r"""
        Generates sequences for models with a language modeling head using greedy decoding.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes it as an empty
                :obj:`torch.LongTensor` of shape :obj:`(1,)`.
            logits_processor (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsProcessor` used to modify the prediction scores of the language modeling
                head applied at each generation step.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the :obj:`forward` function of the
                model. If model is an encoder-decoder model the kwargs should include :obj:`encoder_outputs`.

        Return:
            :obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`: The generated
            sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or shorter if all
            batches finished early due to the :obj:`eos_token_id`.

        Examples::

            >>> from transformers import (
            ... AutoTokenizer,
            ... AutoModelForCausalLM,
            ... LogitsProcessorList,
            ... MinLengthLogitsProcessor,
            ... )

            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

            >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
            >>> model.config.pad_token_id = model.config.eos_token_id

            >>> input_prompt = "Today is a beautiful day, and"
            >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

            >>> # instantiate logits processors
            >>> logits_processor = LogitsProcessorList([
            ...     MinLengthLogitsProcessor(15, eos_token_id=model.config.eos_token_id),
            ... ])

            >>> outputs = model.greedy_search(input_ids, logits_processor=logits_processor)

            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        """
        # print('greedy search overide!')
        # print(model_kwargs)
        pre_input_ids = model_kwargs.get('pre_input_ids', None)
        pre_attention_mask = model_kwargs.get('pre_attention_mask', None)
        pre_ds_indice = model_kwargs.get('pre_ds_indice', None)
        sep_token_id = model_kwargs.get('sep_token_id', None)
        bos_token_id = model_kwargs.get('bos_id', None)
        ds_ids = model_kwargs.get('ds_ids', None)

        assert pre_input_ids is not None
        assert pre_ds_indice is not None
        assert sep_token_id is not None
        assert ds_ids is not None

        # Obtain ds embeddings
        assert input_ids.shape[0] == 1  # batch size 1

        pre_forward_results = self.pre_forward(
            pre_input_ids=pre_input_ids,
            pre_attention_mask=pre_attention_mask,
            pre_ignore_len=None,
            pre_ds_indice=pre_ds_indice,
        )

        ds_embeddings = pre_forward_results['ds_embeddings']
        graph_forward_results = self.graph_forward(
            ds_embeddings=ds_embeddings
        )
        ds_embeddings = graph_forward_results['ds_embeddings']

        lm_head = self.get_output_embeddings()

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        # init sequence length tensors
        sequence_lengths, unfinished_sequences, cur_len = self._init_sequence_length_for_generation(
            input_ids, max_length
        )

        count = 0
        flag = -1
        while cur_len < max_length:

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)
            hidden_states = outputs.output_hidden_states
            cat_hidden_states = torch.zeros_like(hidden_states)  # B x L x hidden_size

            # print(input_ids[0, -1])
            # if input_ids[0, -1] == sep_token_id:
            #     count += 1
            #     flag = False
            # else:
            #     if (input_ids[0, -len(ds_ids[count]):].cpu() == torch.LongTensor(ds_ids[count])).all():
            #         # from here add embeddings
            #         flag = True
            #
            # if flag:
            #     cat_hidden_states[:, -1, :] = ds_embeddings[count]

            ########### v1 ###########
            if input_ids[0, -1] == sep_token_id:
                count += 1

            if input_ids[0, -1] != bos_token_id and input_ids[0, -1] != sep_token_id:
                if count < len(ds_embeddings):
                    cat_hidden_states[:, -1, :] = ds_embeddings[count]
                else:
                    cat_hidden_states[:, -1, :] = ds_embeddings[-1]

            ########### v2 ###########
            # if input_ids[0, -1] == sep_token_id:
            #     count += 1
            #     if count > 29:
            #         count = 29
            #
            # if input_ids.shape[1] >= len(ds_ids[count]):
            #     if (input_ids[0, -len(ds_ids[count]):].cpu() == torch.LongTensor(ds_ids[count])).all():
            #         # when found the last token of the ds pair, set flag
            #         flag = input_ids.shape[1] - 1
            #
            # if flag > 0:
            #     if count < len(ds_embeddings):
            #         hidden_states[:, flag:, :] = ds_embeddings[count]
            #     else:
            #         hidden_states[:, flag:, :] = ds_embeddings[-1]
            ##############################
            if input_ids[0, -1] == bos_token_id or input_ids[0, -1] == sep_token_id:
                # close flag when meeting BOS / SEP toke
                flag = -1

            hidden_states = torch.cat([hidden_states, cat_hidden_states], dim=-1)
            hidden_states = self.dropout(hidden_states)
            # print(hidden_states.shape)
            # input()

            # Output logits
            next_token_logits = lm_head(hidden_states)[:, -1, :]
            # next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            scores = logits_processor(input_ids, next_token_logits)

            # argmax
            next_tokens = torch.argmax(scores, dim=-1)
            # print(next_tokens, eos_token_id)
            if count >= len(ds_ids):
                if next_tokens == sep_token_id:
                    next_tokens[:] = eos_token_id
            # add code that transfomers next_tokens to tokens_to_add
            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)

            # add token and increase length by one
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            # update sequence length
            if eos_token_id is not None:
                sequence_lengths, unfinished_sequences = self._update_seq_length_for_generation(
                    sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id
                )

            # update model kwargs
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sequences.max() == 0:
                break

            # increase cur_len
            cur_len = cur_len + 1

        return input_ids

    def refresh_embeddings(self):
        '''
        Refresh value candidate embeddings using the wte layer
        '''
        self.ontology_value_embeddings = torch.zeros((len(self.ontology_value_list), self.config.n_embd)).to(self.sys_config.device)
        for index, text in self.ontology_value_id2tokenized_text.items():
            ids = torch.LongTensor(text).to(self.sys_config.device)
            embeddings = self.embedding_layer(ids)
            # TODO Embedding aggregatioon
            agg_embeddings = torch.sum(embeddings, dim=0)
            # print(agg_embeddings.shape)
            self.ontology_value_embeddings[index] = agg_embeddings

        self.ontology_value_embeddings = self.ontology_value_embeddings.detach()

        # self.value_id2embedding = {}
        # self.value_embeddings = {}
        # value_id2tokenized_text = self.value_id2tokenized_text
        #
        # # all value embeddings in the graph
        # self.ontology_value_embeddings = torch.zeros((len(self.ontology_value_list), self.config.n_embd))
        #
        # # Obtain value embeddings for all nodes
        # for ds_index, str_ds_pair in enumerate(self.ds_list):
        #     self.value_id2embedding[str_ds_pair] = {}
        #     value_dict = value_id2tokenized_text[str_ds_pair]
        #     all_value_embeddings = []
        #     for i in range(len(value_dict)):
        #         # print('ds {} item {}: {}'.format(str_ds_pair, i, self.value_id2text[str_ds_pair][i]))
        #         index_in_ontology = self.ontology_value_text2id[self.value_id2text[str_ds_pair][i]]
        #         # print('corresponding index in ontology:', index_in_ontology)
        #         ids = torch.LongTensor(value_dict[i]).to(self.sys_config.device)
        #         embeddings = self.embedding_layer(ids)
        #         # TODO Embedding aggregatioon
        #         agg_embeddings = torch.sum(embeddings, dim=0)
        #         self.value_id2embedding[str_ds_pair][i] = agg_embeddings
        #         # print(agg_embeddings.shape)
        #         self.ontology_value_embeddings[index_in_ontology] = agg_embeddings
        #         all_value_embeddings.append(agg_embeddings)
        #     all_value_embeddings = torch.stack(all_value_embeddings)
        #     self.value_embeddings[ds_index] = all_value_embeddings
        #     # print(self.ontology_value_embeddings)
        #     # input()

    def add_KB(self,
               value_id2tokenized_text,
               value_id2text,
               ds_list,
               ontology_value_list,
               ontology_value_text2id,
               ontology_value_id2text,
               ontology_value_id2tokenized_text,
               ):
        """Add KB data to the model

        Args:
            value_id2tokenized_text (Dict): {str_ds_pair: {0: [id1 id2 id3], 1: ...}}
            value_id2text (Dict): {str_ds_pair: {0: 'none', 1: ...}}
            ds_list (List): list of ds pairs
            ontology_value_list (List): all values
            ontology_value_text2id (Dict):  {'none': 0, 'dont care':1,...}
            ontology_value_id2text (Dict): {0: 'none', 1: 'dont care',...}
            ontology_value_id2tokenized_text (Dict): {0: [id1 id2 id3], 1: ...}
        """
        self.value_id2tokenized_text = value_id2tokenized_text
        self.value_id2text = value_id2text
        self.ds_list = ds_list
        self.ontology_value_list = ontology_value_list
        self.ontology_value_text2id = ontology_value_text2id
        self.ontology_value_id2text = ontology_value_id2text
        self.ontology_value_id2tokenized_text = ontology_value_id2tokenized_text
        # input('add KB')
        self.mapping_class_indices_to_ontology = {}
        for ds_index, str_ds_pair in enumerate(ds_list):
            all_indices_of_values = []
            # self.mapping_class_indices_to_ontology[ds_index] = []
            for id in range(len(value_id2text[str_ds_pair])):
                text = value_id2text[str_ds_pair][id]
                all_indices_of_values.append(ontology_value_text2id[text])
            self.mapping_class_indices_to_ontology[ds_index] = all_indices_of_values

        self.refresh_embeddings()

        if self.sys_config.model_config.graph_mode == 'full':
            N_ds = len(self.ds_list)
            N_v = len(self.ontology_value_list)
            N = N_ds + N_v
            S = torch.zeros((N, N))

            for ds_index, str_ds_pair in enumerate(self.ds_list):
                value_dict = self.value_id2text[str_ds_pair]
                for i in range(len(value_dict)):
                    # print('ds {} item {}: {}'.format(str_ds_pair, i, self.value_id2text[str_ds_pair][i]))
                    index_in_ontology = self.ontology_value_text2id[self.value_id2text[str_ds_pair][i]]
                    # print('corresponding index in ontology:', index_in_ontology)

                    # i: target, j: src
                    # connect ds node to value node
                    if self.sys_config.model_config.connect_type == 'ds_value_only':
                        S[index_in_ontology + N_ds, ds_index] = 1
                    else:
                        # allow all ds nodes to pass features to this value node
                        S[index_in_ontology + N_ds, :N_ds] = 1

                    # Connect value node to ds node
                    S[ds_index, index_in_ontology + N_ds] = 1
                    # print('{} and {} connected'.format(index_in_ontology + N_ds, ds_index))

            self.S = S

