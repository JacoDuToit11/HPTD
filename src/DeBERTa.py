import transformers
import torch
import torch.nn as nn
import numpy as np
import os

import torch.nn as nn
from torch.nn import LayerNorm

def swish(x):
  return x * torch.sigmoid(x)

def linear_act(x):
  return x

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish, "tanh": torch.tanh, "linear": linear_act, 'sigmoid': torch.sigmoid}

class LMMaskPredictionHead(nn.Module):
  """ Replaced token prediction head
  """
  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.transform_act_fn = ACT2FN[config.hidden_act] \
      if isinstance(config.hidden_act, str) else config.hidden_act
    self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
    self.classifier = nn.Linear(config.hidden_size, 1)

  def forward(self, hidden_states, input_ids, input_mask, lm_labels=None):
    # b x d
    ctx_states = hidden_states[:,0,:]
    seq_states = self.LayerNorm(ctx_states.unsqueeze(-2) + hidden_states)
    seq_states = self.dense(seq_states)
    seq_states = self.transform_act_fn(seq_states)

    # b x max_len
    logits = self.classifier(seq_states).squeeze(-1)
    mask_loss = torch.tensor(0).to(logits).float()
    mask_labels = None
    if lm_labels is not None:
      mask_logits = logits.view(-1)
      _input_mask = input_mask.view(-1).to(mask_logits)
      input_idx = (_input_mask>0).nonzero().view(-1)
      mask_labels = ((lm_labels>0) & (lm_labels!=input_ids)).view(-1)
      mask_labels = torch.gather(mask_labels.to(mask_logits), 0, input_idx)
      mask_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
      mask_logits = torch.gather(mask_logits, 0, input_idx).float()
      mask_loss = mask_loss_fn(mask_logits, mask_labels)
    return logits

from transformers import DebertaV2Model, DebertaV2PreTrainedModel

class DebertaV3ForPreTraining(DebertaV2PreTrainedModel):
  def __init__(self, config):
    super().__init__(config)
    
    # base Transformer
    self.deberta = DebertaV2Model(config)

    # mask predictions head
    self.mask_predictions = LMMaskPredictionHead(config)

  def forward(self, input_ids, attention_mask=None, token_type_ids = None, labels=None, position_ids=None):
    outputs = self.deberta(input_ids, attention_mask, token_type_ids)
    sequence_output = outputs[0]
       
    # apply heads
    logits = self.mask_predictions(sequence_output, input_ids, attention_mask)
    return logits