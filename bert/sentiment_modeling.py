from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

import math
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from bert.modeling import BertModel, BERTLayerNorm
import torch
from bert.dynamic_rnn import DynamicLSTM
import torch.nn.functional as F
# Do not hard-code CUDA. New tensors below are created on the same device as existing tensors.

def flatten(x):
    if len(x.size()) == 2:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        return x.view([batch_size * seq_length])
    elif len(x.size()) == 3:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        hidden_size = x.size()[2]
        return x.view([batch_size * seq_length, hidden_size])
    else:
        raise Exception()

def reconstruct(x, ref):
    if len(x.size()) == 1:
        batch_size = ref.size()[0]
        turn_num = ref.size()[1]
        return x.view([batch_size, turn_num])
    elif len(x.size()) == 2:
        batch_size = ref.size()[0]
        turn_num = ref.size()[1]
        sequence_length = x.size()[1]
        return x.view([batch_size, turn_num, sequence_length])
    else:
        raise Exception()

def flatten_emb_by_sentence(emb, emb_mask):
    batch_size = emb.size()[0]
    seq_length = emb.size()[1]
    flat_emb = flatten(emb)
    flat_emb_mask = emb_mask.view([batch_size * seq_length])
    return flat_emb[flat_emb_mask.nonzero().squeeze(), :]

def get_span_representation(span_starts, span_ends, input, input_mask):
    '''
    :param span_starts: [N, M]
    :param span_ends: [N, M]
    :param input: [N, L, D]
    :param input_mask: [N, L]
    :return: [N*M, JR, D], [N*M, JR]
    '''
    #print(input.size())
    input_mask = input_mask.to(dtype=span_starts.dtype)  # fp16 compatibility
    input_len = torch.sum(input_mask, dim=-1) # [N]
    word_offset = torch.cumsum(input_len, dim=0) # [N]
    word_offset -= input_len

    span_starts_offset = span_starts + word_offset.unsqueeze(1)
    span_ends_offset = span_ends + word_offset.unsqueeze(1)

    span_starts_offset = span_starts_offset.view([-1])  # [N*M]
    span_ends_offset = span_ends_offset.view([-1])

    span_width = span_ends_offset - span_starts_offset + 1
    JR = torch.max(span_width)

    context_outputs = flatten_emb_by_sentence(input, input_mask)  # [<N*L, D]
    text_length = context_outputs.size()[0]

    span_indices = torch.arange(JR).unsqueeze(0).to(span_starts_offset.device) + span_starts_offset.unsqueeze(1)  # [N*M, JR]
    span_indices = torch.min(span_indices, (text_length - 1)*torch.ones_like(span_indices))
    span_text_emb = context_outputs[span_indices, :]    # [N*M, JR, D]

    row_vector = torch.arange(JR).to(span_width.device)
    span_mask = row_vector < span_width.unsqueeze(-1)   # [N*M, JR]
    return span_text_emb, span_mask

def get_self_att_representation(input, input_score, input_mask):
    '''
    :param input: [N, L, D]
    :param input_score: [N, L]
    :param input_mask: [N, L]
    :return: [N, D]
    '''
    input_mask = input_mask.to(dtype=input_score.dtype)  # fp16 compatibility
    input_mask = (1.0 - input_mask) * -10000.0
    input_score = input_score + input_mask
    input_prob = nn.Softmax(dim=-1)(input_score)
    input_prob = input_prob.unsqueeze(-1)
    output = torch.sum(input_prob * input, dim=1)
    return output

def distant_cross_entropy(logits, positions, mask=None):
    '''
    :param logits: [N, L]
    :param positions: [N, L]
    :param mask: [N]
    '''
    log_softmax = nn.LogSoftmax(dim=-1)
    log_probs = log_softmax(logits)
    if mask is not None:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               (torch.sum(positions.to(dtype=log_probs.dtype), dim=-1) + mask.to(dtype=log_probs.dtype)))
    else:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               torch.sum(positions.to(dtype=log_probs.dtype), dim=-1))
    return loss

def pad_sequence(sequence, length):
    while len(sequence) < length:
        sequence.append(0)
    return sequence

def convert_crf_output(outputs, sequence_length, device):
    predictions = []
    for output in outputs:
        pred = pad_sequence(output[0], sequence_length)
        predictions.append(torch.tensor(pred, dtype=torch.long))
    predictions = torch.stack(predictions, dim=0)
    if device is not None:
        predictions = predictions.to(device)
    return predictions

class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.classifier1 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.classifier2 = nn.Linear(int(hidden_size/2), num_label)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        features_output1 = nn.ReLU()(features_output1)
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2



def contrastive_loss(sequence_output, start_positions=None, end_positions=None,
                     span_starts=None, span_ends=None, polarity_labels=None,
                     label_masks=None, sentiment_prototypes=None,
                     attention_mask=None, temperature=0.07,
                     hard_negative_radius=2, eps=1e-8):
    """Boundary-aware hard-negative InfoNCE loss.

    The earlier token contrastive loss only contrasted generic start/end token
    sets. This version uses the actual gold span pairs and constructs hard
    negatives that match ABSA boundary errors:
      1) overlapping spans with shifted start/end boundaries;
      2) other aspect spans in the same sentence;
      3) span-sentiment negatives through neutral/positive/negative prototypes.

    ``span_starts/span_ends/polarity_labels/label_masks`` may contain extra
    pseudo candidates. Only candidates with a valid non-``other`` label are used
    as positives; pseudo ``other`` candidates do not become positives.
    """
    if sequence_output.dim() != 3:
        raise ValueError("sequence_output must be [batch_size, seq_len, hidden_dim]")

    batch_size, seq_len, hidden_size = sequence_output.size()
    normalized_output = F.normalize(sequence_output, p=2, dim=-1)
    losses = []

    if span_starts is None or span_ends is None or polarity_labels is None or label_masks is None:
        # Backward-compatible fallback: build positives from binary start/end
        # positions, pairing starts and ends in sorted order when possible.
        if start_positions is None or end_positions is None:
            return sequence_output.sum() * 0.0
        batch_span_starts, batch_span_ends, batch_labels, batch_masks = [], [], [], []
        for b in range(batch_size):
            starts = torch.nonzero(start_positions[b] > 0, as_tuple=False).view(-1)
            ends = torch.nonzero(end_positions[b] > 0, as_tuple=False).view(-1)
            m = min(starts.numel(), ends.numel())
            if m == 0:
                batch_span_starts.append(sequence_output.new_zeros((1,), dtype=torch.long))
                batch_span_ends.append(sequence_output.new_zeros((1,), dtype=torch.long))
                batch_labels.append(sequence_output.new_zeros((1,), dtype=torch.long))
                batch_masks.append(sequence_output.new_zeros((1,), dtype=torch.long))
            else:
                batch_span_starts.append(starts[:m])
                batch_span_ends.append(ends[:m])
                batch_labels.append(sequence_output.new_ones((m,), dtype=torch.long))
                batch_masks.append(sequence_output.new_ones((m,), dtype=torch.long))
        max_m = max(x.numel() for x in batch_span_starts)
        span_starts = sequence_output.new_zeros((batch_size, max_m), dtype=torch.long)
        span_ends = sequence_output.new_zeros((batch_size, max_m), dtype=torch.long)
        polarity_labels = sequence_output.new_zeros((batch_size, max_m), dtype=torch.long)
        label_masks = sequence_output.new_zeros((batch_size, max_m), dtype=torch.long)
        for b in range(batch_size):
            m = batch_span_starts[b].numel()
            span_starts[b, :m] = batch_span_starts[b]
            span_ends[b, :m] = batch_span_ends[b]
            polarity_labels[b, :m] = batch_labels[b]
            label_masks[b, :m] = batch_masks[b]

    for b in range(batch_size):
        valid = (label_masks[b] > 0) & (polarity_labels[b] > 0) & (span_starts[b] > 0) & (span_ends[b] >= span_starts[b])
        if valid.sum() == 0:
            continue
        gold_starts = span_starts[b][valid].long()
        gold_ends = span_ends[b][valid].long()
        gold_labels = polarity_labels[b][valid].long()
        valid_token_limit = int(attention_mask[b].sum().item()) if attention_mask is not None else seq_len
        gold_pairs = {(int(s.item()), int(e.item())) for s, e in zip(gold_starts, gold_ends)}

        for s, e, y in zip(gold_starts, gold_ends, gold_labels):
            s_i, e_i = int(s.item()), int(e.item())
            if s_i >= seq_len or e_i >= seq_len or e_i < s_i:
                continue

            # start -> end: positive is the true end; negatives are shifted ends
            # and ends of other aspects in the same sentence.
            end_candidates = [e_i]
            for delta in range(-hard_negative_radius, hard_negative_radius + 1):
                cand_e = e_i + delta
                if cand_e == e_i:
                    continue
                if s_i <= cand_e < valid_token_limit and (s_i, cand_e) not in gold_pairs:
                    end_candidates.append(cand_e)
            for other_e in gold_ends.tolist():
                other_e = int(other_e)
                if other_e != e_i and s_i <= other_e < valid_token_limit:
                    end_candidates.append(other_e)
            end_candidates = list(dict.fromkeys(end_candidates))
            if len(end_candidates) > 1:
                anchor = normalized_output[b, s_i]
                cand = normalized_output[b, torch.tensor(end_candidates, dtype=torch.long, device=sequence_output.device)]
                logits = torch.matmul(cand, anchor) / max(float(temperature), eps)
                target = torch.zeros(1, dtype=torch.long, device=sequence_output.device)
                losses.append(F.cross_entropy(logits.unsqueeze(0), target))

            # end -> start: positive is the true start; negatives are shifted
            # starts and starts of other aspects.
            start_candidates = [s_i]
            for delta in range(-hard_negative_radius, hard_negative_radius + 1):
                cand_s = s_i + delta
                if cand_s == s_i:
                    continue
                if 0 < cand_s <= e_i and cand_s < valid_token_limit and (cand_s, e_i) not in gold_pairs:
                    start_candidates.append(cand_s)
            for other_s in gold_starts.tolist():
                other_s = int(other_s)
                if other_s != s_i and 0 < other_s <= e_i:
                    start_candidates.append(other_s)
            start_candidates = list(dict.fromkeys(start_candidates))
            if len(start_candidates) > 1:
                anchor = normalized_output[b, e_i]
                cand = normalized_output[b, torch.tensor(start_candidates, dtype=torch.long, device=sequence_output.device)]
                logits = torch.matmul(cand, anchor) / max(float(temperature), eps)
                target = torch.zeros(1, dtype=torch.long, device=sequence_output.device)
                losses.append(F.cross_entropy(logits.unsqueeze(0), target))

            # span -> sentiment prototype: makes same-boundary / different-sentiment
            # cases explicit when sentiment labels are available.
            if sentiment_prototypes is not None and int(y.item()) in (1, 2, 3):
                label_to_proto = {1: 0, 2: 1, 3: 2}  # neutral, positive, negative
                span_rep = F.normalize((sequence_output[b, s_i] + sequence_output[b, e_i]) * 0.5, p=2, dim=-1)
                proto = F.normalize(sentiment_prototypes, p=2, dim=-1)
                logits = torch.matmul(proto, span_rep) / max(float(temperature), eps)
                target = torch.tensor([label_to_proto[int(y.item())]], dtype=torch.long, device=sequence_output.device)
                losses.append(F.cross_entropy(logits.unsqueeze(0), target))

    if len(losses) == 0:
        return sequence_output.sum() * 0.0
    return torch.stack(losses).mean()

class BCEFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.6, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        # pt = _input
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
class BertForSpanAspectExtraction(nn.Module):

    def __init__(self, config):
        super(BertForSpanAspectExtraction, self).__init__()
        self.bert = BertModel(config)

        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_outputs = nn.Linear(config.hidden_size, 1)
        self.span_outputs=nn.Linear(2*config.hidden_size,1)
        self.span_embedding = MultiNonLinearClassifier(config.hidden_size * 2, 1, 0.1)  # 0.1 dropout
        self.activation_sigmoid = nn.Sigmoid()
        self.activation_softmax = nn.Softmax(dim=-1)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, start_positions=None, end_positions=None, weight_start=None,
                weight_end=None, weight_span=None,weight_bias=None):
        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = all_encoder_layers[-1]
        batch_size, seq_len, hid_size = sequence_output.size()
        start_logits = self.start_outputs(sequence_output) # [batch_size, seq_len, 1]
        end_logits = self.end_outputs(sequence_output) # [batch_size, seq_len, 1]
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        start_extend = sequence_output.unsqueeze(2).expand(-1, -1, seq_len, -1) # [batch_size , seq_len ,seq_len, hidden_dim]
        end_extend = sequence_output.unsqueeze(1).expand(-1, seq_len, -1, -1) #  [batch_size , seq_len ,seq_len, hidden_dim]
        span_matrix = torch.cat([start_extend, end_extend], 3) # batch x seq_len x seq_len x 2*hidden_dim
        span_logits = self.span_embedding(span_matrix)  # batch x seq_len x seq_len x 1
        span_logits = span_logits.squeeze(-1)  # [batch, seq_len, seq_len]

        if start_positions is not None and end_positions is not None:
            start_positions_extend=start_positions.unsqueeze(2).expand(-1, -1, seq_len)
            end_positions_extend=end_positions.unsqueeze(1).expand(-1, seq_len, -1)
            span_positions = torch.mul(start_positions_extend,end_positions_extend)

            loss_fct = nn.BCELoss()
            start_logits = self.activation_softmax(start_logits)
            end_logits = self.activation_softmax(end_logits)  # [batch_size, seq_len]
            #loss_fct = BCEFocalLoss(gamma=2, alpha=0.1, reduction='elementwise_mean')
            start_loss = loss_fct(start_logits, start_positions.float())
            end_loss = loss_fct(end_logits, end_positions.float())

            #span_loss=BCEFocalLoss(gamma=0, alpha=weight_end, reduction='elementwise_mean')
            pos_weight = torch.tensor([weight_bias], dtype=span_logits.dtype, device=span_logits.device)
            span_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            span_loss = span_loss(span_logits.view(batch_size,-1), span_positions.view(batch_size, -1).float())
            

            #################################################################
            total_loss=weight_start*start_loss+weight_end*end_loss+weight_span*span_loss
            ################################################################

            return total_loss
        else:
            span_logits = torch.sigmoid(span_logits) # [batch , seq_len , seq_len]
            return start_logits, end_logits, span_logits

class BertForSpanAspectClassification(nn.Module):
    def __init__(self, config):
        super(BertForSpanAspectClassification, self).__init__()
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.affine = nn.Linear(config.hidden_size, 1)
        self.classifier = nn.Linear(config.hidden_size, 5)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, mode, attention_mask, input_ids=None, token_type_ids=None, span_starts=None, span_ends=None,
                labels=None, label_masks=None):
        '''
        :param input_ids: [N, L]
        :param token_type_ids: [N, L]
        :param attention_mask: [N, L]
        :param span_starts: [N, M]
        :param span_ends: [N, M]
        :param labels: [N, M]
        '''
        if mode == 'train':
            assert input_ids is not None and token_type_ids is not None
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]

            assert span_starts is not None and span_ends is not None and labels is not None
            span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_output,
                                                             attention_mask)  # [N*M, JR, D], [N*M, JR]
            span_score = self.affine(span_output)
            span_score = span_score.squeeze(-1)  # [N*M, JR]
            span_pooled_output = get_self_att_representation(span_output, span_score, span_mask)  # [N*M, D]

            span_pooled_output = self.dense(span_pooled_output)
            span_pooled_output = self.activation(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)
            cls_logits = self.classifier(span_pooled_output)  # [N*M, 4]

            cls_loss_fct = CrossEntropyLoss(reduction='none')
            flat_cls_labels = flatten(labels)
            flat_label_masks = flatten(label_masks)
            loss = cls_loss_fct(cls_logits, flat_cls_labels)
            mean_loss = torch.sum(loss * flat_label_masks.to(dtype=loss.dtype)) / torch.sum(flat_label_masks.to(dtype=loss.dtype))
            return mean_loss

        elif mode == 'inference':
            assert input_ids is not None and token_type_ids is not None
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]

            assert span_starts is not None and span_ends is not None
            span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_output,
                                                             attention_mask)  # [N*M, JR, D], [N*M, JR]
            span_score = self.affine(span_output)
            span_score = span_score.squeeze(-1)  # [N*M, JR]
            span_pooled_output = get_self_att_representation(span_output, span_score, span_mask)  # [N*M, D]

            span_pooled_output = self.dense(span_pooled_output)
            span_pooled_output = self.activation(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)
            cls_logits = self.classifier(span_pooled_output)  # [N*M, 4]
            return reconstruct(cls_logits, span_starts)

        else:
            raise Exception

class BertForJointSpanExtractAndClassification(nn.Module):
    def __init__(self, config, args):
        super(BertForJointSpanExtractAndClassification, self).__init__()
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_outputs = nn.Linear(config.hidden_size, 1)
        self.unary_affine = nn.Linear(config.hidden_size, 1)
        self.binary_affine = nn.Linear(config.hidden_size, 2)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.span_embedding = MultiNonLinearClassifier(config.hidden_size* 2, 1, 0.1)
        self.activation_relu = nn.ReLU()
        self.activation_sigmoid=nn.Sigmoid()
        self.activation_softmax = nn.Softmax(dim=-1)
        self.classifier = nn.Linear(config.hidden_size,5)
        self.pair_sentiment_embedding = nn.Embedding(5, config.hidden_size)
        self.pair_verifier = nn.Sequential(
            nn.Linear(config.hidden_size * 3 + 3, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 1),
        )
        self.use_pair_verifier = bool(getattr(args, "use_pair_verifier", False))
        # Sentiment-conditioned boundary extraction.  The order matches
        # absa.utils label ids for non-other polarities: neutral=1,
        # positive=2, negative=3.  These prototypes allow sentiment to
        # influence AE directly instead of only being predicted after AE.
        self.sentiment_prototypes = nn.Parameter(torch.empty(3, config.hidden_size))
        self.sentiment_start_outputs = nn.Linear(config.hidden_size, 1)
        self.sentiment_end_outputs = nn.Linear(config.hidden_size, 1)
        self.use_sentiment_conditioned_boundary = bool(getattr(args, "use_sentiment_conditioned_boundary", False))
        self.max_answer_length = int(getattr(args, "max_answer_length", 12))

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)
        nn.init.normal_(self.sentiment_prototypes, mean=0.0, std=config.initializer_range)

    def _pair_verifier_logits(self, span_features, cls_features, sentiment_ids,
                              boundary_confidences, ac_probs):
        """Score whether a decoded (span, sentiment) pair should be kept.

        Features follow the pair-level consistency verifier design:
        span representation, [CLS] representation, predicted sentiment
        embedding, boundary confidence, AC non-other probability, and AC
        other probability.
        """
        num_labels = ac_probs.size(-1)
        sentiment_ids = sentiment_ids.clamp(min=0, max=num_labels - 1)
        sentiment_features = self.pair_sentiment_embedding(sentiment_ids)

        safe_non_other_ids = sentiment_ids.clamp(min=1, max=num_labels - 1)
        non_other_prob = ac_probs.gather(1, safe_non_other_ids.unsqueeze(1)).squeeze(1)
        other_prob = ac_probs[:, 0]
        scalar_features = torch.stack([
            boundary_confidences.to(dtype=span_features.dtype),
            non_other_prob.to(dtype=span_features.dtype),
            other_prob.to(dtype=span_features.dtype),
        ], dim=-1)

        pair_features = torch.cat([
            span_features,
            cls_features,
            sentiment_features,
            scalar_features,
        ], dim=-1)
        return self.pair_verifier(pair_features).squeeze(-1)

    def _sentiment_boundary_logits(self, sequence_output, base_start_logits=None, base_end_logits=None):
        """Return neutral/positive/negative conditioned start/end logits."""
        proto = torch.tanh(self.sentiment_prototypes).view(1, 3, 1, -1)
        conditioned_output = sequence_output.unsqueeze(1) * proto
        sent_start_logits = self.sentiment_start_outputs(conditioned_output).squeeze(-1)
        sent_end_logits = self.sentiment_end_outputs(conditioned_output).squeeze(-1)
        if base_start_logits is not None:
            sent_start_logits = sent_start_logits + base_start_logits.unsqueeze(1)
        if base_end_logits is not None:
            sent_end_logits = sent_end_logits + base_end_logits.unsqueeze(1)
        return sent_start_logits, sent_end_logits

    @staticmethod
    def _sentiment_boundary_targets(span_starts, span_ends, polarity_labels, label_masks, seq_len,
                                    supervision_mask=None):
        """Build [B, 3, L] targets for neutral/positive/negative boundaries.

        ``supervision_mask`` should mark only reliable gold candidates.  Weak
        pseudo candidates can be useful for the AC loss, but they should not
        become hard boundary targets or they can corrupt the sentiment-specific
        start/end heads.
        """
        batch_size = span_starts.size(0)
        start_targets = span_starts.new_zeros((batch_size, 3, seq_len), dtype=torch.float)
        end_targets = span_ends.new_zeros((batch_size, 3, seq_len), dtype=torch.float)
        label_to_offset = {1: 0, 2: 1, 3: 2}
        for b in range(batch_size):
            for k in range(span_starts.size(1)):
                if int(label_masks[b, k].item()) == 0:
                    continue
                if supervision_mask is not None and int(supervision_mask[b, k].item()) == 0:
                    continue
                label = int(polarity_labels[b, k].item())
                if label not in label_to_offset:
                    continue
                s = int(span_starts[b, k].item())
                e = int(span_ends[b, k].item())
                if 0 <= s < seq_len and 0 <= e < seq_len and e >= s:
                    offset = label_to_offset[label]
                    start_targets[b, offset, s] = 1.0
                    end_targets[b, offset, e] = 1.0
        return start_targets, end_targets

    @staticmethod
    def _masked_bce_with_logits(logits, targets, mask):
        """BCEWithLogits loss for multi-label token/span targets with padding masked out."""
        targets = targets.to(dtype=logits.dtype)
        mask = mask.to(dtype=logits.dtype)
        while mask.dim() < logits.dim():
            mask = mask.unsqueeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = loss * mask
        return loss.sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def _span_validity_targets(start_positions, end_positions, attention_mask, max_answer_length=None,
                               span_starts=None, span_ends=None, polarity_labels=None,
                               label_masks=None, supervision_mask=None):
        """Build valid start-end pair targets and pair mask for the span-validity head.

        Do not use the cartesian product of all gold starts and all gold ends
        when true span pairs are available.  In a sentence with two aspects,
        start_1 x end_2 is usually a false span, and treating it as positive
        makes the span-validity head over-generate candidates.
        """
        batch_size, seq_len = start_positions.size()
        token_mask = attention_mask.to(dtype=torch.float)
        pair_mask = token_mask.unsqueeze(2) * token_mask.unsqueeze(1)
        tri_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=pair_mask.dtype, device=pair_mask.device))
        pair_mask = pair_mask * tri_mask.unsqueeze(0)
        if max_answer_length is not None and int(max_answer_length) > 0:
            row_ids = torch.arange(seq_len, device=pair_mask.device).view(1, seq_len, 1)
            col_ids = torch.arange(seq_len, device=pair_mask.device).view(1, 1, seq_len)
            length_mask = (col_ids - row_ids + 1 <= int(max_answer_length)).to(dtype=pair_mask.dtype)
            pair_mask = pair_mask * length_mask

        span_targets = start_positions.new_zeros((batch_size, seq_len, seq_len), dtype=torch.float)
        if span_starts is not None and span_ends is not None and label_masks is not None:
            for b in range(batch_size):
                for k in range(span_starts.size(1)):
                    if int(label_masks[b, k].item()) == 0:
                        continue
                    if supervision_mask is not None and int(supervision_mask[b, k].item()) == 0:
                        continue
                    if polarity_labels is not None and int(polarity_labels[b, k].item()) <= 0:
                        continue
                    s = int(span_starts[b, k].item())
                    e = int(span_ends[b, k].item())
                    if 0 <= s < seq_len and 0 <= e < seq_len and e >= s:
                        span_targets[b, s, e] = 1.0
        else:
            # Fallback for old callers only.  This can create false positive
            # cross-pairs in multi-aspect sentences, so the joint model should
            # pass explicit span_starts/span_ends whenever possible.
            span_targets = start_positions.unsqueeze(2).to(dtype=torch.float) * end_positions.unsqueeze(1).to(dtype=torch.float)

        span_targets = span_targets * pair_mask
        return span_targets, pair_mask

    def _span_validity_logits(self, sequence_output):
        """Score every start-end pair so candidate generation can use a real span-validity signal."""
        batch_size, seq_len, _ = sequence_output.size()
        start_extend = sequence_output.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = sequence_output.unsqueeze(1).expand(-1, seq_len, -1, -1)
        span_matrix = torch.cat([start_extend, end_extend], dim=-1)
        return self.span_embedding(span_matrix).squeeze(-1)

    def forward(self, mode, attention_mask, input_ids=None, token_type_ids=None, start_positions=None, end_positions=None,
                span_starts=None, span_ends=None, polarity_labels=None, label_masks=None, candidate_confidences=None,
                sequence_input=None, weight_start=None, weight_end=None, weight_span=None, weight_ac=None,
                weight_pair_verifier=None, use_expectation=None, return_pair_logits=False):
        if mode == 'train':
            assert input_ids is not None and token_type_ids is not None
            all_encoder_layers,_= self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]  #[batch_size,seq_len,hidden_dim]

            assert start_positions is not None and end_positions is not None
            assert span_starts is not None and span_ends is not None
            assert polarity_labels is not None and label_masks is not None
            assert weight_start is not None and weight_end is not None
            assert weight_span is not None and weight_ac is not None
            if weight_pair_verifier is None:
                weight_pair_verifier = 0.0
            assert use_expectation is not None
            # temp=torch.ones(sequence_output_commom.size(0),dtype=torch.int64,device=cuda)
            # seq_len=sequence_output_commom.size(1)*temp  #
            #
            # sequence_output_nerinit, (_, _) = self.ner_bigru(sequence_output_commom, seq_len)  # [batch_size,seq_len,D]
            # sequence_output_absainit, (_, _) = self.absa_bigru(sequence_output_commom,seq_len)  # [batch_size,seq_len,D]
            # layters_GRU  update
            # for i in range(layer_GRU-1):
            #     sequence_output_ner=self.activation_sigmoid(self.same_ner)*sequence_output_nerinit+self.activation_sigmoid(self.absa2ner)*sequence_output_absainit
            #     sequence_output_absa=self.activation_sigmoid(self.same_absa)*sequence_output_absainit+self.activation_sigmoid(self.ner2absa)*sequence_output_nerinit
            #     sequence_output_nerinit, (_, _) = self.ner_bi_GRU(sequence_output_ner, seq_len)
            #     sequence_output_absainit, (_, _) = self.absa_bi_GRU(sequence_output_absa, seq_len)
            batch_size, seq_len, hid_size = sequence_output.size()
            start_logits = self.start_outputs(sequence_output)  # [batch_size , seq_len , 1]
            end_logits = self.end_outputs(sequence_output)  # [batch_size , seq_len , 1]
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)
            raw_start_logits = start_logits
            raw_end_logits = end_logits
            sent_start_logits, sent_end_logits = self._sentiment_boundary_logits(
                sequence_output, raw_start_logits, raw_end_logits)

            # Boundary detection is multi-label: one sentence may contain several
            # aspect starts/ends.  Use BCEWithLogits directly on raw logits instead
            # of Softmax+BCE, otherwise multiple gold boundaries compete with each
            # other and precision collapses.
            start_loss = self._masked_bce_with_logits(raw_start_logits, start_positions.float(), attention_mask)
            end_loss = self._masked_bce_with_logits(raw_end_logits, end_positions.float(), attention_mask)

            if candidate_confidences is None:
                reliable_supervision_mask = label_masks
            else:
                reliable_supervision_mask = label_masks * (candidate_confidences >= 0.999).long()

            sent_boundary_loss = raw_start_logits.sum() * 0.0
            if self.use_sentiment_conditioned_boundary:
                sent_start_targets, sent_end_targets = self._sentiment_boundary_targets(
                    span_starts, span_ends, polarity_labels, label_masks, seq_len,
                    supervision_mask=reliable_supervision_mask)
                sent_start_loss = self._masked_bce_with_logits(sent_start_logits, sent_start_targets, attention_mask)
                sent_end_loss = self._masked_bce_with_logits(sent_end_logits, sent_end_targets, attention_mask)
                sent_boundary_loss = 0.5 * (sent_start_loss + sent_end_loss)

            info_loss = contrastive_loss(
                sequence_output, start_positions, end_positions,
                span_starts=span_starts, span_ends=span_ends,
                polarity_labels=polarity_labels, label_masks=reliable_supervision_mask,
                sentiment_prototypes=self.sentiment_prototypes,
                attention_mask=attention_mask, temperature=0.07)

            span_validity_logits = self._span_validity_logits(sequence_output)
            span_targets, span_pair_mask = self._span_validity_targets(
                start_positions, end_positions, attention_mask,
                max_answer_length=getattr(self, "max_answer_length", None),
                span_starts=span_starts, span_ends=span_ends,
                polarity_labels=polarity_labels, label_masks=label_masks,
                supervision_mask=reliable_supervision_mask)
            positive_count = span_targets.sum().clamp_min(1.0)
            negative_count = (span_pair_mask - span_targets).clamp_min(0.0).sum().clamp_min(1.0)
            pos_weight = (negative_count / positive_count).detach().clamp(max=200.0)
            span_loss_raw = F.binary_cross_entropy_with_logits(
                span_validity_logits, span_targets, pos_weight=pos_weight, reduction='none')
            span_validity_loss = (span_loss_raw * span_pair_mask).sum() / span_pair_mask.sum().clamp_min(1.0)

            # The span-validity loss gets a small fixed coefficient so the span
            # matrix used at inference is actually trained, without introducing
            # another tunable hyperparameter.
            ae_loss = weight_start * start_loss + weight_end * end_loss + \
                      0.5 * (weight_start + weight_end) * sent_boundary_loss + \
                      0.1 * span_validity_loss + weight_span * info_loss

            # Confidence-aware candidate transfer: classify all retained
            # candidates, not only the first gold span.  Gold candidates have
            # confidence 1.0; pseudo candidates are softly weighted by their
            # calibrated boundary confidence.
            span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_output,
                                                             attention_mask)  # [N*M, JR, D], [N*M, JR]
            span_score = self.unary_affine(span_output)
            span_score = span_score.squeeze(-1)
            span_pooled_output = get_self_att_representation(span_output, span_score, span_mask)
            span_pooled_output = self.dense(span_pooled_output)
            span_pooled_output = self.activation_relu(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)
            ac_logits = self.classifier(span_pooled_output)  # [batch_size*M, 5]

            ac_loss_fct = CrossEntropyLoss(reduction='none')
            flat_polarity_labels = flatten(polarity_labels)
            flat_label_masks = flatten(label_masks).to(dtype=ac_logits.dtype)
            if candidate_confidences is None:
                flat_confidences = flat_label_masks
            else:
                flat_confidences = flatten(candidate_confidences).to(dtype=ac_logits.dtype)
            flat_weights = flat_label_masks * flat_confidences
            ac_loss = ac_loss_fct(ac_logits, flat_polarity_labels)
            ac_loss = torch.sum(flat_weights * ac_loss) / flat_weights.sum().clamp_min(1.0)

            pair_verifier_loss = ac_loss * 0.0
            if self.use_pair_verifier and float(weight_pair_verifier) != 0.0:
                batch_size, candidate_num = span_starts.size()
                cls_features = sequence_output[:, 0, :].unsqueeze(1).expand(
                    -1, candidate_num, -1).contiguous().view(-1, hid_size)
                ac_probs = torch.softmax(ac_logits, dim=-1)
                classifier_sentiment_ids = ac_logits.detach().argmax(dim=-1)

                # Use the candidate/gold sentiment when it exists; otherwise use
                # the current classifier proposal for pseudo negative candidates.
                pair_sentiment_ids = torch.where(
                    flat_polarity_labels > 0,
                    flat_polarity_labels,
                    classifier_sentiment_ids,
                )
                if candidate_confidences is None:
                    flat_candidate_confidences = flat_label_masks
                else:
                    flat_candidate_confidences = flatten(candidate_confidences).to(dtype=ac_logits.dtype)

                pair_logits = self._pair_verifier_logits(
                    span_pooled_output,
                    cls_features,
                    pair_sentiment_ids,
                    flat_candidate_confidences,
                    ac_probs,
                )
                pair_targets = (
                    (flat_polarity_labels > 0)
                    & (flat_label_masks > 0)
                    & (flat_candidate_confidences >= 0.999)
                ).to(dtype=pair_logits.dtype)
                pair_masks = ((flat_label_masks > 0) | (flat_candidate_confidences > 0)).to(dtype=pair_logits.dtype)
                positive_count = (pair_targets * pair_masks).sum().clamp_min(1.0)
                negative_count = ((1.0 - pair_targets) * pair_masks).sum().clamp_min(1.0)
                pos_weight = (negative_count / positive_count).detach().clamp(max=200.0)
                pair_loss_raw = F.binary_cross_entropy_with_logits(
                    pair_logits, pair_targets, pos_weight=pos_weight, reduction='none')
                pair_verifier_loss = (pair_loss_raw * pair_masks).sum() / pair_masks.sum().clamp_min(1.0)

            return ae_loss + weight_ac * ac_loss + float(weight_pair_verifier) * pair_verifier_loss

        elif mode == 'extract_inference':
            assert input_ids is not None and token_type_ids is not None
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]  # [batch_size , seq_len , hidden_dim ]

            # temp = torch.ones(sequence_output_common.size(0), dtype=torch.int64, device=cuda)
            # seq_len = sequence_output_common.size(1) * temp
            # sequence_output_nerinit, (_, _) = self.ner_bigru(sequence_output_common, seq_len)  # [batch_size , seq_len , D ]
            # sequence_output_absainit, (_, _) = self.absa_bigru(sequence_output_common, seq_len)
            # for i in range(layer_GRU-1):
            #     sequence_output_ner = self.activation_sigmoid(
            #         self.same_ner) * sequence_output_nerinit + self.activation_sigmoid(
            #         self.absa2ner) * sequence_output_absainit
            #     sequence_output_absa = self.activation_sigmoid(
            #         self.same_absa) * sequence_output_absainit + self.activation_sigmoid(
            #         self.ner2absa) * sequence_output_nerinit
            #     # sequence_output_ner=sequence_output_nerinit
            #     # sequence_output_absa=sequence_output_absainit
            #     sequence_output_nerinit, (_, _)=self.ner_bi_GRU(sequence_output_ner, seq_len)
            #     sequence_output_absainit, (_, _)=self.absa_bi_GRU(sequence_output_absa, seq_len)

            batch_size, seq_len, hid_size = sequence_output.size()
            base_start_logits = self.start_outputs(sequence_output).squeeze(-1)
            base_end_logits = self.end_outputs(sequence_output).squeeze(-1)
            sent_start_logits, sent_end_logits = self._sentiment_boundary_logits(
                sequence_output, base_start_logits, base_end_logits)

            # Candidate annotation receives the ordinary AE logits as its base
            # source and receives sentiment-conditioned logits separately.  Do
            # not replace the base source with max(sentiment logits); otherwise
            # enabling sentiment-conditioned extraction discards the trained AE
            # fallback and makes early inference much less recall-stable.
            start_logits = base_start_logits
            end_logits = base_end_logits

            span_logits = torch.sigmoid(self._span_validity_logits(sequence_output))

            return start_logits, end_logits, span_logits, sequence_output, sent_start_logits, sent_end_logits

        elif mode == 'classify_inference':
            assert span_starts is not None and span_ends is not None and sequence_input is not None
            #span_starts_ture=torch.index_select(span_starts,dim=1,index=torch.tensor([0]).to(cuda))
            #span_ends_ture=torch.index_select(span_ends,dim=1,index=torch.tensor([0]).to(cuda))
            span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_input,
                                                             attention_mask)  # [N*M, JR, D], [N*1, JR]
            span_score = self.unary_affine(span_output)
            span_score = span_score.squeeze(-1)  # [N*M, JR]
            span_pooled_output = get_self_att_representation(span_output, span_score, span_mask)  # [N*M, D]

            span_pooled_output = self.dense(span_pooled_output)
            span_pooled_output = self.activation_relu(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)
            ac_logits = self.classifier(span_pooled_output)  # [N*M, 5]

            reconstructed_ac_logits = reconstruct(ac_logits, span_starts)
            if not (self.use_pair_verifier and return_pair_logits):
                return reconstructed_ac_logits

            batch_size, candidate_num = span_starts.size()
            hidden_size = sequence_input.size(-1)
            cls_features = sequence_input[:, 0, :].unsqueeze(1).expand(
                -1, candidate_num, -1).contiguous().view(-1, hidden_size)
            if candidate_confidences is None:
                flat_candidate_confidences = torch.ones(
                    ac_logits.size(0), dtype=ac_logits.dtype, device=ac_logits.device)
            else:
                flat_candidate_confidences = flatten(candidate_confidences).to(dtype=ac_logits.dtype)
            ac_probs = torch.softmax(ac_logits, dim=-1)
            pair_logits_by_label = []
            for label_id in range(1, ac_logits.size(-1)):
                sentiment_ids = torch.full(
                    (ac_logits.size(0),), label_id, dtype=torch.long, device=ac_logits.device)
                pair_logits_by_label.append(self._pair_verifier_logits(
                    span_pooled_output,
                    cls_features,
                    sentiment_ids,
                    flat_candidate_confidences,
                    ac_probs,
                ))
            pair_logits_by_label = torch.stack(pair_logits_by_label, dim=-1)
            return reconstructed_ac_logits, reconstruct(pair_logits_by_label, span_starts)

def distant_loss(start_logits, end_logits, start_positions=None, end_positions=None, mask=None):
    start_loss = distant_cross_entropy(start_logits, start_positions, mask)
    end_loss = distant_cross_entropy(end_logits, end_positions, mask)
    total_loss = (start_loss + end_loss) / 2
    return total_loss

