
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import json
import argparse

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import bert.tokenization as tokenization
from bert.modeling import BertConfig
from bert.sentiment_modeling import BertForJointSpanExtractAndClassification

from absa.utils import read_absa_data, convert_absa_data, convert_examples_to_features, RawFinalResult, RawSpanResult, span_annotate_candidates, wrapped_get_final_text
from absa.run_base import copy_optimizer_params_to_model, set_optimizer_params_grad, prepare_optimizer, post_process_loss, bert_load_state_dict
from absa.run_cls_span import eval_absa , eval_ac

try:
    import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy
    from xml.sax.saxutils import escape
except:
    sys.exit('Some package is missing... Perhaps <re>?')

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")



def move_batch_to_device(batch, device):
    """Move every tensor in a dataloader batch to the selected device."""
    return tuple(t.to(device) for t in batch)


def build_batch_candidate_inputs(batch_start_logits, batch_end_logits, batch_span_logits,
                                 example_indices, features,
                                 batch_sentiment_start_logits=None, batch_sentiment_end_logits=None):
    """Convert model logits into the list-based format used by candidate annotation."""
    batch_features, batch_results, batch_logits, batch_sentiment_logits = [], [], [], []
    for row_id, example_index in enumerate(example_indices):
        feature = features[int(example_index.item())]
        unique_id = int(feature.unique_id)
        batch_features.append(feature)
        batch_results.append(
            RawSpanResult(
                unique_id=unique_id,
                start_logits=batch_start_logits[row_id].detach().cpu().tolist(),
                end_logits=batch_end_logits[row_id].detach().cpu().tolist(),
            )
        )
        batch_logits.append(batch_span_logits[row_id].detach().cpu().tolist())
        if batch_sentiment_start_logits is not None and batch_sentiment_end_logits is not None:
            batch_sentiment_logits.append((
                batch_sentiment_start_logits[row_id].detach().cpu().tolist(),
                batch_sentiment_end_logits[row_id].detach().cpu().tolist(),
            ))
    if batch_sentiment_start_logits is None or batch_sentiment_end_logits is None:
        batch_sentiment_logits = None
    return batch_features, batch_results, batch_logits, batch_sentiment_logits


def tensorize_candidates(span_starts, span_ends, labels, label_masks, confidences, device):
    """Convert candidate lists returned by span_annotate_candidates to tensors."""
    return (
        torch.tensor(span_starts, dtype=torch.long, device=device),
        torch.tensor(span_ends, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
        torch.tensor(label_masks, dtype=torch.long, device=device),
        torch.tensor(confidences, dtype=torch.float, device=device),
    )


def get_eval_candidate_pool_size(args):
    """Return the candidate pool size used before final decoding.

    Consistency decoding needs a wider span pool than the final output size;
    otherwise reranking has almost no effect.  The final number of retained
    predictions is still controlled by ``n_best_size`` unless
    ``consistency_final_top_k`` is explicitly set.
    """
    n_best_size = max(1, int(args.n_best_size))
    if not getattr(args, "use_consistency_decoding", False):
        return n_best_size
    multiplier = max(1, int(getattr(args, "consistency_pool_multiplier", 2)))
    return max(n_best_size, n_best_size * multiplier)


def apply_sentiment_prior_to_logits(args, batch_ac_logits, candidate_sentiments_tensor, confidences_tensor):
    """Backward-compatible logit prior used by the original decoding path."""
    if not (args.use_sentiment_conditioned_boundary and args.sentiment_prior_weight != 0):
        return batch_ac_logits
    prior = torch.zeros_like(batch_ac_logits)
    valid_prior = candidate_sentiments_tensor > 0
    safe_sentiments = candidate_sentiments_tensor.clamp(min=0, max=batch_ac_logits.size(-1) - 1)
    prior.scatter_(2, safe_sentiments.unsqueeze(-1),
                   (confidences_tensor * float(args.sentiment_prior_weight)).unsqueeze(-1))
    return batch_ac_logits + prior * valid_prior.unsqueeze(-1).to(dtype=batch_ac_logits.dtype)


def _span_overlap_ratio(start_a, end_a, start_b, end_b):
    """Return the overlap ratio between two inclusive token spans.

    The denominator is the shorter span length, not the union length.  This is
    intentionally stricter than IoU for ABSA decoding: when a short erroneous
    boundary variant is almost contained in a higher-scoring aspect span, it is
    treated as a duplicate and suppressed.
    """
    start_a, end_a, start_b, end_b = int(start_a), int(end_a), int(start_b), int(end_b)
    inter = max(0, min(end_a, end_b) - max(start_a, start_b) + 1)
    len_a = max(1, end_a - start_a + 1)
    len_b = max(1, end_b - start_b + 1)
    return inter / float(min(len_a, len_b))


def _normalize_span_text(text):
    return " ".join(str(text).lower().strip().split())


def _get_span_text_key(example, feature, start_index, end_index, do_lower_case, verbose_logging, logger):
    """Build a duplicate-suppression key for a candidate aspect span."""
    try:
        if (example is not None and feature is not None and
                int(start_index) in feature.token_to_orig_map and
                int(end_index) in feature.token_to_orig_map):
            final_text = wrapped_get_final_text(example, feature, int(start_index), int(end_index),
                                                do_lower_case, verbose_logging, logger)
            return _normalize_span_text(final_text)
    except Exception:
        # Decoding should not crash because of a logging/text-recovery edge case.
        pass
    return "{}:{}".format(int(start_index), int(end_index))


def consistency_decode_instance(args, ac_logits, span_starts, span_ends, span_masks,
                                candidate_sentiments, candidate_confidences,
                                pair_keep_logits=None,
                                example=None, feature=None, do_lower_case=True,
                                verbose_logging=False, logger=None):
    """Confidence-aware Aspect-Sentiment Consistency Decoding (CA-SCD).

    Each candidate span is decoded as an aspect-sentiment pair rather than as
    an isolated aspect followed by post-hoc sentiment filtering.  For every
    non-other sentiment y, the pair score is

        log q(a) + log P(y | a, x) - log P(other | a, x)
        + lambda * I[y == y_boundary] * q(a)
        + alpha * log V(a, y),

    where V(a, y) is the pair verifier's keep probability.  If the verifier is
    disabled or unavailable, the last term is omitted.  q(a) is the calibrated boundary confidence returned by candidate
    generation.  ``sentiment_prior_weight`` is reused as lambda so that the
    method does not introduce another free coefficient.

    After pair scoring, a pair-level duplicate suppression step is applied:
      1) identical recovered aspect text keeps only the highest-score pair;
      2) token spans with overlap ratio > ``pair_nms_overlap`` keep only the
         highest-score pair;
      3) if the same aspect receives multiple sentiments, the earlier
         highest-score aspect-sentiment pair is retained.
    """
    eps = 1e-8
    ac_probs = torch.softmax(ac_logits, dim=-1)
    num_labels = ac_probs.size(-1)
    if num_labels <= 1:
        raise ValueError("AC classifier must have at least one non-other label.")

    span_valid = span_masks.to(dtype=torch.bool)
    boundary_conf = candidate_confidences.clamp(min=eps, max=1.0)
    other_prob = ac_probs[:, 0].clamp(min=eps)
    non_other_probs = ac_probs[:, 1:].clamp(min=eps)

    boundary_score = torch.log(boundary_conf).unsqueeze(1)
    sentiment_log_odds = torch.log(non_other_probs) - torch.log(other_prob).unsqueeze(1)
    pair_scores = boundary_score + sentiment_log_odds

    if args.use_sentiment_conditioned_boundary and args.sentiment_prior_weight != 0:
        label_ids = torch.arange(1, num_labels, device=ac_logits.device).unsqueeze(0)
        prior_labels = candidate_sentiments.clamp(min=0, max=num_labels - 1).unsqueeze(1)
        agreement = (prior_labels == label_ids) & (prior_labels > 0)
        pair_scores = pair_scores + agreement.to(dtype=pair_scores.dtype) * (
            float(args.sentiment_prior_weight) * boundary_conf.unsqueeze(1)
        )

    best_pair_score, best_non_other_index = pair_scores.max(dim=1)
    cls_pred_tensor = best_non_other_index + 1
    final_class_conf = ac_probs.gather(1, cls_pred_tensor.unsqueeze(1)).squeeze(1)

    selected_pair_keep_prob = None
    use_pair_verifier = bool(getattr(args, "use_pair_verifier", False))
    if use_pair_verifier and pair_keep_logits is not None:
        pair_keep_probs = torch.sigmoid(pair_keep_logits).clamp(min=eps, max=1.0)
        selected_pair_keep_prob = pair_keep_probs.gather(
            1, best_non_other_index.unsqueeze(1)).squeeze(1)
        alpha = float(getattr(args, "pair_verifier_alpha", 1.0))
        best_pair_score = best_pair_score + alpha * torch.log(selected_pair_keep_prob)

    if args.drop_other_predictions:
        if args.use_ac_confidence_filter:
            valid = (
                span_valid
                & (final_class_conf >= float(args.ac_min_confidence))
                & ((final_class_conf - other_prob) >= float(args.ac_other_margin))
            )
        else:
            valid = span_valid & (final_class_conf > other_prob)
    else:
        valid = span_valid

    min_keep_prob = float(getattr(args, "pair_verifier_min_keep_prob", 0.0))
    if use_pair_verifier and selected_pair_keep_prob is not None and min_keep_prob > 0.0:
        valid = valid & (selected_pair_keep_prob >= min_keep_prob)

    final_top_k = int(getattr(args, "consistency_final_top_k", 0))
    if final_top_k <= 0:
        final_top_k = int(args.n_best_size)
    final_top_k = max(1, min(final_top_k, span_starts.size(0)))

    order = torch.argsort(best_pair_score, descending=True)
    keep = torch.zeros_like(valid, dtype=torch.bool)
    kept_text_keys = set()
    kept_spans = []
    kept = 0
    use_pair_nms = not bool(getattr(args, "disable_pair_nms", False))
    overlap_threshold = float(getattr(args, "pair_nms_overlap", 0.5))

    for idx in order.tolist():
        if not bool(valid[idx]):
            continue

        start_i = int(span_starts[idx].item())
        end_i = int(span_ends[idx].item())
        text_key = _get_span_text_key(example, feature, start_i, end_i,
                                      do_lower_case, verbose_logging, logger)

        if use_pair_nms:
            if text_key in kept_text_keys:
                continue

            redundant = False
            for kept_start, kept_end, _ in kept_spans:
                if _span_overlap_ratio(start_i, end_i, kept_start, kept_end) > overlap_threshold:
                    redundant = True
                    break
            if redundant:
                continue

        keep[idx] = True
        kept_text_keys.add(text_key)
        kept_spans.append((start_i, end_i, int(cls_pred_tensor[idx].item())))
        kept += 1
        if kept >= final_top_k:
            break

    start_indexes = span_starts[order].detach().cpu().tolist()
    end_indexes = span_ends[order].detach().cpu().tolist()
    cls_pred = cls_pred_tensor[order].detach().cpu().tolist()
    span_masks_out = keep[order].to(dtype=torch.long).detach().cpu().tolist()
    return start_indexes, end_indexes, cls_pred, span_masks_out


def should_use_expectation(args, global_step):
    """Decide whether this step uses predicted transfer knowledge.

    ``random_train`` is kept for backward compatibility: it is the probability
    of using the gold-label branch. The complementary probability uses the
    expectation branch. ``expectation_start_step`` can delay pseudo-transfer
    until the extractor is less noisy.
    """
    if getattr(args, "disable_expectation", False):
        return False
    if global_step < getattr(args, "expectation_start_step", 0):
        return False
    gold_branch_probability = float(args.random_train)
    return np.random.rand() >= gold_branch_probability

def read_train_data(args, tokenizer, logger):
    train_path = os.path.join(args.data_dir, args.train_file)
    train_set = read_absa_data(train_path)
    train_examples = convert_absa_data(dataset=train_set, verbose_logging=args.verbose_logging)
    train_features = convert_examples_to_features(train_examples, tokenizer, args.max_seq_length,
                                                  args.verbose_logging, logger)

    num_train_steps = int(
        len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    logger.info("Num orig examples = %d", len(train_examples))
    logger.info("Num split features = %d", len(train_features))
    logger.info("Batch size = %d", args.train_batch_size)
    logger.info("Num steps = %d", num_train_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_positions for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_positions for f in train_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions, all_example_index)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=8, pin_memory=True)
    return train_examples, train_features, train_dataloader, num_train_steps

def read_eval_data(args, tokenizer, logger, eval_file=None):
    eval_name = eval_file if eval_file is not None else args.predict_file
    eval_path = os.path.join(args.data_dir, eval_name)
    eval_set = read_absa_data(eval_path)
    eval_examples = convert_absa_data(dataset=eval_set, verbose_logging=args.verbose_logging)

    eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length,
                                                 args.verbose_logging, logger)

    logger.info("Num orig examples = %d", len(eval_examples))
    logger.info("Num split features = %d", len(eval_features))
    logger.info("Batch size = %d", args.predict_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_span_starts = torch.tensor([f.start_indexes for f in eval_features],dtype=torch.long)
    all_span_ends = torch.tensor([f.end_indexes for f in eval_features],dtype=torch.long)
    all_label_masks = torch.tensor([f.label_masks for f in eval_features],dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_span_starts,
                              all_span_ends,all_label_masks,all_example_index)
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)
    return eval_examples, eval_features, eval_dataloader

def run_train_epoch(args, global_step, model, param_optimizer,
                    train_examples, train_features, train_dataloader,
                    eval_examples, eval_features, eval_dataloader,
                    optimizer, n_gpu, device, logger, log_path, save_path,
                    save_checkpoints_steps, start_save_steps, best_f1):
    running_loss, count = 0.0, 0
    for step, batch in enumerate(train_dataloader):
        batch = move_batch_to_device(batch, device)
        input_ids, input_mask, segment_ids, start_positions, end_positions, example_indices = batch

        # Candidate generation is a non-differentiable teacher signal for the
        # classification branch.  Do it without autograd and with dropout disabled;
        # the actual trainable forward pass below still keeps gradients.
        was_training = model.training
        model.eval()
        with torch.no_grad():
            extract_outputs = model('extract_inference', input_mask, input_ids=input_ids, token_type_ids=segment_ids)
        if was_training:
            model.train()
        batch_start_logits, batch_end_logits, batch_span_logits, _, batch_sent_start_logits, batch_sent_end_logits = extract_outputs

        batch_features, batch_results, batch_logits, batch_sentiment_logits = build_batch_candidate_inputs(
            batch_start_logits, batch_end_logits, batch_span_logits, example_indices, train_features,
            batch_sent_start_logits, batch_sent_end_logits)

        use_expectation = should_use_expectation(args, global_step)
        span_starts, span_ends, labels, label_masks, confidences = span_annotate_candidates(
            train_examples, batch_features, batch_results,
            args.filter_type, True, args.use_heuristics, args.use_nms,
            args.logit_threshold, args.n_best_size, args.max_answer_length,
            args.do_lower_case, args.verbose_logging, logger, batch_logits,
            return_confidence=True,
            soft_candidate_transfer=use_expectation,
            candidate_threshold_mode=args.candidate_threshold_mode,
            confidence_temperature=args.confidence_temperature,
            min_candidate_confidence=args.min_candidate_confidence,
            dynamic_threshold_scale=args.dynamic_threshold_scale,
            candidate_min_keep=args.candidate_min_keep,
            batch_sentiment_logits=batch_sentiment_logits if args.use_sentiment_conditioned_boundary else None,
            pseudo_confidence_scale=args.pseudo_confidence_scale,
            train_pseudo_other=args.train_pseudo_other)

        span_starts, span_ends, labels, label_masks, confidences = tensorize_candidates(
            span_starts, span_ends, labels, label_masks, confidences, device)

        loss = model(
            'train', input_mask,
            input_ids=input_ids,
            token_type_ids=segment_ids,
            start_positions=start_positions,
            end_positions=end_positions,
            span_starts=span_starts,
            span_ends=span_ends,
            polarity_labels=labels,
            label_masks=label_masks,
            candidate_confidences=confidences,
            weight_start=args.weight_start,
            weight_end=args.weight_end,
            weight_span=args.weight_span,
            weight_ac=args.weight_ac,
            weight_pair_verifier=args.weight_pair_verifier,
            use_expectation=use_expectation,
        )
        loss = post_process_loss(args, n_gpu, loss)
        loss.backward()
        running_loss += loss.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16 or args.optimize_on_cpu:
                if args.fp16 and args.loss_scale != 1.0:
                    # scale down gradients for fp16 training
                    for param in model.parameters():
                        param.grad.data = param.grad.data / args.loss_scale
                is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                if is_nan:
                    logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                    args.loss_scale = args.loss_scale / 2
                    model.zero_grad()
                    continue
                optimizer.step()
                copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
            else:
                optimizer.step()
            model.zero_grad()
            global_step += 1
            count += 1

            if global_step % save_checkpoints_steps == 0 and count != 0:
                logger.info("step: {}, loss: {:.4f}".format(global_step, running_loss / count))

            if global_step % save_checkpoints_steps == 0 and global_step > start_save_steps and count != 0:  # eval & save model
                # if global_step % save_checkpoints_steps == 0 and count != 0:
                logger.info("***** Running evaluation *****")
                model.eval()
                metrics = evaluate(args, model, device, eval_examples, eval_features, eval_dataloader, logger)
                print("P_all: {:.4f}, R_all: {:.4f}, F1_all: {:.4f}, AvgPred: {:.2f}".format(metrics['p_all'], metrics['r_all'],
                                                                            metrics['f1_all'], metrics.get('avg_pred_per_sentence', 0.0)))
                print('>' * 30)

                f = open(log_path, "a")
                print("step: {}, loss: {:.4f}, "
                      "P_all: {:.4f}, R_all: {:.4f}, F1_all: {:.4f}, AvgPred: {:.2f} "
                      .format(global_step, running_loss / count,
                              metrics['p_all'], metrics['r_all'], metrics['f1_all'],
                              metrics.get('avg_pred_per_sentence', 0.0)), file=f)
                print(" ", file=f)
                f.close()
                running_loss, count = 0.0, 0
                model.train()
                if metrics['f1_all'] > best_f1:
                    best_f1 = metrics['f1_all']
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': global_step
                    }, save_path)
                if args.debug:
                    break
    return global_step, model, best_f1

def evaluate(args, model, device, eval_examples, eval_features, eval_dataloader, logger, write_pred=False):
    all_results = []
    eval_candidate_pool_size = get_eval_candidate_pool_size(args)
    for batch in eval_dataloader:
        batch = move_batch_to_device(batch, device)
        input_ids, input_mask, segment_ids, _, _, label_masks, example_indices = batch
        with torch.no_grad():
            extract_outputs = model('extract_inference', input_mask, input_ids=input_ids, token_type_ids=segment_ids)
            batch_start_logits, batch_end_logits, batch_span_logits, sequence_output, batch_sent_start_logits, batch_sent_end_logits = extract_outputs

        batch_features, batch_results, batch_logits, batch_sentiment_logits = build_batch_candidate_inputs(
            batch_start_logits, batch_end_logits, batch_span_logits, example_indices, eval_features,
            batch_sent_start_logits, batch_sent_end_logits)

        span_starts, span_ends, candidate_sentiments, label_masks, confidences = span_annotate_candidates(
            eval_examples, batch_features, batch_results,
            args.filter_type, False, args.use_heuristics, args.use_nms,
            args.logit_threshold, eval_candidate_pool_size, args.max_answer_length,
            args.do_lower_case, args.verbose_logging, logger, batch_logits,
            return_confidence=True,
            soft_candidate_transfer=False,
            candidate_threshold_mode=args.candidate_threshold_mode,
            confidence_temperature=args.confidence_temperature,
            min_candidate_confidence=args.min_candidate_confidence,
            dynamic_threshold_scale=args.dynamic_threshold_scale,
            candidate_min_keep=args.candidate_min_keep,
            batch_sentiment_logits=batch_sentiment_logits if args.use_sentiment_conditioned_boundary else None)

        span_starts = torch.tensor(span_starts, dtype=torch.long, device=device)
        span_ends = torch.tensor(span_ends, dtype=torch.long, device=device)
        candidate_sentiments_tensor = torch.tensor(candidate_sentiments, dtype=torch.long, device=device)
        label_masks_tensor = torch.tensor(label_masks, dtype=torch.long, device=device)
        confidences_tensor = torch.tensor(confidences, dtype=torch.float, device=device)
        sequence_output = sequence_output.to(device)
        with torch.no_grad():
            classify_outputs = model('classify_inference', input_mask, span_starts=span_starts,
                                     span_ends=span_ends, sequence_input=sequence_output,
                                     candidate_confidences=confidences_tensor,
                                     return_pair_logits=args.use_pair_verifier)    # [N, M, 5]
        if isinstance(classify_outputs, tuple):
            batch_ac_logits, batch_pair_keep_logits = classify_outputs
        else:
            batch_ac_logits, batch_pair_keep_logits = classify_outputs, None

        if not args.use_consistency_decoding:
            batch_ac_logits = apply_sentiment_prior_to_logits(
                args, batch_ac_logits, candidate_sentiments_tensor, confidences_tensor)

        for j, example_index in enumerate(example_indices):
            if args.use_consistency_decoding:
                eval_feature = eval_features[example_index.item()]
                eval_example = eval_examples[eval_feature.example_index]
                start_indexes, end_indexes, cls_pred, span_masks = consistency_decode_instance(
                    args,
                    batch_ac_logits[j],
                    span_starts[j],
                    span_ends[j],
                    label_masks_tensor[j],
                    candidate_sentiments_tensor[j],
                    confidences_tensor[j],
                    pair_keep_logits=batch_pair_keep_logits[j] if batch_pair_keep_logits is not None else None,
                    example=eval_example,
                    feature=eval_feature,
                    do_lower_case=args.do_lower_case,
                    verbose_logging=args.verbose_logging,
                    logger=logger,
                )
            else:
                # AC confidence filtering is intentionally applied after the optional
                # sentiment-prior adjustment, because these are the logits used for
                # the final sentiment decision.
                ac_probs = torch.softmax(batch_ac_logits[j], dim=-1)  # [M, num_labels]
                classifier_pred_tensor = ac_probs.argmax(dim=1)
                classifier_pred = classifier_pred_tensor.detach().cpu().tolist()
                classifier_conf = ac_probs.gather(1, classifier_pred_tensor.unsqueeze(1)).squeeze(1)
                other_conf = ac_probs[:, 0]

                candidate_sent_pred = candidate_sentiments_tensor[j].detach().cpu().tolist()
                if args.direct_sentiment_boundary_output:
                    cls_pred = [cand if cand > 0 else clf for cand, clf in zip(candidate_sent_pred, classifier_pred)]
                else:
                    cls_pred = classifier_pred

                # Confidence of the final selected class. In normal mode this is the
                # classifier max probability. In direct sentiment-boundary mode, the
                # final class can be supplied by the boundary candidate, so gather its
                # AC probability explicitly.
                cls_pred_tensor = torch.tensor(cls_pred, dtype=torch.long, device=ac_probs.device).clamp(
                    min=0, max=ac_probs.size(-1) - 1)
                final_class_conf = ac_probs.gather(1, cls_pred_tensor.unsqueeze(1)).squeeze(1)
                final_class_conf_list = final_class_conf.detach().cpu().tolist()
                other_conf_list = other_conf.detach().cpu().tolist()

                start_indexes = span_starts[j].detach().cpu().tolist()
                end_indexes = span_ends[j].detach().cpu().tolist()
                span_masks_base = label_masks_tensor[j].detach().cpu().tolist()
                if args.drop_other_predictions:
                    if args.use_ac_confidence_filter:
                        span_masks = [
                            int(
                                mask and pred != 0
                                and conf >= args.ac_min_confidence
                                and (conf - other) >= args.ac_other_margin
                            )
                            for mask, pred, conf, other in zip(span_masks_base, cls_pred, final_class_conf_list, other_conf_list)
                        ]
                    else:
                        span_masks = [int(mask and pred != 0) for mask, pred in zip(span_masks_base, cls_pred)]
                else:
                    span_masks = span_masks_base

            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawFinalResult(unique_id=unique_id, start_indexes=start_indexes,
                                              end_indexes=end_indexes, cls_pred=cls_pred, span_masks=span_masks))

    metrics, all_nbest_json = eval_absa(eval_examples, eval_features, all_results,
                                        args.do_lower_case, args.verbose_logging, logger)
    valid_candidates = sum(sum(int(mask) for mask in result.span_masks) for result in all_results)
    metrics['avg_pred_per_sentence'] = valid_candidates / max(len(all_results), 1)
    logger.info("Average valid predicted candidates per sentence: %.4f", metrics['avg_pred_per_sentence'])
    if write_pred:
        output_file = os.path.join(args.output_dir, "predictions.json")
        with open(output_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        logger.info("Writing predictions to: %s" % (output_file))
    return metrics

def evaluate_ac(args, model, device, eval_examples, eval_features, eval_dataloader, logger, write_pred=False):
    all_results = []
    for batch in eval_dataloader:
        batch = move_batch_to_device(batch, device)
        input_ids, input_mask, segment_ids, span_starts, span_ends, label_masks, example_indices = batch
        with torch.no_grad():
            extract_outputs = model('extract_inference', input_mask, input_ids=input_ids, token_type_ids=segment_ids)
            sequence_output = extract_outputs[3]

        with torch.no_grad():
            batch_ac_logits = model('classify_inference', input_mask, span_starts=span_starts,
                                    span_ends=span_ends, sequence_input=sequence_output)    # [N, M, 4]

        for j, example_index in enumerate(example_indices):
            cls_pred = batch_ac_logits[j].detach().cpu().numpy().argmax(axis=1).tolist()
            start_indexes = span_starts[j].detach().cpu().tolist()
            end_indexes = span_ends[j].detach().cpu().tolist()
            span_masks = label_masks[j]
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawFinalResult(unique_id=unique_id, start_indexes=start_indexes,
                                              end_indexes=end_indexes, cls_pred=cls_pred, span_masks=span_masks))

    metrics, all_nbest_json = eval_ac(eval_examples, eval_features, all_results,
                                        args.do_lower_case, args.verbose_logging, logger)
    if write_pred:
        output_file = os.path.join(args.output_dir, "predictions.json")
        with open(output_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        logger.info("Writing predictions to: %s" % (output_file))
    return metrics



def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # this machine exposes only GPU 0
    torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser()
    ## ----------------------------------init-------------------------------
    parser.add_argument("--num_train_epochs", default=100, type=float)
    parser.add_argument("--weight_start", default=1, type=float)
    parser.add_argument("--weight_end", default=1, type=float)
    #parser.add_argument("--use_expectation", default=True, action='store_true')
    #parser.add_argument("--shared_weight", default=None, type=int,required=True)
    #parser.add_argument("--layer_GRU", default=None, type=int, required=True)
    parser.add_argument("--train_file", default='split_laptop14_train.txt', type=str)
    parser.add_argument("--predict_file", default='laptop14_test.txt' , type=str)
    parser.add_argument("--eval_file", default=None, type=str)
    parser.add_argument("--logit_threshold", default=7.5, type=float, help="Boundary logit threshold used only when candidate_threshold_mode=fixed.")
    parser.add_argument("--candidate_threshold_mode", default="fixed", choices=["fixed", "dynamic", "none"],
                        help="Candidate pruning mode. dynamic uses an instance-wise calibrated threshold; none keeps top-K only.")
    parser.add_argument("--confidence_temperature", default=1.0, type=float,
                        help="Temperature for converting span scores into candidate confidences.")
    parser.add_argument("--min_candidate_confidence", default=0.50, type=float,
                        help="Optional lower bound for retained predicted candidate confidence.")
    parser.add_argument("--dynamic_threshold_scale", default=0.0, type=float,
                        help="Mean + scale*std threshold for dynamic candidate calibration. Default 0.0 is less recall-hostile than +0.5.")
    parser.add_argument("--candidate_min_keep", default=1, type=int,
                        help="Minimum number of predicted candidates kept before confidence filtering can prune. Dynamic mode enforces at least 1.")
    parser.add_argument("--use_sentiment_conditioned_boundary", default=False, type=str2bool, nargs="?", const=True,
                        help="Use neutral/positive/negative prototypes to generate sentiment-specific boundary candidates.")
    parser.add_argument("--sentiment_prior_weight", default=0.25, type=float,
                        help="How strongly sentiment-specific candidate confidence biases final sentiment logits.")
    parser.add_argument("--direct_sentiment_boundary_output", default=False, type=str2bool, nargs="?", const=True,
                        help="Use candidate sentiment as final sentiment when sentiment-conditioned extraction is enabled.")
    parser.add_argument("--drop_other_predictions", default=True, type=str2bool, nargs="?", const=True,
                        help="Drop candidate spans whose final sentiment class is other.")
    parser.add_argument("--use_ac_confidence_filter", default=False, type=str2bool, nargs="?", const=True,
                        help="Drop non-other candidates unless the AC classifier is confident enough.")
    parser.add_argument("--ac_min_confidence", default=0.45, type=float,
                        help="Minimum AC probability required for the final non-other class when AC confidence filtering is enabled.")
    parser.add_argument("--ac_other_margin", default=0.05, type=float,
                        help="Minimum probability margin between the final non-other class and the other class when AC confidence filtering is enabled.")
    parser.add_argument("--use_consistency_decoding", default=False, type=str2bool, nargs="?", const=True,
                        help="Use confidence-aware aspect-sentiment consistency decoding (CA-SCD) for final predictions.")
    parser.add_argument("--consistency_pool_multiplier", default=2, type=int,
                        help="Candidate pool multiplier used before CA-SCD. Final output is still limited by n_best_size unless consistency_final_top_k is set.")
    parser.add_argument("--consistency_final_top_k", default=0, type=int,
                        help="Maximum candidates kept after CA-SCD. 0 means use n_best_size.")
    parser.add_argument("--pair_nms_overlap", default=0.5, type=float,
                        help="Pair-level duplicate suppression threshold after CA-SCD."
                             " Overlap is measured over the shorter inclusive token span.")
    parser.add_argument("--disable_pair_nms", default=False, action="store_true",
                        help="Disable pair-level duplicate suppression after CA-SCD for ablation.")
    parser.add_argument("--use_pair_verifier", default=True, type=str2bool, nargs="?", const=True,
                        help="Use the pair-level consistency verifier for training and CA-SCD reranking.")
    parser.add_argument("--weight_pair_verifier", default=0.2, type=float,
                        help="Training loss weight for pair-level consistency verification.")
    parser.add_argument("--pair_verifier_alpha", default=1.0, type=float,
                        help="Reranking coefficient for alpha * log(pair_keep_score).")
    parser.add_argument("--pair_verifier_min_keep_prob", default=0.0, type=float,
                        help="Optional minimum pair verifier keep probability. 0 disables this hard filter.")
    parser.add_argument("--random_train", default=0.9, type=float, help="Probability of using the gold-label branch during joint training.")
    parser.add_argument("--expectation_start_step", default=0, type=int, help="Delay confidence-aware pseudo-candidate transfer until this global step.")
    parser.add_argument("--disable_expectation", default=False, action="store_true", help="Always use only gold candidate supervision.")
    parser.add_argument("--pseudo_confidence_scale", default=0.30, type=float,
                        help="Loss weight multiplier for weak pseudo sentiment labels in the expectation branch.")
    parser.add_argument("--train_pseudo_other", default=False, type=str2bool, nargs="?", const=True,
                        help="Ablation switch. If true, unmatched pseudo candidates are trained as weak other; default false avoids pseudo-other noise.")
    parser.add_argument("--weight_span", default=1e-7, type=float)
    parser.add_argument("--output_dir", default="out/Res/default", type=str)
    parser.add_argument("--weight_ac", default=1, type=float)
    parser.add_argument("--bert_config_file", default='bert-large-uncased/bert_config.json', type=str,
                        help="The config json file corresponding to the pre-trained BERT model. "
                             "This specifies the model architecture.")
    parser.add_argument("--vocab_file", default='bert-large-uncased/vocab.txt', type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--debug", default=False, action='store_true', help="Whether to run in debug mode.")
    parser.add_argument("--data_dir", default='data/absa', type=str, help="SemEval data dir")
    parser.add_argument("--init_checkpoint", default='bert-large-uncased/pytorch_model.bin', type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Whether to lower case the input text. Should be True for uncased "
                             "models and False for cased models.")
    parser.add_argument("--max_seq_length", default=85, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_train", default=True, type=str2bool, nargs="?", const=True,
                        help="Whether to run training. Use --do_train False to skip training.")
    parser.add_argument("--do_predict", default=True, type=str2bool, nargs="?", const=True,
                        help="Whether to run prediction. Use --do_predict False to skip prediction.")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=16, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--save_proportion", default=0.7, type=float,
                        help="Proportion of steps to save models for. E.g., 0.5 = 50%% of training.")
    parser.add_argument("--n_best_size", default=5, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=12, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--filter_type", default="f1", type=str, help="Which filter type to use")
    parser.add_argument("--use_heuristics", default=True, action='store_true',
                        help="If true, use heuristic regularization on span length")
    parser.add_argument("--use_nms", default=True, action='store_true',
                        help="If true, use nms to prune redundant spans")
    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    args = parser.parse_args()

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")
    if not 0.0 <= args.random_train <= 1.0:
        raise ValueError("random_train must be in [0, 1]; got {}".format(args.random_train))
    if args.confidence_temperature <= 0:
        raise ValueError("confidence_temperature must be > 0; got {}".format(args.confidence_temperature))
    if not 0.0 <= args.min_candidate_confidence <= 1.0:
        raise ValueError("min_candidate_confidence must be in [0, 1]; got {}".format(args.min_candidate_confidence))
    if not 0.0 <= args.pseudo_confidence_scale <= 1.0:
        raise ValueError("pseudo_confidence_scale must be in [0, 1]; got {}".format(args.pseudo_confidence_scale))


    if args.consistency_pool_multiplier < 1:
        raise ValueError("consistency_pool_multiplier must be >= 1; got {}".format(args.consistency_pool_multiplier))
    if args.consistency_final_top_k < 0:
        raise ValueError("consistency_final_top_k must be >= 0; got {}".format(args.consistency_final_top_k))
    if not 0.0 <= args.pair_nms_overlap <= 1.0:
        raise ValueError("pair_nms_overlap must be in [0, 1]; got {}".format(args.pair_nms_overlap))
    if args.weight_pair_verifier < 0.0:
        raise ValueError("weight_pair_verifier must be >= 0; got {}".format(args.weight_pair_verifier))
    if args.pair_verifier_alpha < 0.0:
        raise ValueError("pair_verifier_alpha must be >= 0; got {}".format(args.pair_verifier_alpha))
    if not 0.0 <= args.pair_verifier_min_keep_prob <= 1.0:
        raise ValueError("pair_verifier_min_keep_prob must be in [0, 1]; got {}".format(args.pair_verifier_min_keep_prob))


    if args.do_train and not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict and not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("torch_version: {} device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        torch.__version__, device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_seq_length, bert_config.max_position_embeddings))

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger.info('output_dir: {}'.format(args.output_dir))
    save_path = os.path.join(args.output_dir, 'checkpoint.pth.tar')
    log_path = os.path.join(args.output_dir, 'performance.txt')
    network_path = os.path.join(args.output_dir, 'network.txt')
    parameter_path = os.path.join(args.output_dir, 'parameter.txt')

    f = open(parameter_path, "w")
    for arg in sorted(vars(args)):
        print("{}: {}".format(arg, getattr(args, arg)), file=f)
    f.close()

    logger.info("***** Preparing model *****")
    model = BertForJointSpanExtractAndClassification(bert_config,args)
    if args.init_checkpoint is not None and not os.path.isfile(save_path):
        model = bert_load_state_dict(model, torch.load(args.init_checkpoint, map_location='cpu'))
        logger.info("Loading model from pretrained checkpoint: {}".format(args.init_checkpoint))

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model'])
        step = checkpoint['step']
        logger.info("Loading model from finetuned checkpoint: '{}' (step {})"
                    .format(save_path, step))

    f = open(network_path, "w")
    for n, param in model.named_parameters():
        print("name: {}, size: {}, dtype: {}, requires_grad: {}"
              .format(n, param.size(), param.dtype, param.requires_grad), file=f)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total trainable parameters: {}".format(total_trainable_params), file=f)
    print("Total parameters: {}".format(total_params), file=f)
    f.close()

    logger.info("***** Preparing data *****")
    train_examples, train_features, train_dataloader, num_train_steps = None, None, None, None
    eval_examples, eval_features, eval_dataloader = None, None, None
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    if args.do_train:
        logger.info("***** Preparing training *****")
        train_examples, train_features, train_dataloader, num_train_steps = read_train_data(args, tokenizer, logger)
        logger.info("***** Preparing evaluation *****")
        checkpoint_eval_file = args.eval_file if args.eval_file is not None else args.predict_file
        eval_examples, eval_features, eval_dataloader = read_eval_data(args, tokenizer, logger, checkpoint_eval_file)

    logger.info("***** Preparing optimizer *****")
    optimizer, param_optimizer = prepare_optimizer(args, model, num_train_steps)

    global_step = 0
    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']
        logger.info("Loading optimizer from finetuned checkpoint: '{}' (step {})".format(save_path, step))
        global_step = step

    if args.do_train:
        logger.info("***** Running training *****")
        best_f1 = 0
        save_checkpoints_steps = max(1, int(num_train_steps / (5 * args.num_train_epochs)))
        start_save_steps = int(num_train_steps * args.save_proportion)
        if args.debug:
            args.num_train_epochs = 1
            save_checkpoints_steps = 20
            start_save_steps = 0
        model.train()
        for epoch in range(int(args.num_train_epochs)):
            logger.info("***** Epoch: {} *****".format(epoch+1))
            global_step, model, best_f1 = run_train_epoch(args, global_step, model, param_optimizer,
                                                          train_examples, train_features, train_dataloader,
                                                          eval_examples, eval_features, eval_dataloader,
                                                          optimizer, n_gpu, device, logger, log_path, save_path,
                                                          save_checkpoints_steps, start_save_steps, best_f1)

    if args.do_predict:
        logger.info("***** Running prediction *****")
        eval_examples, eval_features, eval_dataloader = read_eval_data(args, tokenizer, logger, args.predict_file)

        # restore from best checkpoint
        if save_path and os.path.isfile(save_path) and args.do_train:
            checkpoint = torch.load(save_path)
            model.load_state_dict(checkpoint['model'])
            step = checkpoint['step']
            logger.info("Loading model from finetuned checkpoint: '{}' (step {})"
                        .format(save_path, step))

        model.eval()
        metrics_0 = evaluate(args, model, device, eval_examples, eval_features, eval_dataloader, logger,
                             write_pred=True)
        metrics_1 = evaluate_ac(args, model, device, eval_examples, eval_features, eval_dataloader, logger)
        f = open(log_path, "a")
        print("threshold: {}, step: {}, "
              "P_all: {:.4f}, R_all: {:.4f}, F1_all: {:.4f} "
              "P_ae: {:.4f}, R_ae: {:.4f}, F1_ae: {:.4f} "
              "Acc_ac: {:.4f}"
              .format(args.logit_threshold, global_step,
                      metrics_0['p_all'], metrics_0['r_all'], metrics_0['f1_all'],
                      metrics_0['p_ae'], metrics_0['r_ae'], metrics_0['f1_ae'],
                      metrics_1['Acc_ac']), file=f)
        print(" ", file=f)
        f.close()

        print(">" * 50)
        print(" P_all: {:.4f}, R_all: {:.4f}, F1_all: {:.4f}".format(metrics_0['p_all'], metrics_0['r_all'],
                                                                     metrics_0['f1_all']))
        print(" P_ae: {:.4f}, R_ae: {:.4f}, F1_ae: {:.4f}".format(metrics_0['p_ae'], metrics_0['r_ae'],
                                                                  metrics_0['f1_ae']))
        print(" Acc_ac: {:.4f}".format(metrics_1['Acc_ac']))

        logger.info("start delete the model file ...")
        if os.path.exists(save_path):
            os.remove(save_path)
            logger.info("Well done! The model file has been deleted.")
        else:
            print("Sorry! No such file")

if __name__=='__main__':

    main()