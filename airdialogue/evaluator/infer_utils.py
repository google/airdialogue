# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility for evaluating various tasks."""
import codecs
import tensorflow.compat.v1 as tf

from airdialogue.evaluator.metrics import bleu
from airdialogue.evaluator.metrics import rouge
from airdialogue.evaluator.metrics import kl

ROLE_TOKENS = ["<t1>", "<t2>"]


def evaluate(ref_file, trans_file, metric):
  """Pick a metric and evaluate depending on task."""
  if ":" in metric:
    metric, mode = metric.split(":")
  else:
    mode = "brief"
  assert mode in ["brief", "all"]
  # BLEU scores for translation task
  if metric.lower() == "bleu":
    evaluation_score = _bleu(
        ref_file, trans_file, mode=mode)
  # ROUGE scores for summarization tasks
  elif metric.lower() == "rouge":
    evaluation_score = _rouge(
        ref_file, trans_file, mode=mode)
  # kl scores for evaluating the ngram kl distribution of the whole corpus
  elif metric.lower() == "kl":
    evaluation_score = _kl(
        ref_file, trans_file, mode=mode)
  elif metric.lower() == "accuracy":
    evaluation_score = _accuracy(ref_file, trans_file)
  else:
    raise ValueError("Unknown metric %s" % metric)

  return evaluation_score

def _kl(ref_file, trans_file, mode="brief"):
  """Compute KL divergence and handling BPE."""
  max_order = 4

  ref_files = [ref_file]
  reference_text = []
  role_tokens = []
  for reference_filename in ref_files:
    with codecs.getreader("utf-8")(tf.gfile.GFile(reference_filename,
                                                  "rb")) as fh:
      for line in fh:
        reference, role = process_dialogue_infer(
            line.rstrip(), get_role_token=True)
        reference_text.append(reference.split(" "))
        role_tokens.append(role)

  translations = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
    for line in fh:
      translations.append(line.rstrip().split(" "))

  results = {}
  kl_scores = kl.compute_kl(reference_text, translations, max_order)
  for key in kl_scores:
    results["all-" + key] = kl_scores[key]
  if mode == "brief":
    return sum(results.values()) / len(results)

  for role in ROLE_TOKENS:
    _sub_ref_texts = []
    _sub_trans = []
    for _r, _t, _role in zip(reference_text, translations, role_tokens):
      if _role == role:
        _sub_ref_texts.append(_r)
        _sub_trans.append(_t)
    kl_scores = kl.compute_kl(_sub_ref_texts, _sub_trans, max_order)
    for key in kl_scores:
      results[role + "-" + key] = kl_scores[key]
  return results

def _bleu(ref_file, trans_file, mode="brief"):
  """Compute BLEU scores and handling BPE."""
  max_order = 4
  smooth = False

  ref_files = [ref_file]
  reference_text = []
  for reference_filename in ref_files:
    with codecs.getreader("utf-8")(tf.gfile.GFile(reference_filename,
                                                  "rb")) as fh:
      reference_text.append(fh.readlines())

  per_segment_references = []
  role_tokens = []
  for references in zip(*reference_text):
    reference_list = []
    for reference in references:
      reference, role = process_dialogue_infer(
          reference.rstrip(), get_role_token=True)
      reference_list.append(reference.split(" "))
    per_segment_references.append(reference_list)
    role_tokens.append(role)

  translations = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
    for line in fh:
      translations.append(line.rstrip().split(" "))

  results = {}
  bleu_score, _, _, _, _, _ = bleu.compute_bleu(per_segment_references,
                                                translations, max_order, smooth)
  results["all"] = 100 * bleu_score
  if mode == "brief":
    return results["all"]

  for role in ROLE_TOKENS:
    _sub_ref_texts = []
    _sub_trans = []
    for _r, _t, _role in zip(per_segment_references, translations, role_tokens):
      if _role == role:
        _sub_ref_texts.append(_r)
        _sub_trans.append(_t)
    bleu_score, _, _, _, _, _ = bleu.compute_bleu(_sub_ref_texts, _sub_trans,
                                                  max_order, smooth)
    results[role] = 100 * bleu_score

  return results


def _rouge(ref_file, summarization_file, mode="brief"):
  """Compute ROUGE scores and handling BPE."""

  results = {}

  references = []
  role_tokens = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(ref_file, "rb")) as fh:
    for line in fh:
      ref, role = process_dialogue_infer(line.rstrip(), get_role_token=True)
      references.append(ref)
      role_tokens.append(role)

  hypotheses = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(summarization_file,
                                                "rb")) as fh:
    for line in fh:
      hypotheses.append(line)

  rouge_score_map = rouge.rouge(hypotheses, references)
  results["all"] = 100 * rouge_score_map["rouge_l/f_score"]
  if mode == "brief":
    return results["all"]

  for role in ROLE_TOKENS:
    _sub_ref_texts = []
    _sub_hypos = []
    for _r, _t, _role in zip(references, hypotheses, role_tokens):
      if _role == role:
        _sub_ref_texts.append(_r)
        _sub_hypos.append(_t)
    rouge_score_map = rouge.rouge(_sub_hypos, _sub_ref_texts)
    results[role] = 100 * rouge_score_map["rouge_l/f_score"]

  return results


def process_dialogue_infer(file_line, get_role_token=False):
  # split the end token (<t1>,<t2>)
  _line = file_line.replace(" <eod>", "")
  _line = _line.rstrip().split("|")[1].rsplit(" ", 1)
  if not get_role_token:
    return _line[0]
  else:
    return _line[0], _line[1]


def _accuracy(label_file, pred_file):
  """Compute accuracy, each line contains a label."""

  with codecs.getreader("utf-8")(tf.gfile.GFile(label_file, "rb")) as label_fh:
    with codecs.getreader("utf-8")(tf.gfile.GFile(pred_file, "rb")) as pred_fh:
      count = 0.0
      match = 0.0
      for label, pred in zip(label_fh, pred_fh):
        label = process_dialogue_infer(label.strip()).strip()
        pred = pred.strip()
        if label == pred:
          match += 1
        count += 1
  return 100 * match / count
