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
"""This module builds a falsk server for visualization."""

import argparse
from os.path import expanduser
from tensorflow.compat.v1 import gfile
from collections import Counter
import nltk
import numpy as np
import json
import sys

from airdialogue.prepro.tokenize_lib import tokenize_kb

from airdialogue.evaluator.metrics.f1 import f1_score
from airdialogue.evaluator.metrics.bleu import compute_bleu
from airdialogue.evaluator.infer_utils import evaluate as evaluate_infer
from airdialogue.evaluator.selfplay_utils import compute_reward

from tqdm import tqdm
import tensorflow.compat.v1 as tf

FLAGS = None


def add_arguments(parser):
  """Build ArgumentParser."""
  parser.add_argument(
      '--true_data', type=str, default='', help='Path to the true data file.')
  parser.add_argument(
      '--true_kb', type=str, default='', help='Path to the kb file.')
  parser.add_argument(
      '--pred_data', type=str, default='', help='Path to the prediction file.')
  parser.add_argument(
      '--task',
      type=str,
      default='infer',
      help='type of the task, one of |human|infer|selfplay|')
  parser.add_argument(
      '--infer_metrics',
      type=str,
      default='bleu:brief',
      help='For infer task, choose one of multiple metric in (bleu:all|rouge:all|kl:all) or (bleu:brief|kl:brief),'
      ' this will give you a single number metric. (bleu|kl) is equivalent to (belu:brief|kl:brief) ')
  parser.add_argument(
      '--output',
      type=str,
      default='score.json',
      help='output path for score json.')

def score_human_data(flags):
  assert flags.true_data and flags.true_kb
  scores = []
  expanded_kb = expanduser(flags.true_kb)
  expanded_data = expanduser(flags.true_data)
  f2 = gfile.Open(expanded_kb)
  with gfile.Open(expanded_data) as f:
    for line in tqdm(f):
      a = json.loads(line)
      kb_line = f2.readline()
      if a['correct_sample'] == False:
        pred_action = action_obj_to_str(a['action'])
        true_action = action_obj_to_str(a['expected_action'])
        kb = tokenize_kb(json.loads(kb_line))
        ss = compute_reward(pred_action, true_action, kb)
        scores.append(ss)
      else:
        scores.append([1, 1, 1, 1])
  sn = np.array(scores)
  # np.mean(sn[:,0]), np.mean(sn[:,1]),np.mean(sn[:,2]),np.mean(sn[:,3])
  score = np.mean(sn[:, 0])
  print('final score', score)
  return {'score': score}


def score_inference(flags):
  assert flags.true_data and flags.pred_data
  expanded_true_data = expanduser(flags.true_data)
  expanded_pred_data = expanduser(flags.pred_data)

  infer_metrics = flags.infer_metrics.split(',')
  results = {}

  for metric in infer_metrics:
      infer_result = evaluate_infer(expanded_true_data, expanded_pred_data, metric)
      metric = metric.split(":")[0]
      print('infer ', metric, ': ', infer_result)
      results[metric] = infer_result
  return results

def action_obj_to_str(o):
  fl = ['empty']
  if 'flight' in o and o['flight']:
    fl = [str(f) for f in o['flight']]
  if 'name' not in o:
    o['name'] = '<unk> <unk>'
  if 'status' not in o:
    o['status'] = 'unk'
  return o['name'], "_".join(['<fl_' + f + '>' for f in fl]), '<st_' + o['status'] + '>'

def json_obj_to_tokens(o):
  d = o['dialogue']
  decapped = [' '.join(s.split(':')[1:]).strip() for s in d]
  one_string = ' '.join(decapped)
  tokenized = nltk.word_tokenize(one_string)
  return tokenized


def score_selfplay(flags):
  assert flags.true_data and flags.true_kb and flags.pred_data
  # check output

  all_score = []
  bleu_scores = []
  with tf.gfile.GFile(flags.pred_data) as f:
    with tf.gfile.GFile(flags.true_data) as t:
      with tf.gfile.GFile(flags.true_kb) as kb:
        for pred_line, true_line, kb_line in tqdm(list(zip(f, t, kb))):
          pred_json_obj = json.loads(pred_line)
          true_json_obj = json.loads(true_line)
          kb = tokenize_kb(json.loads(kb_line))
          pred_action = ''
          if 'action' not in pred_json_obj:
            pred_action = '<unk> <unk> <unk> <unk>'.split(' ')
          else:
            pred_action = action_obj_to_str(pred_json_obj['action'])
          true_action = action_obj_to_str(true_json_obj['expected_action'])
          score = compute_reward(pred_action, true_action, kb)
          all_score.append(score)

          pred_raw_text = json_obj_to_tokens(pred_json_obj)
          true_raw_text = json_obj_to_tokens(true_json_obj)

          b = compute_bleu([[true_raw_text]], [pred_raw_text])
          bleu_scores.append(b[0] * 100)

  avg_score = np.mean(all_score)
  avg_bleu = np.mean(bleu_scores)
  print('score=', avg_score)
  print('bleu=', avg_bleu)

  return {'score': avg_score, 'bleu': avg_bleu}


def main(flags):
  if flags.task == 'human':
    score = score_human_data(flags)
  elif flags.task == 'infer':
    score = score_inference(flags)
  else:
    score = score_selfplay(flags)

  with tf.gfile.GFile(flags.output, 'w') as f:
    f.write(json.dumps(score))


def run_main(unused):
  main(FLAGS)


if __name__ == '__main__':
  this_parser = argparse.ArgumentParser()
  add_arguments(this_parser)
  FLAGS, unparsed = this_parser.parse_known_args()
  tf.app.run(main=run_main, argv=[sys.argv[0]] + unparsed)
