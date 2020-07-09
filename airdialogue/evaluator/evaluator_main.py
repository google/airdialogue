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
from airdialogue.evaluator.selfplay_utils import compute_reward as compute_reward2

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
      '--output',
      type=str,
      default='score.json',
      help='output path for score json.')


def distance_calculator(flight1, flight2, flight_db):  # flight2 is benchmark
  # this will be the deviation from benchmark flight (2).
  # simple checks to deal with no flight symbol
  if flight1 == flight2:
    return 0.0, True
  splitted = np.array(
      flight_db.split(' ')[1:])  # the first one is has reervation object
  flight_arr = np.reshape(splitted, [-1, 13])
  flight_1_actual = None
  flight_2_actual = None
  #   print flight_arr,flight_arr[0]
  for flight in flight_arr:
    if flight1 == flight[-1]:
      flight_1_actual = flight[0:-1]  # should not in clude flight number
    if flight2 == flight[-1]:
      flight_2_actual = flight[0:-1]
  if (flight_2_actual is None) or (flight_1_actual is None):
    # they are not equal and one of them is none, will return zero reward
    return 1.0, False
  # assert flight_1_actual and flight_2_actual
  #  <a1_HOU> <a2_JFK> <m1_Feb> <m2_Feb> <d1_22> <d2_23> <tn1_2> <tn2_11> <cl_economy> <pr_100> <cn_1> <al_JetBlue> <fl_1000>
  total = 0.0
  category_set = set([0, 1, 2, 3, 4, 5, 8, 11])
  # numerical_upper = {2: 12, 3: 12, 4: 31, 5: 31, 6: 24, 7: 24, 9: 5000, 10: 2}
  months = [
      'Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct',
      'Nov', 'Dec'
  ]
  # 0 is normal cost
  # 1 is low cost
  airline_list = {
      'UA': 0,
      'AA': 0,
      'Delta': 0,
      'Hawaiian': 0,
      'Southwest': 1,
      'Frontier': 1,
      'JetBlue': 1,
      'Spirit': 1
  }

  for i in range(len(flight_1_actual)):
    a = flight_1_actual[i]
    b = flight_2_actual[i]
    if i in category_set:
      if i == 11:
        total += airline_list[a.split('_')[1].split('>')[0]] != airline_list[
            b.split('_')[1].split('>')[0]]
      else:
        total += a != b
    else:
      if i == 2 or i == 3:
        num_a = (months.index(a.split('_')[1].split('>')[0]) + 1)
        num_b = (months.index(b.split('_')[1].split('>')[0]) + 1)
      else:
        num_a = float(a.split('_')[1].split('>')[0])
        num_b = float(b.split('_')[1].split('>')[0])
      # assert num_a <= 1.0 and num_b <= 1.0
      if num_b == 0.0:
        if num_a == 0.0:
          deviation = 0
        else:
          deviation = 1
      else:
        deviation = (num_a - num_b) * 1.0 / num_b

      if deviation > 1:
        deviation = 1
      if deviation < -1:
        deviation = -1

      total += abs(deviation)
  return total / 12.0, True  # there are 12 elements


def generate_scaled_flight(predicted_flight, actual_flight, flight_db):
  splitted = np.array(
      flight_db.split(' ')[1:])  # the first one is has reervation object
  flight_arr = np.reshape(splitted, [-1, 13])
  all_flights = flight_arr[:, -1]
  flight_dist = []
  for flight in all_flights:
    score, _ = distance_calculator(flight, actual_flight,
                                   flight_db)  # flight 2 is benchmark
    flight_dist.append(score)
  max_score = max(flight_dist)
  original_score, valid = distance_calculator(predicted_flight, actual_flight,
                                              flight_db)
  if valid == False:  # this is not a valid flivght
    return 1
  else:
    return min(1, original_score / max_score)


def compute_reward(predicted_action,
                   actual_action,
                   flight_db,
                   alpha=0.5,
                   beta=0.2,
                   gamma=0.3,
                   debug=False):

  predicted_name, predicted_flight, predicted_state = predicted_action
  actual_name, actual_flight, actual_state = actual_action

  score1 = f1_score(predicted_name, actual_name)
  score2 = 1 - generate_scaled_flight(predicted_flight, actual_flight,
                                      flight_db)
  score3 = float(predicted_state == actual_state)

  reward_compliment = score1 * 0.2 + score2 * 0.5 + score3 * 0.3

  acc1 = score1
  acc2 = score2
  acc3 = score3
  return reward_compliment, acc1, acc2, acc3


def process_action(original_action):
  """add dummy flight and dummy name in the cases that they are not present."""
  if 'flight' not in original_action:
    original_action['flight'] = 'dummy_flight'
  if 'name' not in original_action:
    original_action['name'] = 'dummy_name'
  return original_action['name'], original_action['flight'], original_action[
      'status']


def score_human_data(flags):
  assert flags.true_data and flags.true_kb
  scores = []
  expanded_kb = expanduser(flags.true_kb)
  expanded_data = expanduser(flags.true_data)
  f2 = gfile.Open(expanded_kb)
  with gfile.Open(expanded_data) as f:
    for line in tqdm(f):
      a = json.loads(line)
      line2 = f2.readline()
      if a['correct_sample'] == False:
        ss = compute_reward(
            process_action(a['action']), process_action(a['expected_action']),
            tokenize_kb(json.loads(line2)))
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
  infer_bleu = evaluate_infer(expanded_true_data, expanded_pred_data, 'bleu')
  print('infer bleu: ', infer_bleu)
  return {'bleu': infer_bleu}

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
          score = compute_reward2(pred_action, true_action, kb)
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
