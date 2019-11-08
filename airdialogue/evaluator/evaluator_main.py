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
from tensorflow import gfile
from collections import Counter
import numpy as np
import json
from airdialogue.prepro.tokenlize_lib import tokenlize_kb
from airdialogue.evaluator.metrics.f1 import f1_score
from tqdm import tqdm


def add_arguments(parser):
  """Build ArgumentParser."""
  parser.add_argument(
      '--data', type=str, default='', help='Path to the data file.')
  parser.add_argument('--kb', type=str, default='', help='Path to the kb file.')
  parser.add_argument(
      '--pred', type=str, default='', help='Path to the prediction file.')


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
        total += airline_list[a.split('_')[1].split(
            '>')[0]] != airline_list[b.split('_')[1].split('>')[0]]
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


def main(FLAGS):
  scores = []
  expanded_kb = expanduser(FLAGS.kb)
  expanded_data = expanduser(FLAGS.data)
  f2 = gfile.Open(expanded_kb)
  cnt = 0
  with gfile.Open(expanded_data) as f:
    for line in tqdm(f):
      # if cnt % 1000 == 0:
      #   print cnt
      cnt += 1

      a = json.loads(line)
      line2 = f2.readline()

      if a['correct_sapmle'] == False:
        ss = compute_reward(
            process_action(a['action']), process_action(a['expected_action']),
            tokenlize_kb(json.loads(line2)))
        scores.append(ss)
      else:
        scores.append([1, 1, 1, 1])
  sn = np.array(scores)
  # np.mean(sn[:,0]), np.mean(sn[:,1]),np.mean(sn[:,2]),np.mean(sn[:,3])
  print 'final score', np.mean(sn[:, 0])


if __name__ == '__main__':
  this_parser = argparse.ArgumentParser()
  add_arguments(this_parser)
  FLAGS, unparsed = this_parser.parse_known_args()
  main(FLAGS)
