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

"""calculates the distance between two flights."""
import numpy as np


def normalize_diff(num_a, num_b, numerical_upper, i):
  if i not in numerical_upper:
    # if no upper bound found, normalize against itself
    num_b, num_a = min(num_a, num_b), max(num_a, num_b)
    if num_b == 0.0:  #  special case when num_b is zero
      if num_a == 0.0:
        deviation = 0
      else:
        deviation = 1
    else:
      deviation = abs(num_a - num_b) * 1.0 / num_b
  else:
    deviation = abs(num_a - num_b) * 1.0 / numerical_upper[i]
    # normalize deviation to be in the same bucket from 0-1
  if deviation > 1:
    deviation = 1
  if deviation < -1:
    deviation = -1
  return abs(deviation)


def distance_calculator(flight1, flight2, debug=False):  # flight2 is benchmark
  # this will be the deviation from benchmark flight (2).
  # simple checks to deal with no flight symbol
  category_set = set([0, 1, 8, 11])
  # numerical_upper = {2: 12, 3: 12, 4: 31, 5: 31, 6: 24, 7: 24, 9: 5000, 10: 2}
  # 9 is missing
  numerical_upper = {2: 12, 3: 12, 4: 31, 5: 31, 6: 24, 7: 24, 10: 2}
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
  # total = 0.0
  values = []
  for i in range(len(flight1)):
    a = flight1[i]
    b = flight2[i]
    if i in category_set:  # for categorical features
      if i == 11:  # for airport, we need to get its attributes
        values.append(airline_list[a.split('_')[1].split('>')[0]] !=
                      airline_list[b.split('_')[1].split('>')[0]])
      else:  #  otherwise we just directly compare their categorical values
        values.append(a != b)
    else:  # numerical features
      # although 2,3 are categorical, we know they are actually numerical
      # and conver them into indices.
      if i == 2 or i == 3:
        num_a = (months.index(a.split('_')[1].split('>')[0]) + 1)
        num_b = (months.index(b.split('_')[1].split('>')[0]) + 1)
      else:
        num_a = float(a.split('_')[1].split('>')[0])
        num_b = float(b.split('_')[1].split('>')[0])
      # we calculate a deviation, which is the increment of a to b
      # compare against b
      # to make them symmetrical
      deviation = normalize_diff(num_a, num_b, numerical_upper, i)
      values.append(deviation)

      if debug:
        if i in numerical_upper:
          upper = numerical_upper[i]
        else:
          upper = None
        print(i, num_a, num_b, upper)
  if debug:
    print(values)
  return sum(values) / 12.0  # there are 12 elements


def retrieve_flight(identity, db_content):
  if 'empty' in identity:
    return None
  for flight in db_content:
    if identity == flight[-1].strip():
      return flight[0:-1]  # should not include flight number
  assert False, 'this should not happen' + str(identity) + str(type(identity))


def split_flight(actual_flight_concat):
  arr = actual_flight_concat.split('_<')
  for i in range(len(arr)):
    arr[i] = arr[i].strip()
    if arr[i][0] != '<':
      arr[i] = '<' + arr[i]
  return arr


def split_db(flight_db):
  # the first one is has reservation object
  splitted = np.array(flight_db.split(' ')[1:])
  flight_db_arr = np.reshape(splitted, [-1, 13])
  return flight_db_arr


def generate_scaled_flight(pred_idx, truth_idx_concat, db_concat):
  """generate the scaled score between two flights based on a flight distance measure.
  """
  # the first one is has reservation object
  db_content = split_db(db_concat)
  all_idx = db_content[:, -1]
  # get flight arr
  truth_idx_arr = split_flight(truth_idx_concat)
  pred_idx = pred_idx.strip()
  if '<fl_empty>' in truth_idx_arr:
    assert len(truth_idx_arr) == 1
    # this is distance measure. not equal should have maximum distance
    return pred_idx != '<fl_empty>'
  else:
    if ('empty' in pred_idx) or pred_idx not in all_idx:
      return 1  # maximum distance
    if pred_idx in truth_idx_arr:  # empty might also in truth_idx_arr
      return 0  # minimal distance

    flight_dist = []
    pred_dist = []
    for fi in all_idx:
      for fi2 in truth_idx_arr:
        if fi != fi2:
          fc = retrieve_flight(fi, db_content)
          fc2 = retrieve_flight(fi2, db_content)
          score = distance_calculator(fc, fc2)  # flight 2 is benchmark
          flight_dist.append(score)
          if fi == pred_idx:
            pred_dist.append(score)
    if not pred_dist:
      # predicted flight not in KB
      return 1
    return min(1, min(pred_dist) * 1.0 / max(flight_dist))
