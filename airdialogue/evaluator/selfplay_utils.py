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

from airdialogue.evaluator.metrics.flight_distance import generate_scaled_flight
from airdialogue.evaluator.metrics import f1


def compute_reward_batch(utterance,
                         predicted_action,
                         actual_action_concat,
                         flight_db,
                         hparams,
                         alpha=0.5,
                         beta=0.2,
                         gamma=0.3):
  """Calcualte the reward for a batch."""
  rewards = []
  acc1 = []
  acc2 = []
  acc3 = []
  discrete_score = []
  ds1_arr = []
  ds2_arr = []
  ds3_arr = []
  train_rw_arr = []
  for pa, aa_con, fl in zip(predicted_action, actual_action_concat, flight_db):
    aa = aa_con.split(' ')
    rw, ac1, ac2, ac3 = compute_reward(pa, aa, fl)
    rewards.append(rw)
    acc1.append(ac1)
    acc2.append(ac2)
    acc3.append(ac3)

    ds, ds1, ds2, ds3 = compute_01_score(pa, aa)
    discrete_score.append(ds)
    ds1_arr.append(ds1)
    ds2_arr.append(ds2)
    ds3_arr.append(ds3)
    train_rw_arr.append(
        get_training_reward(hparams, ac1, ac2, ac3, ds1, ds2, ds3))
  return train_rw_arr, rewards, acc1, acc2, acc3, discrete_score, ds1_arr, ds2_arr, ds3_arr


def parse_action(action):
  """parse the action and consider multiple name scenario.
  name will also appear first.
  """
  name = ' '.join(action[0:-2])
  flight = action[-2]
  state = action[-1]
  return name, flight, state


def compute_reward(predicted_action,
                   actual_action,
                   flight_db,
                   alpha=0.5,
                   beta=0.2,
                   gamma=0.3,
                   debug=False):
  """here we compute the scaled reward."""
  predicted_name, predicted_flight, predicted_state = parse_action(
      predicted_action)
  actual_name, actual_flight, actual_state = parse_action(actual_action)

  # this will do normalization including lower case and prouncation/space
  # removal
  score1 = f1.f1_score(predicted_name, actual_name)
  score2 = 1 - generate_scaled_flight(predicted_flight, actual_flight,
                                      flight_db)
  score3 = float(predicted_state == actual_state)

  reward_compliment = score1 * 0.2 + score2 * 0.5 + score3 * 0.3

  acc1 = score1
  acc2 = score2
  acc3 = score3
  return reward_compliment, acc1, acc2, acc3


