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

"""This file contains content related parameters."""

from tensorflow.compat.v1 import gfile


class Facts(object):
  """This class contains a collection of hyper-parameters and known facts."""

  def __init__(self, firstname_file, lastname_file, airportcode_file):
    self.airline_list = self._get_airline_list()
    self.price_limit_list = [200, 500, 1000, 5000]
    self.time_list = ['morning', 'afternoon', 'evening', 'all']
    self.time_prior = [0.03, 0.04, 0.03, 0.9]
    self.airline_preference = ['normal-cost', 'all']
    self.airline_preference_prior = [0.05, 0.95]

    # for airlines
    # mean, std
    self.flight_price_mean = {'economy': 300, 'business': 1300}
    self.flight_price_norm_std = {0: 0.2, 1: 0.4, 2: 0.6}
    self.low_cost_mean_fraction = {'economy': 0.7, 'business': 0.5}

    # prob
    self.class_member = ['economy', 'business']
    self.class_prior = [0.9, 0.1]

    self.class_list = ['all', 'economy', 'business']
    self.class_list_prior = [0.9, 0.07, 0.03]

    # prob_class={'economy':0.9,'business':0.1}
    self.max_connections = 2
    self.connection_member = [0, 1, 2]
    self.connection_prior = [0.07, 0.9, 0.03]

    # flight price= N(fraction*mean +-fraction*mean*n_std)
    self.airline_type = ['low-cost', 'normal-cost']

    self.first_name_list = self._read_file(firstname_file)
    self.last_name_list = self._read_file(lastname_file)
    airport_list_raw = self._read_file(airportcode_file)
    ob = self._get_airport_and_full_name(airport_list_raw)
    self.airport_list, self.airport_to_full_name = ob
    # all months
    self.months = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct',
        'Nov', 'Dec'
    ]

    # has reservation
    self.has_reservation_probability = 0.10
    self.goal_probaility = [0.80, 0.1, 0.1]  # book, change, cancel
    self.goal_str_arr = ['book', 'change', 'cancel']

    # departure_time
    self.base_departure_time_epoch = 1300000000
    self.time_deviation = 3600 * 12

    self.airport_to_full_name = self._get_airline_list()

    self.final_action_list = [
        'no_flight', 'book', 'no_reservation', 'cancel', 'change'
    ]

  def _get_airline_list(self):
    """generates a list of airline and their cost attributes."""
    airline_list = {
        'UA': 'normal-cost',
        'AA': 'normal-cost',
        'Delta': 'normal-cost',
        'Hawaiian': 'normal-cost',
        'Southwest': 'low-cost',
        'Frontier': 'low-cost',
        'JetBlue': 'low-cost',
        'Spirit': 'low-cost'
    }
    return airline_list

  def _read_file(self, filename):
    original_content = gfile.Open(filename).read().strip().split('\n')
    # make sure that no empty items are added.
    content = []
    for element in original_content:
      element = element.strip()
      if element:
        content.append(element)
    return content

  def _get_airport_and_full_name(self, airport_list_raw):
    airport_to_full_name = {}
    airport_code = []
    for element in airport_list_raw:
      code, content = element.split(':')
      airport_code.append(code)
      airport_to_full_name[code] = content
    return airport_code, airport_to_full_name
