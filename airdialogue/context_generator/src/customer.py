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

"""This file contains the structure to model customers."""
# import ast
import random
from . import utils


class Customer(object):
  """This class contains information of a customer."""

  def __init__(self, facts_obj, book_window, airport_list):
    # 1. origin and destination, airport_list guarantees to have unique locations
    self.origin = random.randint(0, len(airport_list) - 1)
    self.dest = random.randint(0, len(airport_list) - 1)
    if self.dest == self.origin:
      self.dest = (self.dest + 1) % len(airport_list)
    self.dest = airport_list[self.dest]
    self.origin = airport_list[self.origin]
    # 2. date
    base_time = facts_obj.base_departure_time_epoch
    a_year_from_now = base_time + 3600 * 24 * 365
    # randomly pick a date between base_time and a_year_from_now
    self.departure_date = random.randint(base_time, a_year_from_now)
    # return adte is book_window away from the departure date
    self.return_date = self.departure_date + 3600 * 24 * book_window
    # 4. passenger information
    num_passengers = 1
    self.passengers = []
    len_first_name = len(facts_obj.first_name_list)
    len_last_name = len(facts_obj.last_name_list)
    # '_' will later be replaced in intent standalization
    for _ in range(num_passengers):
      self.passengers.append(
          facts_obj.first_name_list[random.randint(0, len_first_name - 1)] +
          '_' + facts_obj.last_name_list[random.randint(0, len_last_name - 1)])
    # non-required fields during initial query
    # 3. time
    self.departure_time = utils.choice(
        facts_obj.time_list, 1, p=facts_obj.time_prior)[0]
    self.return_time = utils.choice(
        facts_obj.time_list, 1, p=facts_obj.time_prior)[0]

    # 5. class limit and price limit
    self.class_limit = utils.choice(
        facts_obj.class_list, 1, p=facts_obj.class_list_prior)[0]

    # 6. price limist
    if self.class_limit == 'all':
      self.price_limit = facts_obj.price_limit_list[random.randint(
          0,
          len(facts_obj.price_limit_list) - 1)]
    elif self.class_limit == 'economy':
      self.price_limit = facts_obj.price_limit_list[random.randint(
          0,
          len(facts_obj.price_limit_list) - 2)]
    elif self.class_limit == 'business':
      self.price_limit = facts_obj.price_limit_list[random.randint(
          1,
          len(facts_obj.price_limit_list) - 1)]
    # 7. num of connections
    self.max_connection = utils.choice(
        facts_obj.connection_member, 1, p=facts_obj.connection_prior)[0]

    # 8. airline preference
    self.airline = utils.choice(
        facts_obj.airline_preference, 1,
        p=facts_obj.airline_preference_prior)[0]
    # 10 post process
    self.departure_month, self.departure_day = utils.get_month_and_day(
        facts_obj, self.departure_date)
    self.return_month, self.return_day = utils.get_month_and_day(
        facts_obj, self.return_date)
    # 11 change reservation
    self.goal = utils.choice([0, 1, 2], p=facts_obj.goal_probaility)

  def get_departure_and_return_date(self):
    return self.departure_date, self.return_date


  def get_json(self):
    """This function serializes the object into a json."""
    intention_jobject = {}
    intention_jobject['departure_airport'] = self.origin
    intention_jobject['return_airport'] = self.dest
    intention_jobject['departure_month'] = self.departure_month
    intention_jobject['departure_day'] = self.departure_day
    intention_jobject['return_month'] = self.return_month
    intention_jobject['return_day'] = self.return_day
    intention_jobject['name'] = self.passengers[0]
    intention_jobject['departure_time'] = self.departure_time
    intention_jobject['return_time'] = self.return_time
    intention_jobject['class'] = self.class_limit
    intention_jobject['max_price'] = self.price_limit
    intention_jobject['max_connections'] = self.max_connection
    intention_jobject['airline_preference'] = self.airline
    intention_jobject['goal'] = self.goal

    # add departure and return date
    intention_jobject['departure_date'] = self.departure_date
    intention_jobject['return_date'] = self.return_date
    return intention_jobject

  def get_customer_condition(self):
    """This function returns the condition file."""
    condition = self.get_json()
    if condition['airline_preference'] == 'all':
      del condition['airline_preference']
    if condition['max_connections'] == 2:
      del condition['max_connections']
    if condition['class'] == 'all':
      del condition['class']
    if condition['departure_time'] == 'all':
      del condition['departure_time']
    if condition['return_time'] == 'all':
      del condition['return_time']
    return condition

