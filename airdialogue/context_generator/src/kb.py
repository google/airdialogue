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

"""This file defines the behavior of agents."""
import random
import numpy as np
from . import utils


class AirFare(object):
  """This class contains the information for flights."""

  def __init__(self, fact_obj, airport_list, flight_number, ref_departure_date,
               ref_return_date):
    # 1. origin and destination
    self.origin = random.randint(0, len(airport_list) - 1)
    self.dest = random.randint(0, len(airport_list) - 1)
    if self.dest == self.origin:
      self.dest = (self.dest + 1) % len(airport_list)
    self.dest = airport_list[self.dest]
    self.origin = airport_list[self.origin]
    # 2. date and time
    d1 = np.random.normal(ref_departure_date, fact_obj.time_deviation, 1)[0]
    d2 = np.random.normal(ref_return_date, fact_obj.time_deviation, 1)[0]

    # makes ure that departure date comes first
    if d1 < d2:
      self.departure_date = d1
      self.return_date = d2
    else:
      self.departure_date = d2
      self.return_date = d1

    # assert self.departure_date <= self.return_date
    # assert self.return_date - self.departure_date <= 3600*24*365/2

    # 3. class
    self.flight_class = utils.choice(
        fact_obj.class_member, 1, p=fact_obj.class_prior)[0]

    # 4. connections
    self.connection = utils.choice(
        fact_obj.connection_member, 1, p=fact_obj.connection_prior)[0]

    # 5. price
    mean_base_price = fact_obj.flight_price_mean[self.flight_class]
    base = mean_base_price * fact_obj.low_cost_mean_fraction[self.flight_class]
    self.price = int(
        np.random.normal(
            base, base * fact_obj.flight_price_norm_std[self.connection]))
    self.price = utils.discrete_price(self.price)

    # 6. flight number
    # self.flight_number = random.randint(1000, 9999)
    self.flight_number = flight_number

    # 7. airline
    airline_ind = random.randint(0, len(fact_obj.airline_list) - 1)
    self.airline = list(fact_obj.airline_list.keys())[airline_ind]

    # post process
    self.departure_month, self.departure_day = utils.get_month_and_day(
        fact_obj, self.departure_date)
    self.return_month, self.return_day = utils.get_month_and_day(
        fact_obj, self.return_date)
    self.departure_time_num = utils.get_hour(self.departure_date)
    self.return_time_num = utils.get_hour(self.return_date)


  def get_json(self):
    return {
        'departure_airport': self.origin,
        'return_airport': self.dest,
        'departure_month': self.departure_month,
        'return_month': self.return_month,
        'departure_day': self.departure_day,
        'return_day': self.return_day,
        'departure_time_num': self.departure_time_num,
        'return_time_num': self.return_time_num,
        'class': self.flight_class,
        'num_connections': self.connection,
        'price': self.price,
        'flight_number': self.flight_number,
        'airline': self.airline
    }


class Knowledgebase(object):
  """This class contains a collection of flights."""

  def __init__(self, fact_obj, num_flights, airport_list, departure_date,
               return_date):
    self.knowledgebase = []
    base_flight_num = 1000
    for i in range(num_flights):
      self.knowledgebase.append(
          AirFare(fact_obj, airport_list, base_flight_num + i, departure_date,
                  return_date))
    has_reserv = np.random.random() < fact_obj.has_reservation_probability
    self.has_reservation = has_reserv
    if self.has_reservation:
      ind = np.random.randint(0, len(
          self.knowledgebase))  # note the numpy random above behavior
      self.reservation = self.knowledgebase[ind].flight_number
    else:
      self.reservation = 0  # 0 serves as the null number

  def get_json(self):
    knowledge_base_json = []
    for flight in self.knowledgebase:
      knowledge_base_json.append(flight.get_json())
    wrapper_json = {'kb': knowledge_base_json, 'reservation': self.reservation}
    return wrapper_json
