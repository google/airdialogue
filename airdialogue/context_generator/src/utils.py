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

"""this file contains a list of utility functions."""
import calendar
from datetime import datetime
import numpy as np


def continue_booking(facts, ag_cond, kb, goal_str):
  flights = airflight_selector(facts, ag_cond, kb)  # this is a list of flights
  if not flights:  # terminal condition
    status = 'no_flight'
  else:
    status = goal_str
  return status, flights


def airflight_selector(facts, condition, airfare_database):
  """This function selects a flight based on the condition."""
  candidates = []
  for i, flight in enumerate(airfare_database):
    cond = check_condition(facts, flight, condition, get_full_diff=False)
    if cond == 'satisfied':
      candidates.append(i)
  candidates.sort(key=lambda a: airfare_database[a]['price'])
  if not candidates:
    return None
  else:
    upper = 0
    flight = airfare_database[candidates[upper]]
    val = flight['price']
    all_good_flights = [flight]
    upper += 1
    # find all flights with the same price
    while upper < len(candidates):
      flight = airfare_database[candidates[upper]]
      if flight['price'] == val:
        all_good_flights.append(flight)
        upper += 1
      else:
        break
    return all_good_flights


def generate_expected_action(facts, agent_condition, airfare_database,
                             reservation):
  """this function generates the expected action based context."""
  goal_str = agent_condition['goal']

  if goal_str == 'book' or (goal_str == 'change' and reservation != 0):
    status, flights = continue_booking(facts, agent_condition, airfare_database,
                                       goal_str)
  elif goal_str in ['change', 'cancel'] and reservation == 0:
    status = 'no_reservation'
    flights = []
  elif goal_str == 'cancel':
    status = 'cancel'
    flights = []
  expected_action = generate_action(flights, agent_condition['name'], status)
  return expected_action


def check_states(user_action, expected_action):
  """this function check the dialogue states of the json_object.
  assume both expected action and user action went through
  action standarlization, which means they must have a flight and name field.
  """
  # first status needs to match
  status_match = expected_action['status'] == user_action['status']
  # second flight need to match
  #  if user flight is empty, expected flight has to be empty
  if len(user_action['flight']) == 0:
    flight_match = len(expected_action['flight']) == 0
  else:
    # there can only be one user flight
    assert len(user_action['flight']) == 1
    flight_match = user_action['flight'][0] in expected_action['flight']
  a = expected_action['name'].strip().lower()
  b = user_action['name'].strip().lower()
  name_match = (a == b)
  res = status_match and flight_match and name_match
  return res, status_match, flight_match, name_match


def get_day_segment_by_hour(hour):
  if hour < 3 or hour > 19:
    return 'evening'
  elif hour >= 3 and hour <= 11:
    return 'morning'
  else:
    return 'afternoon'


def check_condition(facts, flight, condition, get_full_diff=False):
  """Check the condition of the fliths to see whether it is satisfied."""
  # all the keys here have been confirmed to appear in step2 of data release.
  diff = []
  do_not_consider = set(['name', 'goal', 'departure_date', 'return_date'])
  for key in set(condition.keys()) - do_not_consider:
    if key == 'departure_time':
      day_segment = get_day_segment_by_hour(flight['departure_time_num'])
      if day_segment != condition[key]:
        diff.append(key)
    elif key == 'return_time':
      day_segment = get_day_segment_by_hour(flight['return_time_num'])
      if day_segment != condition[key]:
        diff.append(key)
    elif key == 'airline_preference':
      if condition[key] != facts.airline_list[flight['airline']]:
        diff.append(key)
    elif key == 'max_connections':
      if condition[key] < flight['num_connections']:
        # print ('ss',condition[key],flight['num_connections'])
        diff.append(key)
    elif key == 'max_price':
      if condition[key] < flight['price']:
        diff.append(key)
    else:
      # this includes, class, departure_airport/return_airport
      # departure_month/return_month, departure_day/return_day
      if flight[key] != condition[key]:
        diff.append(key)
    if not get_full_diff and len(diff):
      return diff
  if not diff:
    return 'satisfied'
  else:
    return diff


def generate_action(flights, name, status):
  """generate dialogue action."""
  action_json = {}
  if status.startswith('no_flight'):
    status = 'no_flight'
  action_json['status'] = status
  flight_lst = []  #  always have an empty flight list
  if status in ['change', 'book']:
    for f in flights:
      flight_lst.append(f['flight_number'])
  action_json['flight'] = flight_lst
  action_json['name'] = name
  # standarlize it just in case
  standarlized_action = standardize_action(action_json)
  return standarlized_action


def standardize_intent(org_intent):
  """1) get ride of departure_date and return_date in the intent.
     2) also get tide of intents with 'all'. also for max_conn if val is 2.
     3) make goal a string.
     4) replace '_' with ' ' in name.
  """
  new_intent = {}
  for key in org_intent:
    if key in ['departure_date', 'return_date']:
      continue
    if org_intent[key] == 'all':
      continue
    if key == 'max_connections' and org_intent[key] == 2:
      continue
    new_intent[key] = org_intent[key]
  new_intent['goal'] = ['book', 'change', 'cancel'][new_intent['goal']]
  new_intent['name'] = new_intent['name'].replace('_', ' ').strip()

  return new_intent


def standardize_action(org_action):
  """if status is not book or change the flight number will be empty.
  name is always required."""
  # some human raters will end a name with . or ,
  # since names in intent are standarlized (with - being replaced by space),
  # it will not be necessary to consider - again in the action standarlization.
  original_name = org_action['name'].strip()
  name = []
  for d in original_name:
    if d.isalpha() or d == ' ':
      name.append(d)
  name = ''.join(name)
  status = org_action['status']
  # if flight is a single int, we will need to convert it to a list
  # ground truth can have multiple flights
  # prediction and real data have no more than one element in the flight list.
  flight = org_action['flight']
  if type(flight) == int:
    flight = [flight]
  if status == 'book' or status == 'change':
    new_flight = []
    # get ride of anything that is not a valid flight number.
    # This could be the empty flights in early version of the UI.
    for f in flight:
      if int(f) >= 1000:
        new_flight.append(f)
  else:
    #  otherwise we provide an empty list of the flight
    #  any user selecged flights that does not come with bookable status
    #  will be ignored.
    new_flight = []

  return {'flight': new_flight, 'name': name, 'status': status}


def get_unix_epoch(dt):
  return calendar.timegm(dt.utctimetuple())


def get_datetime(unix_epoch):
  return datetime.utcfromtimestamp(unix_epoch)


def get_month_and_day(fact_obj, unix_epoch):
  m = get_datetime(unix_epoch).month
  d = get_datetime(unix_epoch).day
  month = fact_obj.months[m - 1]
  return month, str(d)


def get_hour(unix_epoch):
  return get_datetime(unix_epoch).hour


def discrete_price(original_price):
  return max(100, int(original_price / 100) * 100)


def format_time(hour):
  return str(hour) + ':00 in the ' + get_day_segment_by_hour(hour)


def get_connection(con):
  if con == 0:
    return 'direct service'
  elif con == 1:
    return '1 connection'
  else:
    return '2 connections'


def discrete_sample_(probabilities):
  if abs(sum(probabilities) - 1) > 1e-3:
    raise ValueError('sum of probability not equal to 1')
  sm = 0.0
  random_number = np.random.random()
  for i in range(len(probabilities)):
    sm += probabilities[i]
    if sm >= random_number:
      return i
  assert False, 'invalid path'


def choice(values, cnt=-1, p=None):
  if p is None:
    p = [float(1) / float(len(values))] * len(values)
  arr = []
  for _ in range(abs(cnt)):
    ind = discrete_sample_(p)
    arr.append(values[ind])
  if cnt == -1:
    return arr[0]
  else:
    return arr
