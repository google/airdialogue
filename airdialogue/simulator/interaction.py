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

"""This file contains the main dialogue logic."""

import numpy as np
from airdialogue.context_generator.src import utils
greeting_pairs = [('Hello.', 'Hi.'), ('Hey, how are you.',
                                      'I am fine. Thanks for asking.'),
                  ('Hello there.', 'Hey!'), ('Hey there.', 'Hi.'),
                  ('Hi.', 'Hello.'), ('How is it going?',
                                      'I am good. Thank you.')]

# spaces will be processed in tokenizer
agent_ask = ['How can I help you today?', 'What can I do for you today?']

customer_request = {
    'book': [
        'I would like to book a flight {0}.', 'I want to book a flight {0}.',
        'Please help me to book a flight {0}.',
        'Could you help me to book a flight {0}?'
    ],
    'cancel': [
        'Could you help me to cancel my flight?',
        'Please help me to cnacel my reservation.',
        'I would like to cancel my reservation.'
    ],
    'change': [
        'I would like to change my reservation.',
        'Could you help me to change my recent flight reservation?',
        'I need helps to change my reservation.'
    ]
}

# c1=departure city
# d1= departure date
# c2=return city
# d2=return date
agent_first_respond = {
    'book': {
        'c1c2': [
            'Sure, may I start with the departure and arrival city',
            'Not a problem. What\'s your departure and arrival city?',
            'Happy to do so. Could you tell me your departure and arrival city?'
        ],
        'd1d2': [
            'What\'s the departure and return date?',
            'What day do you want to depart and return?'
        ]
    },
    'change': {
        'c1c2': [
            'Sure. What\' the departure and arrival city for your updated '
            'reservation?',
            'Not a problem. Could you tell me the departure and arrival city '
            'for your updated reservation?',
            'I see. Could you tell me the departure and arrival city of your '
            'updated reservation?'
        ],
        'd1d2': [
            'What\'s the departure and return date?',
            'What day do you want to depart and return?'
        ]
    }
}

customer_first_respond = {
    'c1c2': [
        'My departure city is {0} and my return city is {1}.',
        'I will be flying from {0} to {1}.',
        'Departure city is {0} and return city is {1}.'
    ],
    'd1d2': [
        'I will be departing on {0} and returning on {1}.',
        'I am plnning to depart on {0} and return on {1}.'
    ]
}

# agent_recurrening_respond={['What\'s your {0}?',
#                            'Could you let me know your {0}?',
#                            'I will need to know your {0}.']}
customer_recurring_respond = {
    'departure_time': [
        'I am actually considering departing in the {0}.',
        'Is there any flight that departs in the {0}?',
        'I am actually thinknig about departing in the {0}.'
    ],
    'return_time': [
        'I would prefer to return in the {0}.',
        'Can you find a flight that returns in the {0}?'
    ],
    'class': [
        'I am looking for a {0} class flight.',
        'Do you have any {0} class flights available?'
    ],
    'max_connections': [
        'I prefer no more than {0} connections.',
        'Could you find me flights that are less than or equal to {0} '
        'connections?'
    ],
    'max_price': [
        'I have a strict budget constraint. Could you find me flights that are'
        ' below ${0}.',
        'My budget limits me to flights that are cheaper than ${0}.'
    ],
    'airline_preference': [
        'Could you help me to find a normal fare airline other than a low-cost'
        ' one?', 'I prefer not to fly low-cost airlines.'
    ],
    'satisfied': ['This flight looks good.', 'Sounds good.', 'Looking good.']
}

agent_suggestion = {
    'full': [
        'We have flight {0} departs from {1} on {2} at {3} and returns from '
        '{4} on {5} at {6}. This is a {7} class flight with {8}. The total '
        'price is ${9}. '
    ],
    'without_cd': [
        'I have flight {0} departs the same day at {1} and returns the same '
        'day at {2}. This is a {3} class flight with {4} and a total price of '
        '${5}.',
        'How about flight {0} departs the same day at {1} and rturns the same '
        'day at {3}. It is a {3} class with {4} and a price of ${5}.'
    ]
}

agent_ask_name = [
    'May I have the name for the reservation?',
    'What\'s the name for this reservation?',
    'Could you let me know the name for this reservation?'
]
cutomer_name = ['My name is {0}.', '{0}']

agent_confirm_change = {
    'cancel': [
        'I am able to locate your ticket. Could you confirm that you want to '
        'cancel it?',
        'I found your reservation. Are you sure you want to cancel it?',
        'Just want to confirm that you want to cancel the ticket, correct?'
    ],
    'change': [
        'I just want to confirm everything we have worked on so far. You are '
        'about to change the reservation to {0}. Could you confirm?',
        'Just one more step and I will be done. Could you confirm your change '
        'to {0}?'
    ],
    'book': [
        'We have your ticket of {0}. Do I have your permission to proceed?',
        'I just want to confirm with your on flight {0}. Is this the ticket we'
        ' are going to book?'
    ]
}

customer_confirm_change = {
    'book': [
        'Yes, please go ahead and book the ticket.',
        'Confirmed. Please go ahead.'
    ],
    'change': [
        'That is correct. Please make the cahnges.',
        'Sounds good. Please proceed'
    ],
    'abort': [
        'Actually, I have changed my mind.',
        'Well, I made a second through and I have decided to keep my original '
        'reservation.'
    ],
    'cancel': ['Yes, please cancel it.', 'Yup, please cancel it.']
}

agent_conclusion_message = {
    'cancel': [
        'Your ticket has been canceled. Thank you for using our flight booking'
        ' service.',
        'Not a problem. I have canceled your ticket. Have a good day.',
        'Your ticket is canceled.'
        ' Please let us know if you have other requests.'
    ],
    'book': [
        'Your booking has been confirmed. Thank you.',
        'Your ticket has been booked. Have a nice day.',
        'We have successfully booked your trip. Thanks for using our flight '
        'booking servie.'
    ],
    'change': [
        'I have updated your reservation. Thanks for using our flight booking '
        'service.',
        'Your reservation has been updated. Please let us know if you have '
        'other requests.',
        'We have successfully updated your reservation. Have a nice day.'
    ],
    'no_flight': [
        'I can not find any flights that satisfy your requests.',
        'There is no flight that satisfies your requests.'
    ],
    'abort': [
        'You have chosen not to proceed. Thanks for using our flight booking '
        'service.', 'Not a problem. We will keep your'
        ' reservation unchanged. Have a nice day.'
    ],
    'no_res': [
        'I am sorry, but I can not locate your reservation.',
        'Your reservation can not be found in our system. Please check your '
        'account.'
    ]
}

customer_turn_prefix = 'customer:'
agent_turn_prefix = 'agent:'
secondary_error = True


class Interaction(object):
  """This class contains the dialogue interaction between the two agents."""

  def __init__(self,
               fact_obj,
               skip_greeting=0,
               fix_response_candidate=True,
               first_ask_prob=0,
               regret_prob=0,
               random_respond_error=False):
    # probability then customer will skip always greeting if he/she speaks first
    self.skip_greeting = skip_greeting
    # if true then only the first response candidate will be used.
    # otherwise randomly decided.
    self.fix_resp_candidate = fix_response_candidate
    # probability that customer ask turn will not provide information.
    # otherwise randomly decided.
    self.first_ask_prob = first_ask_prob
    # probability of regetting making changes or purchases at the end
    self.regret_prob = regret_prob
    # randomly choose unsatisfied condition to respond
    self.random_respond_error = random_respond_error
    # fact object
    self.fact_obj = fact_obj

  def get_template(self, list_of_templates):
    if self.fix_resp_candidate:
      choice = 0
    else:
      choice = utils.choice(len(list_of_templates))
    return list_of_templates[choice]

  def customer_turn(self, utterance):
    return customer_turn_prefix + ' ' + utterance

  def agent_turn(self, utterance):
    return agent_turn_prefix + ' ' + utterance

  def generate_confirmation(self, flight, full=True):
    if full:
      tmp = self.get_template(agent_suggestion['full'])
      return tmp.format(
          flight['flight_number'], flight['departure_airport'],
          flight['departure_month'] + ' ' + str(flight['departure_day']),
          utils.format_time(
              flight['departure_time_num']), flight['return_airport'],
          flight['return_month'] + ' ' + str(flight['return_day']),
          utils.format_time(flight['return_time_num']), flight['class'],
          utils.get_connection(flight['num_connections']), flight['price'])
    else:
      tmp = self.get_template(agent_suggestion['without_cd'])
      return tmp.format(
          flight['flight_number'],
          utils.format_time(flight['departure_time_num']),
          utils.format_time(flight['return_time_num']), flight['class'],
          utils.get_connection(flight['num_connections']), flight['price'])

  # this is wrong because we will need to compare all flights. use the one in
  # context generator
  def airflight_selector(self, condition, airfare_database):
    """This function selects a flight based on the condition."""
    candidates = []
    for i, flight in enumerate(airfare_database):
      cond = utils.check_condition(
          self.fact_obj, flight, condition, get_full_diff=False)
      if cond == 'satisfied':
        candidates.append(i)
    candidates.sort(key=lambda a: airfare_database[a]['price'])
    if not candidates:
      return None
    else:
      return airfare_database[candidates[0]]

  def get_flight_from_flight_number(self, flight_number, airfare_database):
    for i, flight in enumerate(airfare_database):
      if flight['flight_number'] == flight_number:
        return airfare_database[i]
    return None

  def get_message(self, ag_condition, cus_condition, erro_state):
    """This function geenrates the message based on errors."""
    if erro_state == 'satisfied':
      error = erro_state
    else:
      if self.random_respond_error:
        error_to_handle = utils.choice(list(range(len(erro_state))))
      else:
        error_to_handle = 0
      error = erro_state[error_to_handle]
    msg = self.get_template(customer_recurring_respond[error])
    if error != 'satisfied':
      ag_condition[error] = cus_condition[error]

    if error not in set(['airline_preference', 'satisfied']):
      return msg.format(cus_condition[error]), ag_condition, error
    else:
      return msg, ag_condition, error

  def generate_greetings(self, utterance, speaker):
    """This function generates greets."""
    # skip greeting
    if speaker == 0 and np.random.random() < self.skip_greeting:
      return utterance
    pair = self.get_template(greeting_pairs)
    if speaker == 0:
      utterance.append(self.customer_turn(pair[0]))
      utterance.append(self.agent_turn(pair[1]))
    else:
      utterance.append(self.agent_turn(pair[0]))
      utterance.append(self.customer_turn(pair[1]))
    return utterance

  def generate_agent_ask(self, utterance):
    utterance.append(self.agent_turn(self.get_template(agent_ask)))
    return utterance

  def generate_customer_request(self, customer_condition, agent_condition,
                                utterance):
    """This function generates customer requests."""
    # first get the goal index (buy, cancel, change)
    goal_index = customer_condition['goal']
    goal_str = self.fact_obj.goal_str_arr[goal_index]
    # get the template
    template = self.get_template(customer_request[goal_str])
    # by some probability customer will say departure and return
    # city in the request statement
    first_ask = np.random.random() < self.first_ask_prob
    if first_ask:
      deprture_city, return_city = customer_condition[
          'departure_airport'], customer_condition['return_airport']
      ask = 'from ' + deprture_city + ' to ' + return_city
      # set agent_condition
      agent_condition['departure_airport'] = deprture_city
      agent_condition['return_airport'] = return_city
    else:
      # otherwise utt will just be empty and generaet
      # a request without filling information
      ask = ''
    utt = template.format(ask)
    utterance.append(self.customer_turn(utt))
    agent_condition['goal'] = customer_condition['goal']
    return utterance, agent_condition, goal_str

  def no_reservation(self, utterance):
    agent_confirm_utt = self.get_template(agent_conclusion_message['no_res'])
    utterance.append(self.agent_turn(agent_confirm_utt))
    return utterance

  def cancel_reservation(self, utterance):
    agent_utt = self.get_template(agent_confirm_change['cancel'])
    utterance.append(self.agent_turn(agent_utt))
    customer_utt = self.get_template(customer_confirm_change['cancel'])
    utterance.append(self.customer_turn(customer_utt))
    agent_confirm_utt = self.get_template(agent_conclusion_message['cancel'])
    utterance.append(self.agent_turn(agent_confirm_utt))
    return utterance

  def fulfill_basic_requirement(self, goal, cus_cond, ag_cond, utterance):
    """This function fulfill bsic requirements."""
    # check departure and return city
    if 'departure_airport' not in ag_cond:
      assert 'return_airport' not in ag_cond
      # print ('goal',goal)
      ask_cities = self.get_template(agent_first_respond[goal]['c1c2'])
      utterance.append(self.agent_turn(ask_cities))
      depart = cus_cond['departure_airport']
      ret = cus_cond['return_airport']
      respond_cities = self.get_template(customer_first_respond['c1c2']).format(
          depart, ret)
      utterance.append(self.customer_turn(respond_cities))
      ag_cond['departure_airport'] = depart
      ag_cond['return_airport'] = ret
    if 'departure_month' not in ag_cond:
      assert 'departure_day' not in ag_cond
      assert 'return_month' not in ag_cond
      assert 'return_day' not in ag_cond
      ask_date = self.get_template(agent_first_respond[goal]['d1d2'])
      utterance.append(self.agent_turn(ask_date))
      d_m, d_d = cus_cond['departure_month'], cus_cond['departure_day']
      a_m, a_d = cus_cond['return_month'], cus_cond['return_day']
      dep = d_m + ' ' + str(d_d)
      ar = a_m + ' ' + str(a_d)
      respond_date = self.get_template(customer_first_respond['d1d2']).format(
          dep, ar)
      utterance.append(self.customer_turn(respond_date))
      ag_cond['departure_month'] = d_m
      ag_cond['departure_day'] = d_d
      ag_cond['return_month'] = a_m
      ag_cond['return_day'] = a_d
    return ag_cond, utterance

  def continue_booking(self, cus_cond, ag_cond, kb, utterance):
    """This function books the flight."""
    goal = ag_cond['goal']
    goal_str = self.fact_obj.goal_str_arr[goal]
    ag_cond, utterance = self.fulfill_basic_requirement(goal_str, cus_cond,
                                                        ag_cond, utterance)
    status = None
    first_time = True
    error = 'basic'
    while not status:
      flight = self.airflight_selector(ag_cond, kb)
      if not flight:  # terminal condition
        utterance.append(
            self.agent_turn(
                self.get_template(agent_conclusion_message['no_flight'])))
        if secondary_error:
          status = 'no_flight' '_' + error
        else:
          status = 'no_flight'

        return utterance, status, flight
      else:
        utterance.append(
            self.agent_turn(self.generate_confirmation(flight, first_time)))
        first_time = False
        condition = utils.check_condition(self.fact_obj, flight, cus_cond)
        msg, ag_cond, error = self.get_message(
            ag_cond, cus_cond, condition)  # will do merge condition within
        utterance.append(self.customer_turn(msg))
        if condition == 'satisfied':
          goal_int = cus_cond['goal']
          status = self.fact_obj.goal_str_arr[goal_int]  # either book or change
    # status is either book, change, or potenitally abort
    ask_confirm = self.get_template(agent_confirm_change[status]).format(
        flight['flight_number'])
    utterance.append(self.agent_turn(ask_confirm))
    regret = np.random.random() < self.regret_prob
    if regret:
      status = 'abort'
    cus_re_conrim = self.get_template(customer_confirm_change[status])
    utterance.append(self.customer_turn(cus_re_conrim))
    agent_conclusion = self.get_template(agent_conclusion_message[status])
    utterance.append(self.agent_turn(agent_conclusion))
    return utterance, status, flight

  def generate_dialogue(self, customer, knowledge_base):
    """This function is the main entry of the dialogue generation logic."""
    airfare_database = knowledge_base.get_json()['kb']
    reservation = knowledge_base.get_json()['reservation']
    utterance = []
    # 0a. decides who speaks first 0--customer, 1--agent
    speaker = int(np.random.random() < 0.5)
    # 0b. generate customer's full condition and agent_condition
    customer_condition = customer.get_customer_condition()
    agent_condition = {}
    # 1. greetings
    utterance = self.generate_greetings(utterance, speaker)
    # 2. generate agent's utterance to ask for request if
    # customer finished the last turn
    if speaker == 1:
      utterance = self.generate_agent_ask(utterance)
    # 3. generate customer request
    utterance, agent_condition, goal_str = self.generate_customer_request(
        customer_condition, agent_condition, utterance)

    # 4 ask for name first
    ask_name_utt = self.get_template(agent_ask_name)
    utterance.append(self.agent_turn(ask_name_utt))
    answer_name_utt = self.get_template(cutomer_name).format(
        customer_condition['name'].replace('_', ' '))
    utterance.append(self.customer_turn(answer_name_utt))

    # 5. fulfill basic requirement
    # (departure/return city, departure/return month/day)

    if goal_str == 'book' or (goal_str == 'change' and reservation != 0):
      # status can be book, change, no_flight, abort
      utterance, status, flight = self.continue_booking(
          customer_condition, agent_condition, airfare_database, utterance)
    elif goal_str in ['change', 'cancel'] and reservation == 0:
      utterance = self.no_reservation(utterance)
      status = 'no_reservation'
      flight = None
    elif goal_str == 'cancel':
      utterance = self.cancel_reservation(utterance)
      status = 'cancel'
      flight = None

    # per new format we will need to return a flight arr to consider the
    # situation where multiple cheapest flights are available. As in the real
    # data here we only choose at most one flight. However, the
    # expected_action should be able to contain multiple flights.
    flight_arr = []
    if flight:
      flight_arr.append(flight)
    # use the generate action function from utils. It will standarlize
    # the action
    # print(customer_condition['name'])
    name = customer_condition['name'].replace('_', ' ')
    action = utils.generate_action(flight_arr, name, status)

    # print utterance
    return utterance, action, status

  # def generate_action(self, flight, name, status):
  #   action_json = {}
  #   if status.startswith('no_flight'):
  #     status = 'no_flight'
  #   action_json['status'] = status
  #   if status in ['change', 'book']:
  #     action_json['flight'] = flight['flight_number']
  #     action_json['name'] = name
  #   return action_json
