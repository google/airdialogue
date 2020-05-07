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

"""This is the entry of the flight simulator generator."""



import copy
import json
import random
import numpy as np
from tensorflow.compat.v1 import gfile
from airdialogue.context_generator.src import customer
from airdialogue.context_generator.src import facts
from airdialogue.context_generator.src import kb as knowledgebase
from airdialogue.context_generator.src import utils


class ContextGenerator(object):
  """context generator."""

  def __init__(self, num_candidate_airports, book_window, num_db_record,
               firstname_file, lastname_file, airportcode_file):
    self.fact_obj = facts.Facts(firstname_file, lastname_file, airportcode_file)
    # number of airports to be considered when generating database.
    self.num_candidate_airports = num_candidate_airports
    # number of days between departuer and arrival
    self.book_window = book_window
    # number of database records
    self.num_db_record = num_db_record

  def _generate_user_action(self, expected_action):
    user_action = copy.deepcopy(expected_action)
    if len(user_action["flight"]) != 0:
      idx = random.randint(0, len(user_action["flight"]) - 1)
      single_flight = user_action["flight"][idx]
      user_action["flight"] = [single_flight]
    return user_action

  def generate_context(self,
                       num_context,
                       output_data=None,
                       output_kb=None,
                       display_freq=None,
                       output_object=False,
                       verbose=False):
    """generate context. if output_file is not none then we write to file."""
    if output_data and output_kb:
      fp_data = gfile.Open(output_data, "w")
      fp_kb = gfile.Open(output_kb, "w")
    else:
      fp_data = None
      fp_kb = None
    all_context = []
    status_stats = {}
    for n in range(num_context):
      airport_candidate = list(
          np.random.choice(
              self.fact_obj.airport_list,
              self.num_candidate_airports,
              replace=False))
      if display_freq:
        if verbose and n % display_freq == 0:
          print((n, "/", num_context))
      cus = customer.Customer(self.fact_obj, self.book_window,
                              airport_candidate)
      ref_departure_date, ref_return_date = cus.departure_date, cus.return_date
      kb = knowledgebase.Knowledgebase(self.fact_obj, self.num_db_record,
                                       airport_candidate, ref_departure_date,
                                       ref_return_date)
      intent_json = utils.standardize_intent(cus.get_json())
      kb_and_res_json = kb.get_json()
      kb_json = kb_and_res_json["kb"]
      res_json = kb_and_res_json["reservation"]
      expected_action = utils.generate_expected_action(
          self.fact_obj, intent_json, kb_json, res_json)
      status = expected_action["status"]
      if status not in status_stats:
        status_stats[status] = 0
      status_stats[status] += 1

      if fp_data and fp_kb:
        context_data = {
            "intent": intent_json,
            "action": self._generate_user_action(expected_action),
            "expected_action": expected_action
        }
        context_kb = {"kb": kb_json, "reservation": res_json}

        fp_data.write(json.dumps(context_data) + "\n")
        fp_kb.write(json.dumps(context_kb) + "\n")
      else:
        if output_object:  # output pythoh object instead of json
          object_structure = (cus, kb, expected_action)
          all_context.append(object_structure)
        else:  # output json
          context_json = {
              "intent": intent_json,
              "kb": kb_json,
              "reservation": res_json,
              "action": self._generate_user_action(expected_action),
              "expected_action": expected_action
          }
          all_context.append(context_json)

    if fp_data:
      fp_data.close()
    if fp_kb:
      fp_kb.close()
    # status stats
    for key in status_stats:
      status_stats[key] /= 1.0 * num_context

    return all_context, status_stats
