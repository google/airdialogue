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
"""library to standardize data."""

from tensorflow.compat.v1.io import gfile
from tqdm import tqdm
import string
import json

printable = set(string.printable)


def add_dot(utt):
  if utt.strip()[-1] != '.' and utt.strip()[-1] != '?':
    return utt.strip() + '.'
  else:
    return utt.strip()


def standardize_message(utterances, time_stamp=None):
  """this function combines adjacent utternaces that belong to the same talker

     into one. Sometimes time_stamp could be None.
     For example
        <t1> how are you. <t2> I am good. <t2> And you? <eod> <t1>
     will be combined into
        <t1> how are you. <t2> I am good. And you? <eod> <t1>
  """
  new_utterance = []
  new_time_stamp = []
  for i, utt in enumerate(utterances):
    if len(utt.strip()) == 0:
      continue
    utts = utt.split(':')
    talker = utts[0]
    sentence = ':'.join(utts[1:]).strip()
    if len(sentence) == 0:
      continue
    if len(new_utterance) == 0 or talker != new_utterance[-1].split(':')[0]:
      new_utterance.append(add_dot(utt))
      if time_stamp:
        new_time_stamp.append(time_stamp[i])
    else:
      new_utterance[-1] += ' ' + add_dot(sentence)
      if time_stamp:
        new_time_stamp[-1] = time_stamp[i]
  return new_utterance, new_time_stamp


def delete_non_ascii(s):
  return ''.join([x for x in s if x in printable])


def load_and_drop(data_file, kb_file, drop_incorrect=True, verbose=False):
  """ this function filter incorrect samples without standardization."""
  fin_data = gfile.GFile(data_file)
  fin_kb = gfile.GFile(kb_file)
  total_in_file = 0
  loaded_data = []
  loaded_kb = []
  for line1 in tqdm(fin_data, desc='loading data'):
    if len(line1.strip()) < 10:
      continue
    line2 = fin_kb.readline()
    if len(line2.strip()) < 10:
      continue
    line1 = delete_non_ascii(line1)
    line2 = delete_non_ascii(line2)

    data_obj = json.loads(line1)
    kb_obj = json.loads(line2)
    if (not drop_incorrect) or (
        'correct_sample' not in data_obj) or data_obj['correct_sample']:
      loaded_data.append(data_obj)
      loaded_kb.append(kb_obj)
    total_in_file += 1

  if verbose:
    print(('loaded: ', len(loaded_data), '/', total_in_file, '=',
           len(loaded_data) * 1.0 / total_in_file))
  return loaded_data, loaded_kb


def load_and_drop_stream(data_file,
                         kb_file,
                         drop_incorrect=True,
                         verbose=False):
  """ this function filter incorrect samples without standardization."""
  if verbose:
    print('loading stream')
  fin_data = gfile.GFile(data_file)
  if gfile.exists(kb_file):
    fin_kb = gfile.GFile(kb_file)
  else:
    fin_kb = None
  if verbose:
    print('gfile loaded: ', fin_data)
  for line1 in fin_data:
    if verbose:
      print(line1)
    if len(line1.strip()) < 10:
      continue
    line1 = delete_non_ascii(line1)
    data_obj = json.loads(line1)

    if fin_kb:
      line2 = fin_kb.readline()
      if len(line2.strip()) < 10:
        continue
      line2 = delete_non_ascii(line2)
      kb_obj = json.loads(line2)
    else:
      kb_obj = None
    if (not drop_incorrect) or (
        'correct_sample' not in data_obj) or data_obj['correct_sample']:
      yield data_obj, kb_obj


def standardize_and_drop(data_file,
                         kb_file,
                         drop_incorrect=True,
                         verbose=False):
  """ this function filter incorrect samples and standardize them

   the same time.
  """
  loaded_data, loaded_kb = load_and_drop(data_file, kb_file, drop_incorrect,
                                         verbose)
  for data_obj in tqdm(loaded_data, desc='standardizing data'):
    org_time = data_obj['timestamps'] if 'timestamps' in data_obj else None
    org_diag = data_obj['dialogue'] if 'dialogue' in data_obj else None
    if org_diag:
      new_diag, new_time = standardize_message(org_diag, org_time)
      data_obj['dialogue'] = new_diag
      if new_time:
        data_obj['timestamps'] = new_time
        assert len(data_obj['dialogue']) == len(data_obj['timestamps'])
  return loaded_data, loaded_kb
