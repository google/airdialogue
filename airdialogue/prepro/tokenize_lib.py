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
"""library file for tokenize."""

import nltk
import numpy as np
from tensorflow.compat.v1 import gfile
from tqdm import tqdm

start_of_turn1 = '<t1>'
start_of_turn2 = '<t2>'
end_of_dialogue = '<eod>'
unk_token = '<unk>'
list_of_action_tokens_except_name = set([])


def tokenize_kb(kb_json):
  """This function tokenizes the knowledge base json."""

  def tokenize_one_flight(flight_json):
    """this function tokenize one flight entry in the knowledge base."""
    a1, a2 = flight_json['departure_airport'], flight_json['return_airport']
    m1, m2 = flight_json['departure_month'], flight_json['return_month']
    d1, d2 = flight_json['departure_day'], flight_json['return_day']
    tn1, tn2 = flight_json['departure_time_num'], flight_json['return_time_num']
    cl = flight_json['class']
    pr = flight_json['price']
    cn = flight_json['num_connections']
    al = flight_json['airline']
    fl = flight_json['flight_number']
    # add tags
    a1, a2, m1, m2 = format_tag('a1', a1), format_tag('a2', a2), format_tag(
        'm1', m1), format_tag('m2', m2)
    d1, d2, tn1, tn2 = format_tag('d1', d1), format_tag('d2', d2), format_tag(
        'tn1', tn1), format_tag('tn2', tn2)
    cl = format_tag('cl', cl)
    pr = format_tag('pr', pr)
    cn = format_tag('cn', cn)
    al = format_tag('al', al)
    fl = format_tag('fl', fl)
    # combine
    arr = [a1, a2, m1, m2, d1, d2, tn1, tn2, cl, pr, cn, al, fl]
    for element in arr:
      list_of_action_tokens_except_name.add(element)
    return ' '.join(arr)

  if kb_json['reservation'] == 0:
    res = 'no_res'
  else:
    res = 'has_res'
  res = format_tag('res', res)
  flight_arr = []
  for flight in kb_json['kb']:
    tokenized_flight = tokenize_one_flight(flight)
    flight_arr.append(tokenized_flight)
  list_of_action_tokens_except_name.add(res)
  return res + ' ' + ' '.join(flight_arr)


def process_kb(raw_kb, word_map, stream=False):
  """main entry to process kb."""
  processed_data = []
  d = raw_kb
  if not stream:
    d = tqdm(raw_kb, desc='process kb')
  for kb_object in d:
    # the database will be flattened into a single sequence of tokens.
    flattened = tokenize_kb(kb_object)
    processed_data.append(flattened)
    word_map = apply_word_map(flattened, word_map)
  return processed_data, word_map


def flatten_json(obj):
  if isinstance(obj, dict):
    new_list = sorted(list(obj.items()), key=lambda a: a[0])
    res_list = []
    for key, value in new_list:
      new_val = flatten_json(value)
      res_list.append(key + ' : ' + str(new_val))
    return ' . '.join(res_list)
  if isinstance(obj, list):
    new_list = []
    for x in obj:
      new_list.append(flatten_json(x))
    return ' . '.join(new_list)
    # return sorted(order_json(x) for x in obj)
  else:
    return obj


def apply_word_map(flattened_sent, word_map):
  words = flattened_sent.split(' ')
  for w in words:
    if w not in word_map:
      word_map[w] = 1
    else:
      word_map[w] += 1
  return word_map


def format_tag(name, val):
  return '<{0}_{1}>'.format(str(name), str(val))


def get_full_intent(intent_json):
  """recovers the full intent json from standalized intent.

  Basically we will add fields that are omitted becauase their values are
  all or 2 back to the intent json.
  """
  # dep/ret time
  if 'departure_time' not in intent_json:
    intent_json['departure_time'] = 'all'
  if 'return_time' not in intent_json:
    intent_json['return_time'] = 'all'
  # class
  if 'class' not in intent_json:
    intent_json['class'] = 'all'
  # max_connections
  if 'max_connections' not in intent_json:
    intent_json['max_connections'] = 2
  # airline_preference
  if 'airline_preference' not in intent_json:
    intent_json['airline_preference'] = 'all'
  return intent_json


def add_dash_to_name(original_name):
  return '_'.join(original_name.split(' '))


def tokenize_intent(intent_json):
  """This function tokenizes intents."""
  a1, a2 = intent_json['departure_airport'], intent_json['return_airport']
  m1, m2 = intent_json['departure_month'], intent_json['return_month']
  d1, d2 = intent_json['departure_day'], intent_json['return_day']
  t1, t2 = intent_json['departure_time'], intent_json['return_time']
  # nm = add_dash_to_name(intent_json['name'])
  nm1, nm2 = intent_json['name'].strip().split(' ')
  cl = intent_json['class']
  pr = intent_json['max_price']
  cn = intent_json['max_connections']
  al = intent_json['airline_preference']
  gl = intent_json['goal']
  # add tags
  a1, a2, m1, m2 = format_tag('a1', a1), format_tag('a2', a2), format_tag(
      'm1', m1), format_tag('m2', m2)
  d1, d2, t1, t2 = format_tag('d1', d1), format_tag('d2', d2), format_tag(
      't1', t1), format_tag('t2', t2)
  # nm=format_tag('nm',nm)
  cl = format_tag('cl', cl)
  pr = format_tag('pr', pr)
  cn = format_tag('cn', cn)
  al = format_tag('al', al)
  gl = format_tag('gl', gl)
  # concatenates
  arr = [a1, a2, m1, m2, d1, d2, t1, t2, nm1, nm2, cl, pr, cn, al, gl]
  return ' '.join(arr)


def tokenize_action(action_json, first_name_cat, last_name_cat, flight_cat,
                    state_cat):
  """Both name and flight will always be in the action.

  Context might have
  multiple flights in the arr but training/eval/testing data should only have
  one arr in the flight arr.
  """
  try:
    name_arr = action_json['name'].strip().split(' ')
    if len(name_arr) < 2:  # name_arr has at least one element
      nm1 = name_arr[0]
      nm2 = ''
    else:
      nm1, nm2 = name_arr
  except:
    print('name', action_json['name'])
  fl_arr = action_json['flight']
  if len(fl_arr) == 0:
    fl = 'empty'
    fl = format_tag('fl', fl)
  else:
    fl_arr_str = []
    for f in fl_arr:
      flight = format_tag('fl', str(f))
      fl_arr_str.append(flight)
      flight_cat.add(str(flight).strip())
      # fl_arr_str.append(str(f).strip())
    fl = '_'.join(fl_arr_str)
  st = action_json['status']
  # add tags
  # fl = format_tag('fl', fl)
  st = format_tag('st', st)
  arr = [nm1, nm2, fl, st]
  # list_of_action_tokens.add(nm)
  list_of_action_tokens_except_name.add(fl)
  list_of_action_tokens_except_name.add(st)
  first_name_cat.add(nm1.strip())
  last_name_cat.add(nm2.strip())
  state_cat.add(st)
  return ' '.join(arr)


# Right now expected action is not used only one flight is considered.
def process_main_data(raw_data,
                      sent_tok,
                      word_tok,
                      word_map,
                      input_type,
                      stream=False,
                      self_play_start_turn=None):
  """This function processes the main data."""

  def process_dialogue(dialogue):
    """This function processes dialogues."""

    def do_tokenize(last, turn):
      """This function tokenizes the diloagues of a specific turn."""
      if turn.startswith('customer: '):
        # agent:
        turn = turn[10:].strip()
        sot = start_of_turn1
        eot = start_of_turn2
      else:  # agent:
        turn = turn[7:].strip()
        sot = start_of_turn2
        eot = start_of_turn1
      sentences = sent_tok(turn)
      tokenized_sents = []
      for s in sentences:
        words = word_tok(s)
        tokenized_sents.append(' '.join(words))
      flat_content = ' '.join(tokenized_sents)
      if last:
        return sot + ' ' + flat_content + ' ' + end_of_dialogue + ' ' + eot
      else:
        return sot + ' ' + flat_content

    tokenized_dialogue = []
    max_turn_length = 0
    for i, turn in enumerate(dialogue):
      tokenized_turn = do_tokenize(i == len(dialogue) - 1, turn)
      max_turn_length = max(max_turn_length, len(tokenized_turn.split(' ')))
      tokenized_dialogue.append(tokenized_turn)

    return ' '.join(tokenized_dialogue), max_turn_length

  def get_dialogue_boundary(start_token, flat_dialogue):
    """This function gets the boundary array of the dialogues."""

    def get_end_token(start, set_of_end_tokens, splitted_dialogues):
      for i in range(start, len(splitted_dialogues)):
        if splitted_dialogues[i] in set_of_end_tokens:
          return i
      assert False, 'end token not found : ' + ' '.join(
          flat_dialogue) + 'start=' + str(start) + '/' + str(
              len(splitted_dialogues))

    def get_next_start_token(end_position, start_token, splitted_dialogues):
      for i in range(end_position, len(splitted_dialogues)):
        if splitted_dialogues[i] == start_token:
          return i
      return len(splitted_dialogues)

    set_of_end_tokens = set([
        start_of_turn1, start_of_turn2
    ])  # taking out end_of_dialogue token because of dynamic rnn decoder
    # set_of_end_tokens=set([start_of_turn1,start_of_turn2,end_of_dialogue])
    splitted_dialogue = flat_dialogue.split(' ')
    i = get_next_start_token(0, start_token, splitted_dialogue)
    all_starts = []
    all_ends = []
    while i < len(splitted_dialogue
                 ) - 1:  # we don't find the end token for the last turn change.
      end_position = get_end_token(i + 1, set_of_end_tokens, splitted_dialogue)
      err_msg = 'start token appeared twice: ' + flat_dialogue
      assert splitted_dialogue[end_position] != start_token, err_msg
      # all_res.append((i,end_position))
      all_starts.append(i)
      all_ends.append(end_position)
      i = get_next_start_token(i + 1, start_token, splitted_dialogue)
    return (all_starts, all_ends)

  def serialize_boundary(start, end):
    all_res = []
    for s in start:
      all_res.append(str(s))
    for e in end:
      all_res.append(str(e))
    return ' '.join(all_res)

  intents = []
  actions = []
  expected_actions = []
  dialogues = []
  boundaries1 = []
  boundaries2 = []
  lengths = []
  max_diag_length = 0
  max_turn1 = 0
  first_name_cat = set([])
  last_name_cat = set([])
  flight_cat = set([])
  state_cat = set([])

  d = raw_data
  if not stream:
    d = tqdm(raw_data, desc='process raw data')

  for loaded_json in d:
    # loaded_json = json.loads(delete_non_ascii(line))
    # input_type
    if 'intent' in loaded_json:
      intent = loaded_json['intent']
      processed_intent = tokenize_intent(get_full_intent(intent))
      intents.append(processed_intent)
      word_map = apply_word_map(processed_intent, word_map)
    else:
      intents.append('')
    if 'action' in loaded_json and loaded_json['action']:
      action = loaded_json['action']
      processed_action = tokenize_action(
          action,
          first_name_cat,
          last_name_cat,
          flight_cat,
          state_cat,
      )
      actions.append(processed_action)
      word_map = apply_word_map(processed_action, word_map)
    else:
      actions.append('')
    if 'expected_action' in loaded_json:
      expected_action = loaded_json['expected_action']
      # print "processed_intent", processed_intent
      processed_expected_action = tokenize_action(
          expected_action,
          first_name_cat,
          last_name_cat,
          flight_cat,
          state_cat,
      )
      # print "processed_action", processed_action
      expected_actions.append(processed_expected_action)

    # NB for word map updates above:
    # word map should contain everything in the supervised eval and training
    # however, it should not contain expected_action since this is for self-play
    # it contains tokens that has _, on flights.

    # process dialogue only when input_type is dialogue
    if input_type == 'dialogue':
      dialogue = loaded_json['dialogue']
      processed_dialogue, max_diag_len_this = process_dialogue(dialogue)
      max_diag_length = max(max_diag_length, max_diag_len_this)
      t1_boundaries = get_dialogue_boundary(start_of_turn1, processed_dialogue)
      t2_boundaries = get_dialogue_boundary(start_of_turn2, processed_dialogue)
      max_turn1 = max(max_turn1, len(t1_boundaries[0]) + len(t2_boundaries[0]))
      boundaries1.append(serialize_boundary(*t1_boundaries))
      boundaries2.append(serialize_boundary(
          *t2_boundaries))  # this is used only for inference files generation
      if processed_dialogue == '' and \
          self_play_start_turn in ['agent', 'customer']:
        if self_play_start_turn == 'agent':
          processed_dialogue = '<t2>'
        else:
          processed_dialogue = '<t1>'

      dialogues.append(processed_dialogue)
      length = len(processed_dialogue.split(' '))
      lengths.append(length)
      word_map = apply_word_map(processed_dialogue, word_map)

  if input_type == 'dialogue' and not stream:  #  output stats only when input is dialogue
    min_length, mean_length, max_length = np.min(lengths), np.mean(
        lengths), np.max(lengths)
    print(
        ('min_len: ${0}, mean_len: {1}, max_len: {2}'
         'max_sent_len: {3}, max_turn: {4}').format(min_length, mean_length,
                                                    max_length, max_diag_length,
                                                    max_turn1))

  # return all the processed data. Some of them will be empty arrays when in
  # context mode.
  all_cat = [first_name_cat, last_name_cat, flight_cat, state_cat]
  return intents, actions, expected_actions, dialogues, word_map, boundaries1, boundaries2, all_cat


def is_ascii(s, keep_non_ascii):
  if keep_non_ascii:
    return True
  else:
    return all(ord(c) < 128 for c in s)


def write_vocabulary(output_file, output_all_vocab_file, word_frequency,
                     frequency_cutoff, keep_non_ascii):
  special_chars = [unk_token, start_of_turn1, start_of_turn2, end_of_dialogue]
  new_word_frequency = set([])
  # if output_filei not None, we write to file, otherwise we don't.
  if output_file:
    f = gfile.Open(output_file, 'w')
  else:
    f = None

  # first one should always be unktoken
  for special_char in special_chars:
    if f:
      f.write(special_char + '\n')
  for key in word_frequency:
    # We write to the vocabulary only when the key is not empty.
    # Otherwise tensorflow will complain.
    if word_frequency[key] >= frequency_cutoff and (
        key not in special_chars) and is_ascii(key, keep_non_ascii) and key:
      if f:
        f.write(key + '\n')
      new_word_frequency.add(key)

  if f:
    f.close()
  # all vocab
  if output_all_vocab_file:
    with gfile.Open(output_all_vocab_file, 'w') as f2:
      f2.write(str(word_frequency))
  return new_word_frequency


def write_cat(files, cats):
  for file, category in zip(files, cats):
    with gfile.Open(file, 'w') as f:
      for cat in category:
        f.write(str(cat) + '\n')


def write_data(data, output_file_data, output_file_kb, alt_infer=False):
  """This function writes data into a text file."""
  f_data = gfile.Open(output_file_data, 'w')
  f_kb = gfile.Open(output_file_kb, 'w')
  for entry in data:
    f_kb.write(flatten_json(entry['kb']) + '\n')
    new_arr = []
    if alt_infer:
      new_arr = [entry['intent'], entry['dialogue'].replace('<eod> ', '')]
    else:
      new_arr = [
          entry['intent'], entry['action'], entry['dialogue'],
          entry['boundaries1']
      ]
      # only boundary1 is used but not 2 because it's not necessary.
    f_data.write('|'.join(new_arr) + '\n')
  f_data.close()
  f_kb.close()


# this needs to be fixed..., turns are randomly selected right now.
def write_completion(data, output_file_data_src, output_file_data_tar,
                     output_file_kb):
  """This function write both kb and main data into the files."""
  f_data_src = gfile.Open(output_file_data_src, 'w')
  f_data_tar = gfile.Open(output_file_data_tar, 'w')
  f_kb = gfile.Open(output_file_kb, 'w')
  for entry in data:
    bd1 = entry['boundaries1'].split(' ')
    bd2 = entry['boundaries2'].split(' ')
    start = bd1[0:len(bd1) // 2] + bd2[0:len(bd2) // 2]
    end = bd1[len(bd1) // 2:] + bd2[len(bd2) // 2:]
    # random_turn = random.randint(0, len(start) - 1)
    for random_turn in range(len(start)):
      f_kb.write(flatten_json(entry['kb']) + '\n')
      # print len(start),len(end),len(bd),random_turn
      turn_start = int(start[random_turn])
      turn_end = int(end[random_turn])
      dialogue_split = entry['dialogue'].split(' ')
      # print turn_start,turn_end
      dialogue_src = dialogue_split[0:turn_start + 1]
      dialogue_tar = dialogue_split[turn_start + 1:turn_end + 1]
      src_arr = [entry['intent'], ' '.join(dialogue_src)]
      f_data_src.write('|'.join(src_arr) + '\n')
      tar_arr = [entry['action'], ' '.join(dialogue_tar)]
      f_data_tar.write('|'.join(tar_arr) + '\n')

  f_data_src.close()
  f_data_tar.close()
  f_kb.close()


def write_self_play(data, output_file_data, output_file_kb):
  f_data = gfile.Open(output_file_data, 'w')
  f_kb = gfile.Open(output_file_kb, 'w')
  for entry in data:
    f_kb.write(flatten_json(entry['kb']) + '\n')
    new_arr = [entry['intent'],
               entry['expected_action']]  # intent and action are both needed
    f_data.write('|'.join(new_arr) + '\n')
  f_data.close()
  f_kb.close()


def word_tokenize(tokens):
  return [
      token.replace("''", '"').replace('``', '"')
      for token in nltk.word_tokenize(tokens)
  ]
