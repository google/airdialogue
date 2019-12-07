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

"""This file only tokenize one file at a time."""

import argparse
import os
import nltk
from tensorflow import gfile
import tensorflow as tf

from airdialogue.prepro.tokenize_lib import list_of_action_tokens_except_name
from airdialogue.prepro.tokenize_lib import process_kb
from airdialogue.prepro.tokenize_lib import process_main_data
from airdialogue.prepro.tokenize_lib import word_tokenize
from airdialogue.prepro.tokenize_lib import write_cat
from airdialogue.prepro.tokenize_lib import write_completion
from airdialogue.prepro.tokenize_lib import write_data
from airdialogue.prepro.tokenize_lib import write_self_play
from airdialogue.prepro.tokenize_lib import write_vocabulary
# Standardization libs
from airdialogue.prepro.standardize_data_lib import standardize_and_drop
from airdialogue.prepro.standardize_data_lib import load_and_drop
import sys
FLAGS = None

def add_arguments(parser):
  '''Build ArgumentParser.'''
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument('--word_cutoff', type=int, default=0,
                      help='number of candidate airports')
  parser.add_argument('--data_file', type=str, default=None,
                      help='path for data_file')
  parser.add_argument('--kb_file', type=str, default=None,
                      help='path for kb_file')
  parser.add_argument('--output_prefix', type=str, default='train',
                      help='prefix of output file')
  parser.add_argument('--output_dir', type=str, default=None,
                      help='output dir')
  parser.add_argument('--input_type', type=str, default='dialogue',
                      help='dialogue/context')
  parser.add_argument('--job_type', type=str, default='1|1|1|0|0',
                      help='train|eval|infer|sp-train|sp-eval')
  parser.add_argument('--nltk_data', type=str, default=None,
                      help='path to NLTK data')
  parser.add_argument('--verbose', type='bool', nargs='?', const=True,
                      default=False,
                      help='if enabled, debug info will be printed out.')
  parser.add_argument('--skip_standardize', type='bool', nargs='?', const=True,
                      default=False,
                      help='''if enabled, dialogues will skip the
                              standardization process, which includes
                              removeing duplicated end of sentence tokens and
                              filtering out non-ascii characters.''')
  parser.add_argument('--keep_incorrect', type='bool', nargs='?', const=True,
                      default=False,
                      help='''if enabled, incorrect dialogues will not be
                              filtered out. In stead, they will be generated
                              along with the correct ones. ''')
  parser.add_argument('--gen_cat', type='bool', nargs='?', const=True,
                      default=False,
                      help='if enabled, category files will be generated.')
  parser.add_argument('--gen_voc', type='bool', nargs='?', const=True,
                      default=False,
                      help='if enabled, vocabulary file will be generated.')
  parser.add_argument('--gen_voc_map', type='bool', nargs='?', const=True,
                      default=False,
                      help='if enabled, vocabulary map with counting will be generated.')
  parser.add_argument('--gen_special_token', type='bool', nargs='?', const=True,
                      default=False,
                      help='if enabled, special token file will be generated.')
  parser.add_argument('--keep_non_ascii', type='bool', nargs='?', const=True,
                      default=False,
                      help='if enabled, non-ascii tokens will not be droped in vocabulary')


def generate_entry(intents, actions, expected_actions, dialogues, kbs,
                   boundaries1, boundaries2, i):
  """generate entry based on the available of data."""
  # required fields
  entry = {'intent': intents[i], 'action': actions[i], 'kb': kbs[i]}
  # optional fields depending on the input type
  if dialogues:
    entry['dialogue'] = dialogues[i]
  if boundaries1:
    entry['boundaries1'] = boundaries1[i]
  if boundaries2:
    entry['boundaries2'] = boundaries2[i]
  if expected_actions:
    entry['expected_action'] = expected_actions[i]
  return entry


def reorganize_data(intents, actions, expected_actions, dialogues, kbs,
                    boundaries1, boundaries2):
  """reorganize data into compact form.

  This is mostly because we want to reuse
  code for the random split version.
  """
  data = []
  for i in xrange(len(intents)):
    entry = generate_entry(intents, actions, expected_actions, dialogues, kbs,
                           boundaries1, boundaries2, i)
    data.append(entry)
  return data


def process_job_type(job_type_str, input_type):
  """process job type string."""
  all_jobs = []
  job_type_arr = job_type_str.split('|')
  assert len(job_type_arr) == 5
  for i in range(5):
    if int(job_type_arr[i]) == 1:
      all_jobs.append('train|eval|infer|sp-train|sp-eval'.split('|')[i])
  assert input_type in ['dialogue', 'context']
  # if input type is context, we can't do train/eval/infer
  if input_type == 'context':
    assert 'train' not in all_jobs
    assert 'eval' not in all_jobs
    assert 'infer' not in all_jobs
  return all_jobs


def main(FLAGS):
  all_jobs = process_job_type(FLAGS.job_type, FLAGS.input_type)
  output_dir = FLAGS.output_dir
  if FLAGS.verbose:
    print 'all_jobs', all_jobs
    print 'input_type', FLAGS.input_type
    print 'output_dir', output_dir
    print 'data_file', FLAGS.data_file
    print 'kb_file', FLAGS.kb_file
    print 'output_prefix', FLAGS.output_prefix
    print 'skip_standardize', FLAGS.skip_standardize
    print 'keep_incorrect', FLAGS.keep_incorrect
    print 'word_cutoff', FLAGS.word_cutoff
    print 'gen_voc', FLAGS.gen_voc

  if not tf.io.gfile.isdir(output_dir):
    gfile.MkDir(output_dir)

  input_data_file = FLAGS.data_file
  input_kb_file = FLAGS.kb_file
  if len(FLAGS.output_prefix.strip()) == 0:
    FLAGS.output_prefix = ''
  else:
    FLAGS.output_prefix = FLAGS.output_prefix
  # output_vab = output_dir + '/{0}.vocab'.format(FLAGS.output_prefix)
  output_vab = output_dir + '/vocab.txt'
  output_all_vab = output_dir + '/{0}.full.vocab'.format(FLAGS.output_prefix)
  all_token_file = output_dir + '/{0}.special.vocab'.format(
      FLAGS.output_prefix)
  first_name_cats_file = output_dir + '/{0}.firstname.cat'.format(
      FLAGS.output_prefix)
  last_name_cats_file = output_dir + '/{0}.lastname.cat'.format(
      FLAGS.output_prefix)
  flight_cats_file = output_dir + '/{0}.flight.cat'.format(
      FLAGS.output_prefix)
  status_cats_file = output_dir + '/{0}.status.cat'.format(
      FLAGS.output_prefix)

  output_data_pattern = output_dir + '/{0}data'
  output_kb_pattern = output_dir + '/{0}kb'

  nltk_path = FLAGS.ntlk_data
  nltk.data.path.append(nltk_path)
  sent_tokenize = nltk.sent_tokenize

  vocal_map = {}
  # load data and do standardization
  if not FLAGS.skip_standardize:
    raw_data, raw_kb = standardize_and_drop(
        input_data_file,
        input_kb_file,
        drop_incorrect=not FLAGS.keep_incorrect,
        verbose=FLAGS.verbose)
  else:
    raw_data, raw_kb = load_and_drop(
        input_data_file,
        input_kb_file,
        drop_incorrect=not FLAGS.drop_incorrect,
        verbose=FLAGS.verbose)
  # has to be there no matter what
  if FLAGS.verbose: print 'processing kb'
  processed_kb, vocal_map = process_kb(raw_kb, vocal_map)
  # if dialogue, everything will be there.
  # if context, only intents, actions, vocal_map will be there
  if FLAGS.verbose: print 'processing data'
  result = process_main_data(
      raw_data,
      sent_tokenize,
      word_tokenize,
      vocal_map,
      input_type=FLAGS.input_type)
  intents, actions, expected_actions, dialogues, vocal_map, boundaries1, boundaries2, cats = result
  frequency_cutoff = FLAGS.word_cutoff
  # 3 is the number of special tokens
  if FLAGS.verbose: print 'vocabulary before cutoff', len(vocal_map) + 3
  if not FLAGS.gen_voc:
    # if we choose not to generate vocabulary file, we set output_vab to None
    output_vab = None
  if not FLAGS.gen_voc_map:
    output_all_vab = None
  vocal_map = write_vocabulary(output_vab, output_all_vab, vocal_map,
                               frequency_cutoff, FLAGS.keep_non_ascii)
  if FLAGS.gen_cat:
    if FLAGS.verbose: print 'writing category'
    cat_files = [
        first_name_cats_file, last_name_cats_file, flight_cats_file,
        status_cats_file
    ]
    write_cat(cat_files, cats)

  if FLAGS.verbose:
    print 'frequency_cutoff= {0}, vocabulary after cutoff'.format(
        frequency_cutoff), len(vocal_map)
  data = reorganize_data(intents, actions, expected_actions, dialogues,
                         processed_kb, boundaries1, boundaries2)

  if 'train' in all_jobs:
    if FLAGS.verbose:
      print 'writing train data'
    write_data(data, output_data_pattern.format(FLAGS.output_prefix + '.'),
               output_kb_pattern.format(FLAGS.output_prefix + '.'))
  if 'eval' in all_jobs:
    if FLAGS.verbose:
      print 'writing eval data'
    write_data(data, output_data_pattern.format(FLAGS.output_prefix + '.eval.'),
               output_kb_pattern.format(FLAGS.output_prefix + '.eval.'))
  if 'infer' in all_jobs:
    if FLAGS.verbose: print 'writing infer data'
    write_completion(
        data, output_data_pattern.format(FLAGS.output_prefix + '.infer.src.'),
        output_data_pattern.format(FLAGS.output_prefix + '.infer.tar.'),
        output_kb_pattern.format(FLAGS.output_prefix+ '.infer.'))
  if 'sp-train' in all_jobs:
    if FLAGS.verbose: print 'writing self play training data'
    write_self_play(
        data,
        output_data_pattern.format(FLAGS.output_prefix + '.selfplay.'),
        output_kb_pattern.format(FLAGS.output_prefix + '.selfplay.'))
  if 'sp-eval' in all_jobs:
    if FLAGS.verbose: print 'writing self play eval data'
    write_self_play(
        data,
        output_data_pattern.format(FLAGS.output_prefix + '.selfplay.eval.'),
        output_kb_pattern.format(FLAGS.output_prefix + '.selfplay.eval.'))

  if FLAGS.gen_special_token:
    # write all token file.
    f_tokens = gfile.Open(all_token_file, 'w')
    for token in list(list_of_action_tokens_except_name):
      f_tokens.write(token + '\n')
    f_tokens.close()


def run_main(unused):
  main(FLAGS)


if __name__ == '__main__':
  this_parser = argparse.ArgumentParser()
  add_arguments(this_parser)
  FLAGS, unparsed = this_parser.parse_known_args()
  # print FLAGS
  tf.app.run(main=run_main, argv=[sys.argv[0]] + unparsed)
