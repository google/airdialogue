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
from tensorflow.compat.v1 import gfile
import tensorflow.compat.v1 as tf

from airdialogue.prepro.tokenize_lib import list_of_action_tokens_except_name
from airdialogue.prepro.tokenize_lib import flatten_json
from airdialogue.prepro.tokenize_lib import process_kb
from airdialogue.prepro.tokenize_lib import process_main_data
from airdialogue.prepro.tokenize_lib import word_tokenize
from airdialogue.prepro.tokenize_lib import write_cat
from airdialogue.prepro.tokenize_lib import write_data
from airdialogue.prepro.tokenize_lib import write_self_play
from airdialogue.prepro.tokenize_lib import write_vocabulary
# Standardization libs
from airdialogue.prepro.standardize_data_lib import standardize_and_drop
from airdialogue.prepro.standardize_data_lib import load_and_drop
import sys
import json
FLAGS = None

def add_arguments(parser):
  '''Build ArgumentParser.'''
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument('--data_file', type=str, default=None,
                      help='path for data_file')
  parser.add_argument('--kb_file', type=str, default=None,
                      help='path for kb_file')
  parser.add_argument('--output_prefix', type=str, default='dev',
                      help='prefix of output file')
  parser.add_argument('--output_dir', type=str, default=None,
                      help='output dir')
  parser.add_argument('--verbose', type='bool', nargs='?', const=True,
                      default=False,
                      help='if enabled, debug info will be printed out.')
  parser.add_argument('--keep_incorrect', type='bool', nargs='?', const=True,
                      default=False,
                      help='''if enabled, incorrect dialogues will not be
                              filtered out. In stead, they will be generated
                              along with the correct ones. ''')

def json_dump(o):
  return json.dumps(o, separators = (',', ':'))

def write_infer_json(data, kb, output_file_src, output_file_tgt
  , output_file_kb):
  """This function write both kb and main data into the files."""
  f_src = gfile.Open(output_file_src, 'w')
  f_tgt = gfile.Open(output_file_tgt, 'w')
  f_kb = gfile.Open(output_file_kb, 'w')
  for entry, entry_kb in zip(data, kb):
    entire_dialogue = entry['dialogue'][:]

    # random_turn = random.randint(0, len(start) - 1)
    for target_turn in range(len(entire_dialogue))[1:]:
      f_kb.write(json_dump(entry_kb) + '\n')
      entry['dialogue'] = entire_dialogue[0:target_turn]
      f_src.write(json_dump(entry) + '\n')
      f_tgt.write(json_dump({'response': entire_dialogue[target_turn]}) + '\n')

  f_src.close()
  f_tgt.close()
  f_kb.close()

def main(FLAGS):
  output_dir = FLAGS.output_dir
  if FLAGS.verbose:
    print('output_dir', output_dir)
    print('data_file', FLAGS.data_file)
    print('kb_file', FLAGS.kb_file)
    print('output_prefix', FLAGS.output_prefix)

  if not tf.io.gfile.isdir(output_dir):
    gfile.MkDir(output_dir)

  input_data_file = FLAGS.data_file
  input_kb_file = FLAGS.kb_file
  if len(FLAGS.output_prefix.strip()) == 0:
    FLAGS.output_prefix = ''
  else:
    FLAGS.output_prefix = FLAGS.output_prefix

  output_data_pattern = output_dir + '/{0}data.json'
  output_kb_pattern = output_dir + '/{0}kb.json'

  # load data and do standardization
  raw_data, raw_kb = load_and_drop(
      input_data_file,
      input_kb_file,
      drop_incorrect=not FLAGS.keep_incorrect,
      verbose=FLAGS.verbose)

  write_infer_json(
      raw_data, raw_kb,
      output_data_pattern.format(FLAGS.output_prefix + '_infer_src_'),
      output_data_pattern.format(FLAGS.output_prefix + '_infer_tgt_'),
      output_kb_pattern.format(FLAGS.output_prefix+ '_infer_'))


def run_main(unused):
  main(FLAGS)


if __name__ == '__main__':
  this_parser = argparse.ArgumentParser()
  add_arguments(this_parser)
  FLAGS, unparsed = this_parser.parse_known_args()
  # print FLAGS
  tf.app.run(main=run_main, argv=[sys.argv[0]] + unparsed)
