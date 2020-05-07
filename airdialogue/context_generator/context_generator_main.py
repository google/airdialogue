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

"""This is the main module that generates contexts."""

import argparse
from airdialogue.context_generator import context_generator_lib
import tensorflow.compat.v1 as tf
import sys

FLAGS= None

def add_arguments(parser):
  """Build ArgumentParser."""
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--num_candidate_airports",
      type=int,
      default=3,
      help="number of candidate airports")
  parser.add_argument(
      "--book_window", type=int, default=2, help="window between bookings")
  parser.add_argument(
      "--num_db_record",
      type=int,
      default=30,
      help="number of database records per booking.")
  parser.add_argument(
      "--firstname_file",
      type=str,
      default="./data/resources/meta_context/first_names.txt",
      help="text file that contains a list of first names.")
  parser.add_argument(
      "--lastname_file",
      type=str,
      default="./data/resources/meta_context/last_names.txt",
      help="text file that contains a list of last names.")
  parser.add_argument(
      "--airportcode_file",
      type=str,
      default="./data/resources/meta_context/airport.txt",
      help="text file that contains a list of airport codes.")
  parser.add_argument(
      "--num_samples",
      type=int,
      default=320000,
      help="number of samples to generate.")
  parser.add_argument(
      "--output_data",
      type=str,
      default=None,
      help="path of the output data file.")
  parser.add_argument(
      "--output_kb", type=str, default=None, help="path of the output kb file.")
  parser.add_argument(
      "--display_freq",
      type=int,
      default=50000,
      help="display frequency for information.")
  parser.add_argument(
      "--verbose",
      type="bool",
      nargs="?",
      const=True,
      default=False,
      help="if enabled, debug info will be printed out.")


def main(FLAGS):
  cg = context_generator_lib.ContextGenerator(
      num_candidate_airports=FLAGS.num_candidate_airports,
      book_window=FLAGS.book_window,
      num_db_record=FLAGS.num_db_record,
      firstname_file=FLAGS.firstname_file,
      lastname_file=FLAGS.lastname_file,
      airportcode_file=FLAGS.airportcode_file)

  _, stats = cg.generate_context(FLAGS.num_samples,
                                 output_data=FLAGS.output_data,
                                 output_kb=FLAGS.output_kb,
                                 display_freq=FLAGS.display_freq,
                                 verbose=FLAGS.verbose)
  if FLAGS.verbose: print(stats)


def run_main(unused):
  main(FLAGS)

if __name__ == "__main__":
  this_parser = argparse.ArgumentParser()
  add_arguments(this_parser)
  FLAGS, unparsed = this_parser.parse_known_args()
  tf.app.run(main=run_main, argv=[sys.argv[0]] + unparsed)


