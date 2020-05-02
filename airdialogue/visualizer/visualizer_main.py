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

"""This module builds a falsk server for visualization."""

import argparse
import json
import linecache
import os
from os.path import expanduser
from flask import Flask
from flask import request
from airdialogue.visualizer.utils import generate_html
import sys

def strip_prefix(name):
  if name.endswith("data"):
    return name[0:-5]
  else:
    return name[0:-3]


def get_partitions(path):
  """This function counts the number of occurries of the same prefix in the
     json dir. If it happens more than twice we will use that as a valid
     partition and add it to the partition list."""
  all_files = os.listdir(path)
  prefix_freq = {}
  for f in all_files:
    if f.endswith(".json"):
      prefix = f.split(".")[0]
      prefix = strip_prefix(prefix)
      if prefix not in prefix_freq:
        prefix_freq[prefix] = 0
      prefix_freq[prefix] += 1
  valid_partitions = []
  for prefix in prefix_freq:
    if prefix_freq[prefix] >= 2:
      valid_partitions.append(prefix)
  return valid_partitions


def wrapper(FLAGS):
  def home():
    # get all the partitions in the directory
    expanded_data_path = expanduser(FLAGS.data_path)
    partitions = get_partitions(expanded_data_path)
    index = request.form.get("index")
    if index:
      index = int(index)
    partition = request.form.get("partition")

    if not partitions:
      print("no data is found in the directory")
      return """No partitions found under {0}. Supported partitions has to end
                with .json extension.""".format(FLAGS.data_path)
    if not partition:
      # choose a default partition
      if "train" in partitions:
        partition = "train"
      else:
        partition = partitions[0].strip()
      index = 1

    try:
      line_data = linecache.getline(
          os.path.join(expanded_data_path, "{0}_data.json".format(partition)),
          index)
      line_kb = linecache.getline(
          os.path.join(expanded_data_path, "{0}_kb.json".format(partition)),
          index)
    except:
      return "Invalid index."

    if (not line_data) and (not line_kb):
      print("invalid partition number.")
    data_object = json.loads(line_data)
    kb_object = json.loads(line_kb)
    html_source = generate_html(data_object, kb_object, index, partitions,
                                partition)
    return html_source
  return home


def add_arguments(parser):
  """Build ArgumentParser."""
  parser.add_argument("--host", type=str, default="0.0.0.0",
                      help="host name for the visualizer.")
  parser.add_argument("--port", type=int, default=5555,
                      help="port number of the visualizer server.")
  parser.add_argument("--data_path", type=str, default=None,
                      help="path that stores data and kb files.")


def main(FLAGS):
  app = Flask(__name__)
  app.route("/", methods=["POST", "GET"])(wrapper(FLAGS))
  app.run(host=FLAGS.host, port=FLAGS.port, debug=True)


if __name__ == "__main__":
  this_parser = argparse.ArgumentParser()
  add_arguments(this_parser)
  FLAGS, unparsed = this_parser.parse_known_args()
  main(FLAGS)
