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

"""Utility function for the visualizer."""
import json
from tensorflow.compat.v1 import gfile


def generate_kv_nested_html(nested_kv, space):
  """This function generates html for nested key-value pair structures."""
  code = """<table style="width:{0}">\n""".format(space)
  for element in nested_kv:
    sorted_list = sorted(list(element.items()), key=lambda a: a[0])
    code = code + """
    <tr>
    """
    for el in sorted_list:
      code = code + '<td>' + str(el[0]) + ',' + str(el[1]) + '</td>\n'
    code = code + """
    </tr>
    """
  code += ' </table>\n'
  return code


def format_simple_string(kv):
  """This function format a simple key-varlue pair.
  It also highlight some of the fields in intent.
  """
  sorted_list = sorted(list(kv.items()), key=lambda a: a[0])
  arr = []
  for el in sorted_list:
    key = str(el[0])
    if key in ['return_time', 'departure_time']:
      key = """<font style="color: #fff; background-color: red">""" + key + """</font>"""
    if key == 'airline_preference':
      key = """<font style="color: #fff; background-color: blue">""" + key + """</font>"""
    arr.append((key + ':' + str(el[1])))
  return ', '.join(arr)


def generate_form(partitions, sid, partition):
  """This function generate the submit form."""

  sel_html = """
  Partition <select name="partition">\n"""
  for p in partitions:
    if p == partition:
      selected = """ selected="selected" """
    else:
      selected = ''
    sel_html += """<option {0} value="{1}">{1}</option>\n""".format(selected, p)
  sel_html += '</select>\n'

  ret = """<form method="POST" action="/">""" + sel_html + '<br>'
  ret += """Sample #<input type="text" name="index" value="{0}" required><br>
            <input type="submit" value="Submit">
            </form>""".format(sid)
  return ret


def generate_res_str(reservation):
  if reservation == 0:
    return 'This customer do not have an existing reservation.'
  else:
    return 'This customer has a reservation on flight #{0}'.format(reservation)


def generate_html(sample,
                  kb_object,
                  sid,
                  partitions,
                  partition):
  """This function generates the html."""
  intent, action, true_action = sample['intent'], sample['action'], sample[
      'expected_action']
  if 'dialogue' in sample:
    dialogue = sample['dialogue']
  else:
    dialogue = None
  db = kb_object['kb']
  reservation = kb_object['reservation']
  res_str = generate_res_str(reservation)
  # print ('reservation', reservation)
  if 'search_info' in sample:
    search_info = sample['search_info']
    if type(search_info) == str:
      search_info = json.loads(search_info)
  else:
    search_info = None
  # print('intent', intent.keys())
  intent_table = format_simple_string(intent)
  action_table = format_simple_string(action)
  true_action_table = format_simple_string(true_action)

  if search_info:
    # print('search_info', search_info)
    search_info_table = generate_kv_nested_html(search_info, '100%')
  else:
    search_info_table = None

  db_table = generate_kv_nested_html(db, '100%')
  html_code = """
  <!DOCTYPE html>
  <html>
  <body>
  AirDialogue Dataset Visualizer"""
  html_code += generate_form(partitions, sid, partition)
  html_code += """<h2>AirDialogue No. {0} on partition {1}. </h2>\n
  """.format(sid, partition)
  html_code = str(html_code)
  html_code += """<table style="width:100%">"""
  html_code += """<tr> {0}</tr>""".format(
      '<strong>intent:</strong>' + intent_table.encode('ascii', 'ignore') +
      '<br>')
  html_code += """<tr> {0}</tr>""".format(
      '<strong>user action:</strong>' + action_table.encode('ascii', 'ignore') +
      '<br>')
  html_code += """<tr> {0}</tr>""".format(
      '<strong>true action:</strong>' +
      true_action_table.encode('ascii', 'ignore') + '<br>')
  html_code += '<strong>' + res_str + '</strong><br>'
  html_code += """<tr> {0}</tr>""".format('<strong>db table:</strong>' +
                                          db_table.encode('ascii', 'ignore') +
                                          '<br>')

  if search_info_table:
    html_code += """<tr> {0}</tr>""".format(
        '<strong>search info:</strong>' +
        search_info_table.encode('ascii', 'ignore') + '<br>')

  if dialogue:
    html_code += """<tr> {0}</tr>""".format('<p>'.join(dialogue).encode(
        'ascii', 'ignore'))

  html_code += """</table>"""
  html_code += """
  </body>
  </html>"""
  return html_code


def make_path(path):
  if not gfile.Exists(path):
    gfile.MkDir(path)
