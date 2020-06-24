# Usage: python initilize.py data.json
# Creates start.data.json

import json
import sys

if __name__ == "__main__":
  start_json = open("start.data.json", "w+")

  data = open(sys.argv[1], "r")
  for s in data:
    obj = json.loads(s)

    seed = {
        "dialogue": [],
        "action": {
            #TODO: add name, flight, status?
        },
        "intent": obj["intent"]
    }
    start_json.write(json.dumps(seed) + "\n")
