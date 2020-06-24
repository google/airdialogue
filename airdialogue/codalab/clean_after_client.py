# Usage: python clean_after_client.py data.json out.txt new_data.json

import json
import sys

if __name__ == "__main__":
  new_data_json = open(sys.argv[3], "w+")

  data = open(sys.argv[1], "r")
  out = open(sys.argv[2], "r")
  for s in data:
    clientOut = out.readline()

    obj = json.loads(s)
    d = {
        "dialogue": obj["dialogue"],
        "action": obj["action"],
        "intent": obj["intent"]
    }
    if not d["action"]:
      # Convo ongoing, add to master
      d["dialogue"].append("customer: " + clientOut.strip())

    new_data_json.write(json.dumps(d) + "\n")
  new_data_json.close()
