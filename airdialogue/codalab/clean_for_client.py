# Usage: python clean_for_client.py data.json
# Creates client.data.json

import json
import sys

if __name__ == "__main__":
  client_json = open("client.data.json", "w+")

  data = open(sys.argv[1], "r")
  for s in data:
    obj = json.loads(s)

    d = {
        "dialogue": obj["dialogue"],
        "action": obj["action"],
        "intent": obj["intent"]
    }
    client_json.write(json.dumps(d) + "\n")
  client_json.close()
