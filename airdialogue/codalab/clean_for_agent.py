# Usage: python clean_for_agent.py data.json
# Creates agent.data.json

import json
import sys

if __name__ == "__main__":
  agent_json = open("agent.data.json", "w+")

  data = open(sys.argv[1], "r")
  for s in data:
    obj = json.loads(s)

    d = {"dialogue": obj["dialogue"], "action": obj["action"]}
    agent_json.write(json.dumps(d) + "\n")
  agent_json.close()
