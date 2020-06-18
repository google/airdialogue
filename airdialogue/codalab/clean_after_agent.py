# Usage: python clean_after_agent.py data.json out.txt new_data.json

import json
import sys

if __name__ == "__main__":
  new_data_json = open(sys.argv[3], "w+")

  data = open(sys.argv[1], "r")
  out = open(sys.argv[2], "r")
  for s in data:
    action = None
    agentOut = out.readline().split("|")
    assert len(agentOut) == 4
    if agentOut[1].strip() or agentOut[2].strip() or agentOut[3].strip():
      action = {
          "name":
              agentOut[1].strip(),
          "flight":
              list(map(int, agentOut[2].strip().split(",")))
              if agentOut[2] else agentOut[2],
          "status":
              agentOut[3].strip(),
      }

    obj = json.loads(s)
    d = {
        "dialogue": obj["dialogue"],
        "action": obj["action"],
        "intent": obj["intent"]
    }
    if not d["action"]:
      # Conversation ongoing
      d["dialogue"].append("agent: " + agentOut[0].strip())
      # Close this convo if we predict an action
      if action:
        d["action"] = action

    new_data_json.write(json.dumps(d) + "\n")
  new_data_json.close()
