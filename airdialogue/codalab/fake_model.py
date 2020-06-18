import json
import random
import sys

if __name__ == "__main__":
  output_txt = open(sys.argv[1], "w+")

  data = open(sys.argv[2], "r")
  if len(sys.argv) > 3:
    kb = open(sys.argv[3], "r")

  for s in data:
    # Response in format of "utterance | name | flight | action"
    response = "Hi I am an agent. How can I help today? |  |  | "
    obj = json.loads(s)
    d = {"dialogue": obj["dialogue"], "action": obj["dialogue"]}
    if "intent" in obj.keys():
      d["intent"] = obj["intent"]
      response = "Hi I am a client. I want to book a flight."
    else:
      # Agent
      if random.randint(1, 20) < 10 and len(d["dialogue"]) > 5:
        response = ("I have reserved for James Smith. | James Smith | 12345 | "
                    "book")

    output_txt.write(response + "\n")
  output_txt.close()
