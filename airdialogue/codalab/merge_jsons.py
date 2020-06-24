import json
import sys

if __name__ == "__main__":
  new_data_json = open(sys.argv[3], "w+")

  a = json.load(open(sys.argv[1], "r"))
  b = json.load(open(sys.argv[2], "r"))

  new_data = {
      "bleu": (a["bleu"] + b["bleu"]) / 2,
      "score": (a["score"] + b["score"]) / 2,
  }

  json.dump(new_data, new_data_json)
  new_data_json.close()
