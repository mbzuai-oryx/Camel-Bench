import json
from utils import medicalmmt_eval

with open("results_gpt.json", "r") as f:
    data = json.load(f)

tot = 0
for r in data:
    tot += medicalmmt_eval(r['pred_answer'], r['answer'])
print(f"{tot} / {len(data)} -> {tot * 100 / len(data):.2f}")