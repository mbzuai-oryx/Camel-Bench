import json
from utils import medicalmmt_eval

p1 = "results_qwen7b.json"
p2 = "results_silma9b.json"
p3 = "results_jais.json"

with open(p1, "r") as f:
    data1 = json.load(f)

with open(p2, "r") as f:
    data2 = json.load(f)

with open(p3, "r") as f:
    data3 = json.load(f)



i1 = [f['index'] for f in data1 if medicalmmt_eval(f['pred_answer'], f['answer'])]
i2 = [f['index'] for f in data2 if medicalmmt_eval(f['pred_answer'], f['answer'])]
i3 = [f['index'] for f in data3 if medicalmmt_eval(f['pred_answer'], f['answer'])]

c12 = set(i1).intersection(i2)
c13 = set(i1).intersection(i3)
c23 = set(i2).intersection(i3)
max_c = max(len(c12), len(c13), len(c23))
# c = c12 if len(c12) == max_c else c13 if len(c13) == max_c else c23
c = set(i1).intersection(i2).intersection(i3)   


print(f"Qwen and Silma: {len(c12)}")
print(f"Qwen and Jais: {len(c13)}")
print(f"Silma and Jais: {len(c23)}")


report = []
for i in c:
    d = {}
    d['index'] = i
    d['pred_answer'] = data1[i]['pred_answer']
    d['answer'] = data1[i]['answer']
    d['question'] = data1[i]['question']
    report.append(d)

with open("results_merged.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)