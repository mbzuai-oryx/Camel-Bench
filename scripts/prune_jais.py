import ollama
import pandas as pd
import json
from tqdm import tqdm
import arabic_reshaper
from bidi.algorithm import get_display
from utils import medicalmmt_doc_to_text

def display_ar(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

def generate(text):
    
    response = ollama.chat(model="jwnder/jais-adaptive:7b", messages=[
        {
            'role': 'user',
            'content': text
        }
    ], options=ollama.Options(temperature=1))
    return response['message']['content']

path = "medical-mmt-mi_checked.parquet"
df = pd.read_parquet(path)
answer_field = "answer"

report = []
for i in tqdm(range(len(df))):
    row = df.iloc[i]
    try:
        d = {}
        d['index'] = i
        d['pred_answer'] = generate(medicalmmt_doc_to_text(row))
        d['answer'] = row[answer_field]
        d['question'] = medicalmmt_doc_to_text(row)
        report.append(d)
    except Exception as e:
        print(e)
        continue

with open("results_jais.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)