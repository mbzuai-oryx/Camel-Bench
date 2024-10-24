import json
from tqdm import tqdm
from pydantic import BaseModel
from openai import OpenAI
from typing import List
import multiprocessing as mp
import os

from dotenv import load_dotenv

load_dotenv()



class AnswerScore(BaseModel):
    score: int

def eval_gpt(row):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    question = row['question'].split("\n")[0]
    pred = row['pred_answer']
    gt = row['answer']
    
    messages = [
        {
            "role": "system",
            "content": """You are an expert in natural language understanding and semantic similarity. Your task is to evaluate the semantic similarity between two given sentences: a predicted answer and a ground truth answer. You should output a score of 1 if the sentences are semantically similar, and 0 if they are not.""",
        },
        {
            "role": "user",
            "content": f"""Here are three examples to guide your evaluation:
Example 1:
Question: "ما هي اللغة المستخدمة في النص؟"
Predicted Answer: "العربية"
Ground Truth: "اللغة العربية"
Score: 1

Example 2:
Question: "ما هو موضوع النص؟"
Predicted Answer: "إثنان"
Ground Truth: "الحب و الكراهية"
Score: 0

Example 3:
Question: "ما هو عدد صفحات الكتاب؟"
Predicted Answer: "الصورة لا تحتوي على عدد صفحات الكتاب."
Ground Truth: "غير معروف"
Score: 1

Now, for each new pair of sentences, analyze their semantic similarity and provide a score of 1 for similar meanings or 0 for different meanings. Always consider the context and potential variations in expressing the same concept.
Question: "{question}"
Predicted Answer: "{pred}"
Ground Truth: "{gt}"
Score: """
        },
    ]

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=300,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "answer_score",
                    "description": "Provide a [0, 1] score to the semantic similarity between two sentences",
                    "parameters": AnswerScore.model_json_schema(),
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "answer_score"}},
    )

    vqa_answer = AnswerScore.model_validate_json(
        completion.choices[0].message.tool_calls[0].function.arguments
    )
    return {
        'index': row['index'],
        'question': question,
        'pred_answer': pred,
        'answer': gt,
        'evaluation': vqa_answer.score
    }

def process_chunk(chunk):
    return [eval_gpt(row) for row in chunk]

def main():
    path = "results_gpt.json"
    with open(path, "r") as f:
        data = json.load(f)

    num_cores = mp.cpu_count()
    chunk_size = len(data) // num_cores
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    pool = mp.Pool(num_cores)
    results = []
    with tqdm(total=len(data)) as pbar:
        for chunk_result in pool.imap_unordered(process_chunk, chunks):
            results.extend(chunk_result)
            pbar.update(len(chunk_result))

    pool.close()
    pool.join()

    correct_count = sum(1 for item in results if item['evaluation'] == 1)
    total_count = len(results)
    accuracy = correct_count / total_count

    results.sort(key=lambda x: x['index'])
    with open("results_gpt.json", "w", encoding="utf-8") as f:
        json.dump({
            'results': results,
            'accuracy': accuracy
        }, f, ensure_ascii=False, indent=2)

    print(f"Evaluation complete. Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()