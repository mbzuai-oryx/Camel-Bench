import pandas as pd
import json
import os
from tqdm import tqdm
from pydantic import BaseModel
from openai import OpenAI
import multiprocessing as mp
import base64
from utils import medicalmmt_doc_to_text
from typing import List
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

class VQAQuestion(BaseModel):
    question: str

class VQAAnswer(BaseModel):
    choice: str

def generate(text: str, images: List[str]) -> VQAAnswer:
    vqa_question = VQAQuestion(question=text)
    base64_images = [base64.b64encode(image).decode('utf-8') for image in images if image is not None]
    images_content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} for base64_image in base64_images]
    messages = [
        {
            "role": "system",
            "content": """You are an advanced AI assistant specialized in Visual Question Answering (VQA). Your task is to analyze the given image and answer the provided question accurately and comprehensively. Consider all visual elements, including objects, colors, text, and spatial relationships within the image. Provide clear, concise, and relevant answers based on the visual information.""",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Please answer the following question based on the provided image:\n\n{vqa_question.question}"},
                *images_content,],
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
                    "name": "answer_vqa_question",
                    "description": "Provide an answer to the Visual Question Answering task",
                    "parameters": VQAAnswer.model_json_schema(),
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "answer_vqa_question"}},
    )

    vqa_answer = VQAAnswer.model_validate_json(
        completion.choices[0].message.tool_calls[0].function.arguments
    )
    return vqa_answer

answer_field = "answer"
def process_row(row):
    i, row = row
    d = {}
    try:
        d['index'] = i
        # images = [row[f'image_{i}']['bytes'] for i in range(1, 8) if row[f'image_{i}'] is not None]
        images = [row['image'][0]['bytes']]
        d['pred_answer'] = generate(medicalmmt_doc_to_text(row), images).choice
        d['answer'] = row[answer_field]
        d['question'] = medicalmmt_doc_to_text(row)
        return d
    except Exception as e:
        print(f"Error processing row: {e}")
        return None

def main():
    path = "data_prunedv1.parquet"
    df = pd.read_parquet(path)

    num_processes = mp.cpu_count()
    pool = mp.Pool(processes=num_processes)
    results = list(tqdm(pool.imap(process_row, [(i, row) for i, row in df.iterrows()]), total=len(df)))
    pool.close()
    pool.join()

    report = [r for r in results if r is not None]
    with open("results_gpt.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()