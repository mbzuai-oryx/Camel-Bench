from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from typing import Callable
import pandas as pd
import json
import os
from tqdm import tqdm
from typing import List
import torch
from utils import *
import shutil

from datasets import load_dataset
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
#    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_id)

def handle_images1(row: pd.Series) -> List[str]:
    return [row['image'].convert('RGB')]

def handle_images2(row: pd.Series) -> List[str]:
    return [row.get(f"image_{i}", None).convert('RGB') for i in range(9) if row.get(f"image_{i}", None) is not None]

def handle_images3(row: pd.Series) -> List[str]:
    return [row['image'][0].convert('RGB')]

def save_images(images: List[str], with_resize: bool = True):
    for i, image in enumerate(images):
        if image is None: continue
        # with open(f"temp/image{i}.jpg", "wb") as f:
        #     f.write(image)

        if with_resize:
            # img = Image.open(f"temp/image{i}.jpg")
            img = image
            width, height = img.size
            req_dim = 420
            new_width = req_dim if width > height else int((req_dim / height) * width)
            new_height = int((req_dim / width) * height) if width > height else req_dim
            img = img.resize((new_width, new_height))
            img = img.convert("RGB")
            img.save(f"temp/image{i}.jpg")

def generate(prompt, images):
    images = images[:1]
    save_images(images)

    image = Image.open("temp/image0.jpg")

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=1000, use_cache=True)
    return processor.decode(output[0])

answer_field = "answer"
def process_row(row: pd.Series, fn: Callable, fn_images: Callable) -> dict:
    i, row = row
    d = {}
    try:
        d['index'] = i
        images = fn_images(row)
        d['pred_answer'] = generate(fn(row), images)
        d['answer'] = str(row[answer_field])
        d['question'] = fn(row)
        return d
    except Exception as e:
        print(f"Error processing row: {e}")
        return None

name_to_processor = {
    "mmmu": mmmu_doc_to_text,
    "mme": mme_doc_to_text,
    "gqa": gqa_doc_to_text,
    "realworldqa": realworldqa_doc_to_text,
    "vqav2": vqav2_doc_to_text,
    "vizwiz": vizwiz_doc_to_text,
    "pope": pope_doc_to_text,
    "countbench": countbench_doc_to_text,
    "medicalMMMU": medicalMMMU_doc_to_text,
    "medicalMMMUPro": medicalMMMUPro_doc_to_text,
    "diagramsMMMU": diagramsMMMU_doc_to_text,
    "mmbench": mmbench_doc_to_text,
    "seed": seed_doc_to_text,
    "medicalmmt": medicalmmt_doc_to_text,
    "hallucinationmmt": hallucinationmmt_doc_to_text,
    "vqammt": vqammt_doc_to_text,
    "mutliimagemmt": mutliimagemmt_doc_to_text,
    "isidocvqa": isidocvqa_doc_to_text,
    "patddocvqa": patddocvqa_doc_to_text,
    "celebvqa": celebvqa_doc_to_text,
    "countriesvqa": countriesvqa_doc_to_text,
    "foodvqa": foodvqa_doc_to_text,
    "objectcoco": objectcoco_doc_to_text,
    "blink": blink_doc_to_text,
    "examsv": examsv_doc_to_text,
    "chartqa": chartqa_doc_to_text,
    "mtvqa": mtvqa_doc_to_text,
    "mathvista": mathvista_doc_to_text,
    "infographicsvqa": infographicsvqa_doc_to_text,
    "agrovqa": agrovqa_doc_to_text,
    "diagramsvqa": diagramsvqa_doc_to_text,
    "tablesvqa": tablesvqa_doc_to_text,
    "iconqa": iconqa_doc_to_text,
    "scienceqa": scienceqa_doc_to_text,
    "ocrisi": ocrisi_doc_to_text,
    "evarest": evarest_doc_to_text,
    "historicalbooks": historicalbooks_doc_to_text,
    "khatt": khatt_doc_to_text,
    "patsocr": patsocr_doc_to_text,
    "arabicocr": arabicocr_doc_to_text,
    "culturevideovqa": culturevideovqa_doc_to_text,
    "videomme": videomme_doc_to_text,
    "geochat": geochat_doc_to_text,
}
name_to_handle_type = {
    "mmmu": handle_images2,
    "mme": handle_images1,
    "gqa": handle_images1,
    "realworldqa": handle_images1,
    "vqav2": handle_images1,
    "vizwiz": handle_images1,
    "pope": handle_images1,
    "countbench": handle_images1,
    "medicalMMMU": handle_images2,
    "medicalMMMUPro": handle_images2,
    "diagramsMMMU": handle_images2,
    "mmbench": handle_images1,
    "seed": handle_images3,
    "medicalmmt": handle_images3,
    "hallucinationmmt": handle_images3,
    "vqammt": handle_images3,
    "mutliimagemmt": handle_images3,
    "isidocvqa": handle_images1,
    "patddocvqa": handle_images1,
    "celebvqa": handle_images1,
    "countriesvqa": handle_images1,
    "foodvqa": handle_images1,
    "objectcoco": handle_images1,
    "blink": handle_images2,
    "examsv": handle_images1,
    "chartqa": handle_images1,
    "mtvqa": handle_images1,
    "mathvista": handle_images1,
    "infographicsvqa": handle_images1,
    "agrovqa": handle_images1,
    "diagramsvqa": handle_images1,
    "tablesvqa": handle_images1,
    "iconqa": handle_images2,
    "scienceqa": handle_images1,
    "ocrisi": handle_images1,
    "evarest": handle_images1,
    "historicalbooks": handle_images1,
    "khatt": handle_images1,
    "patsocr": handle_images1,
    "arabicocr": handle_images1,
    "culturevideovqa": handle_images2,
    "videomme": handle_images2,
    "geochat": handle_images1,
}
names = list(name_to_processor.keys())
os.makedirs("results", exist_ok=True)
os.makedirs("temp", exist_ok=True)

for name in tqdm(names):
    ds = load_dataset(f"ahmedheakl/arabic_{name}", split="train")
    df = pd.DataFrame(ds)
    print(f"Evaluating {name} dataset")
    fn = name_to_processor[name]
    fn_images = name_to_handle_type[name]
    results = []
    for i in tqdm(range(len(df))):
        results.append(process_row((i, df.iloc[i]), fn, fn_images))

    report = [r for r in results if r is not None]
    with open(f"results/llama11b_{name}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

shutil.rmtree("temp")