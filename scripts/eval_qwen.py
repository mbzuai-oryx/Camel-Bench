from typing import Callable
import pandas as pd
import json
import os
from tqdm import tqdm
from typing import List
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import utils
from PIL import Image
import shutil

from datasets import load_dataset


model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

def handle_images1(row: pd.Series) -> List[str]:
    return [row['image']['bytes']]

def handle_images2(row: pd.Series) -> List[str]:
    return [row.get(f"image_{i}", None)['bytes'] for i in range(9) if row.get(f"image_{i}", None) is not None]

def handle_images3(row: pd.Series) -> List[str]:
    return [row['image'][0]['bytes']]

def save_images(images: List[str], with_resize: bool = True):
    for i, image in enumerate(images):
        if image is None: continue
        with open(f"temp/image{i}.jpg", "wb") as f:
            f.write(image)

        if with_resize:
            img = Image.open(f"temp/image{i}.jpg")
            width, height = img.size
            new_width = 512 if width > height else int((512 / height) * width)
            new_height = int((512 / width) * height) if width > height else 512
            img = img.resize((new_width, new_height))
            img = img.convert("RGB")
            img.save(f"temp/image{i}.jpg")

def generate_qwen(prompt: str, images: List[str])-> str:
    images = images[:1]
    save_images(images)
    images_content = [{"type": "image", "image": f"file://{os.getcwd()}/temp/image{i}.jpg"} for i in range(len(images))]
    messages = [
        {
            "role": "user",
            "content": [
                *images_content,
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=2000)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return output_text[0]
        
    
answer_field = "answer"
def process_row(row: pd.Series, fn: Callable, fn_images: Callable) -> dict:
    i, row = row
    d = {}
    try:
        d['index'] = i
        images = fn_images(row)
        d['pred_answer'] = generate_qwen(fn(row), images)
        d['answer'] = str(row[answer_field])
        d['question'] = fn(row)
        return d
    except Exception as e:
        print(f"Error processing row: {e}")
        return None

name_to_processor = {
    "mmmu": utils.mmmu_doc_to_text,
    "mme": utils.mme_doc_to_text,
    "gqa": utils.gqa_doc_to_text,
    "realworldqa": utils.realworldqa_doc_to_text,
    "vqav2": utils.vqav2_doc_to_text,
    "vizwiz": utils.vizwiz_doc_to_text,
    "pope": utils.pope_doc_to_text,
    "countbench": utils.countbench_doc_to_text,
    "medicalMMMU": utils.medicalMMMU_doc_to_text,
    "medicalMMMUPro": utils.medicalMMMUPro_doc_to_text,
    "diagramsMMMU": utils.diagramsMMMU_doc_to_text,
    "mmbench": utils.mmbench_doc_to_text,
    "seed": utils.seed_doc_to_text,
    "medicalmmt": utils.medicalmmt_doc_to_text,
    "hallucinationmmt": utils.hallucinationmmt_doc_to_text,
    "vqammt": utils.vqammt_doc_to_text,
    "mutliimagemmt": utils.mutliimagemmt_doc_to_text,
    "isidocvqa": utils.isidocvqa_doc_to_text,
    "patddocvqa": utils.patddocvqa_doc_to_text,
    "celebvqa": utils.celebvqa_doc_to_text,
    "countriesvqa": utils.countriesvqa_doc_to_text,
    "foodvqa": utils.foodvqa_doc_to_text,
    "objectcoco": utils.objectcoco_doc_to_text,
    "blink": utils.blink_doc_to_text,
    "examsv": utils.examsv_doc_to_text,
    "chartqa": utils.chartqa_doc_to_text,
    "mtvqa": utils.mtvqa_doc_to_text,
    "mathvista": utils.mathvista_doc_to_text,
    "infographicsvqa": utils.infographicsvqa_doc_to_text,
    "agrovqa": utils.agrovqa_doc_to_text,
    "diagramsvqa": utils.diagramsvqa_doc_to_text,
    "tablesvqa": utils.tablesvqa_doc_to_text,
    "iconqa": utils.iconqa_doc_to_text,
    "scienceqa": utils.scienceqa_doc_to_text,
    "ocrisi": utils.ocrisi_doc_to_text,
    "evarest": utils.evarest_doc_to_text,
    "historicalbooks": utils.historicalbooks_doc_to_text,
    "khatt": utils.khatt_doc_to_text,
    "patsocr": utils.patsocr_doc_to_text,
    "arabicocr": utils.arabicocr_doc_to_text,
    "culturevideovqa": utils.culturevideovqa_doc_to_text,
    "videomme": utils.videomme_doc_to_text,
    "geochat": utils.geochat_doc_to_text,
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
    with open(f"results/qwenvl2b_{name}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

shutil.rmtree("temp")

