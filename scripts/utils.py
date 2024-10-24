import pandas as pd 
import ast
import numpy as np

arabic_letters = {
    'A': 'أ',
    'B': 'ب',
    'C': 'ج',
    'D': 'د',
    'E': 'ه',
    'F': 'و',
    'G': 'ز',
    'H': 'ح',
    'I': 'ط',
    'J': 'ي',
    'K': 'ك',
    'L': 'ل',
    'M': 'م',
    'N': 'ن',
    'O': 'ع',
    'P': 'ف',
    'Q': 'ص',
    'R': 'ر',
    'S': 'س',
    'هـ': 'ه',
    'ا': 'أ'
}

def translate_numbers(text: str) -> str:
    english_to_arabic = {
        '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤',
        '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩'
    }
    
    translation_table = str.maketrans(english_to_arabic)
    return text.translate(translation_table)


def mcq_eval(pred: str, gt: str):
    pred = pred.strip()
    gt = gt.strip()
    if len(pred) > 2 and pred[0] == '(' and pred[2] == ')':
        pred = pred[1]
    if len(gt) > 2 and gt[0] == '(' and gt[2] == ')':
        gt = gt[1]
    pred = pred[0]
    gt = gt[0]
    pred = arabic_letters.get(pred, pred)
    gt = arabic_letters.get(gt, gt)
    return pred == gt

def create_options_prompt(row_data, option_candidate):
    available_keys = set(row_data.keys()) & set(option_candidate)
    options = {cand: row_data[cand] for cand in available_keys if row_data[cand]}
    sorted_options = dict(sorted(options.items()))
    options_prompt = f"هناك عدة خيارات:\n"
    for key, item in sorted_options.items():
        if pd.notna(item) and item != "nan":
            arabic_key = arabic_letters[key]
            options_prompt += f"{arabic_key}. {item}\n"
    return options_prompt.rstrip("\n")

def mmbench_doc_to_text(doc):
    option_candidate = ["A", "B", "C", "D", "E"]
    options = create_options_prompt(doc, option_candidate)
    question = f"{doc['hint']} {doc['question']} {options}" if pd.notna(doc["hint"]) and doc["hint"] != "nan" else f"{doc['question']} {options}"
    return f"{question}\nأجب بحرف الخيار من الاختيارات المعطاة مباشرة."

def mmbench_eval(pred, gt):
    return mcq_eval(pred, gt)

def mme_doc_to_text(doc):
    question = doc["question"].strip()
    return question

def mme_eval(pred: str, gt: str):
    pred = pred.strip()
    if pred == "صح":
        pred = 'نعم'
    return pred == gt
    
def default_eval(pred: str, gt: str):
    pred = pred.strip()
    gt = gt.strip()
    return pred == gt


def iconqa_doc_to_text(doc):
    return doc['question']

def iconqa_eval(pred, gt):
    return mcq_eval(pred, gt)

def mmmu_parse_options(options):
    option_letters = [arabic_letters[chr(ord("A") + i)] for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str


MMMU_MULTI_CHOICE_PROMPT = "أجب بحرف الخيار من الاختيارات المعطاة مباشرة."
MMMU_OPEN_ENDED_PROMPT = "أجب عن السؤال باستخدام كلمة أو عبارة واحدة."

def mmmu_doc_to_text(doc):
    question = doc["question"]
    if doc["question_type"] == "multiple-choice":
        # Weirdly, data["options"] is a string in MMMU Huggingface dataset
        parsed_options = mmmu_parse_options(ast.literal_eval(doc["options"].replace("،", ",")))
        question = f"{question}\n{parsed_options}\n\n{MMMU_MULTI_CHOICE_PROMPT}"
    else:
        question = f"{question}\n\n{MMMU_OPEN_ENDED_PROMPT}"
    return question

def mmmu_eval(pred, gt):
    return mcq_eval(pred, gt)

def gqa_doc_to_text(doc):
    question = doc["question"]
    post_prompt = "\nأجب عن السؤال باستخدام كلمة أو عبارة واحدة."
    return f"{question}{post_prompt}"

def gqa_eval(pred, gt):
    return default_eval(pred, gt)

def realworldqa_doc_to_text(doc):
    question = doc["question"].strip()
    pre_prompt = "المستخدم\nالسؤال: "
    return f"{pre_prompt}{question}"

def realworldqa_eval(pred, gt):
    return mcq_eval(pred, gt)

def vqav2_doc_to_text(doc):
    post_prompt = "\nأجب على السؤال باستخدام كلمة أو عبارة واحدة."
    return f"{doc['question']}{post_prompt}"

def vizwiz_doc_to_text(doc):
    post_prompt = "\nعندما تكون المعلومات المقدمة غير كافية، أجب بـ 'لا يمكن الإجابة'.\nأجب عن السؤال باستخدام كلمة واحدة أو عبارة قصيرة."
    text = f"{doc['question'].capitalize()}{post_prompt}"
    return text

def vizwiz_eval(pred: str, gt: str):
    try:
        _ = ast.literal_eval(gt)
        gt = gt.replace(" ", ", ")
        gt = ast.literal_eval(gt)
        print(gt)
    except:
        gt = gt.strip()
    pred = pred.strip()
    if pred == gt:
        return True
    for x in gt:
        if x in pred:
            return True
    return False

def pope_doc_to_text(doc):
    question = doc["question"].strip()
    return f"{question}\nأجب عن السؤال باستخدام كلمة واحدة أو عبارة قصيرة."

def pope_eval(pred, gt):
    pred = pred.strip()
    gt = gt.strip()
    return gt in pred

def countbench_doc_to_text(_):
    return "كم عدد الأشياء الموجودة في الصورة؟\nأجب برقم فقط."

def countbench_eval(pred, gt):
    pred = pred.strip()
    gt = gt.strip()
    return translate_numbers(pred) == translate_numbers(gt)

def diagramsMMMU_eval(pred, gt):
    pred = pred.strip()
    gt = gt.strip()
    if len(gt) == 1:
        return pred[0] == gt
    pred = translate_numbers(pred)
    gt = translate_numbers(gt)
    return gt in pred

def diagramsMMMU_doc_to_text(doc):
    return mmmu_doc_to_text(doc)

def medicalMMMU_eval(pred, gt):
    return mcq_eval(pred, gt)

def medicalMMMU_doc_to_text(doc):
    return mmmu_doc_to_text(doc)

def medicalMMMUPro_eval(pred, gt):
    return mcq_eval(pred, gt)

def medicalMMMUPro_parse_options(options):
    option_letters = [arabic_letters[chr(ord("A") + i)] for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str


def medicalMMMUPro_doc_to_text(doc):
    post_prompt="أجب بحرف الخيار من الخيارات المعطاة مباشرة."
    question = doc["question"]
    # Weirdly, data["options"] is a string in MMMU Huggingface dataset
    parsed_options = medicalMMMUPro_parse_options(ast.literal_eval(doc["options"].replace("،", ",")))
    question = f"{question}\n{parsed_options}\n\n{post_prompt}"
    return question


def mmt_doc_to_text(doc):
    question_text = "سؤال: <image>\n" + doc["question"].strip()

    options = []
    for option in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]:
        option_text = doc.get(option)
        if option_text and option_text.strip():
            options.append(f"{arabic_letters[option]}: {option_text.strip()}")

    options_text = "\n".join(options) if options else ""

    formatted_question = f"{question_text}\n{options_text}"
    post_prompt = "\nأجب عن السؤال باستخدام حرف واحد من الخيارات المعطاة."
    formatted_question = f"{formatted_question}{post_prompt}"

    return formatted_question

def medicalmmt_doc_to_text(doc):
    return mmt_doc_to_text(doc)

def medicalmmt_eval(pred, gt):
    return mcq_eval(pred, gt)

def seed_doc_to_text(doc):
    question = doc["question"]
    question += "\n" + f"أ. {doc['choice_a']}\n"
    question += f"ب. {doc['choice_b']}\n"
    question += f"ج. {doc['choice_c']}\n"
    question += f"د. {doc['choice_d']}"
    return f"{question}\nأجب بحرف الخيار من الاختيارات المعطاة مباشرة."

def seed_eval(pred, gt):
    return mcq_eval(pred, gt)

def hallucinationmmt_doc_to_text(doc):
    return mmt_doc_to_text(doc)

def hallucinationmmt_eval(pred, gt):
    return mcq_eval(pred, gt)

def vqammt_doc_to_text(doc):
    return mmt_doc_to_text(doc)

def vqammt_eval(pred, gt):
    return mcq_eval(pred, gt)



def mutliimagemmt_doc_to_text(doc):
    return mmt_doc_to_text(doc)

def mutliimagemmt_eval(pred, gt):
    return mcq_eval(pred, gt)


def our_options_to_str(options):
    option_prompt_str = ""
    for i, option in enumerate(options):
        option_choice = chr(ord("A") + i)
        option_choice = arabic_letters[option_choice]
        option_prompt_str += f"{option_choice}. {option}\n"

    option_prompt_str = option_prompt_str.rstrip("\n")
    return option_prompt_str

def our_doc_to_text(doc):
    question_text = "سؤال:\n" + doc["question"].strip()
    options = our_options_to_str(doc["options"])
    options_text = "\n".join(options) if options else ""
    formatted_question = f"{question_text}\n{options_text}"
    post_prompt = "\nأجب عن السؤال باستخدام حرف واحد من الخيارات المعطاة."
    formatted_question = f"{formatted_question}{post_prompt}"
    return formatted_question


def isidocvqa_doc_to_text(doc):
    return our_doc_to_text(doc)

def isidocvqa_eval(pred, gt):
    return mcq_eval(pred, gt) 

def patddocvqa_doc_to_text(doc):
    return our_doc_to_text(doc)

def patddocvqa_eval(pred, gt):
    return mcq_eval(pred, gt)

def celebvqa_doc_to_text(doc):
    return our_doc_to_text(doc)

def celebvqa_eval(pred, gt):
    return mcq_eval(pred, gt)

def countriesvqa_doc_to_text(doc):
    return our_doc_to_text(doc)

def countriesvqa_eval(pred, gt):
    return mcq_eval(pred, gt)

def foodvqa_doc_to_text(doc):
    return our_doc_to_text(doc)

def foodvqa_eval(pred, gt):
    return mcq_eval(pred, gt)

def objectcoco_doc_to_text(doc):
    return doc['question']

def objectcoco_eval(pred, gt):
    return mcq_eval(pred, gt)

def blink_doc_to_text(doc):
    return doc['question']

def blink_eval(pred, gt):
    return mcq_eval(pred, gt)

def examsv_doc_to_text(doc):
    return doc['question']

def examsv_eval(pred, gt):
    return mcq_eval(pred, gt)

def chartqa_doc_to_text(doc):
    return doc['question']

def chartqa_eval(pred, gt):
    return mcq_eval(pred, gt)

def mtvqa_doc_to_text(doc):
    return doc['question']

def create_one_query(problem):
    demo_prompt = ""

    question = problem["question"]
    unit = problem["unit"]
    choices = problem["choices"]
    precision = problem["precision"]
    question_type = problem["question_type"]
    answer_type = problem["answer_type"]

    if question_type == "multi_choice":
        assert answer_type == "text"
        hint_text = f"تلميح: يرجى الإجابة على السؤال وتقديم حرف الخيار الصحيح، مثل أ أو ب أو ج أو د، في النهاية."
    else:
        assert answer_type in ["integer", "float", "list"]
        if answer_type == "integer":
            hint_text = f"تلميح: يرجى الإجابة على السؤال الذي يتطلب إجابة بعدد صحيح وتقديم القيمة النهائية، مثل 1 أو 2 أو 3، في النهاية."

        elif answer_type == "float" and precision == 1:
            hint_text = f"تلميح: يرجى الإجابة على السؤال الذي يتطلب رقمًا عشريًا بمنزلة عشرية واحدة وتقديم القيمة النهائية، مثل 1.2 أو 1.3 أو 1.4، في النهاية."

        elif answer_type == "float" and precision == 2:
            hint_text = f"تلميح: يرجى الإجابة على السؤال الذي يتطلب رقمًا عشريًا بمنزلتين عشريتين وتقديم القيمة النهائية، مثل 1.23 أو 1.34 أو 1.45، في النهاية."

        elif answer_type == "list":
            hint_text = f"تلميح: يرجى الإجابة على السؤال الذي يتطلب قائمة بايثون كإجابة وتقديم القائمة النهائية، مثل [1, 2, 3] أو [1.2, 1.3, 1.4]، في النهاية."

    hint_text = translate_numbers(hint_text)    

    question_text = f"سؤال: {question}"
    if unit:
        question_text += f" (الوحدة: {unit})"

    if choices and choices != 'None':
        texts = ["الاختيارات:"]
        choices = ast.literal_eval(choices.replace("' '", '", "'))
        for i, choice in enumerate(choices):
            texts.append(f"({arabic_letters[chr(ord('A')+i)]}) {choice}")
        choices_text = "\n".join(texts)
    else:
        choices_text = ""


    prompt = "الحل: "

    elements = [question_text, choices_text, hint_text, prompt]
    test_query = "\n".join([e for e in elements if e != ""])

    query = demo_prompt + "\n\n" + test_query
    query = query.strip()
    return query

def mathvista_doc_to_text(doc):
    problem = {
        "question_type": doc["question_type"],
        "answer_type": doc["answer_type"],
        "question": doc["question"],
        "unit": doc["unit"] if "unit" in doc else "",
        "choices": doc["choices"],
        "answer": doc["answer"] if "answer" in doc else None,
        "precision": doc["precision"] if "precision" in doc else 0,
    }
    query_prompt = create_one_query(problem)
    return query_prompt


def infographicsvqa_doc_to_text(doc):
    return doc['question']

def infographicsvqa_eval(pred, gt):
    return mcq_eval(pred, gt)

def agrovqa_doc_to_text(doc):
    return doc['question']

def agrovqa_eval(pred, gt):
    return mcq_eval(pred, gt)

def diagramsvqa_doc_to_text(doc):
    return doc['question']

def diagramsvqa_eval(pred, gt):
    return mcq_eval(pred, gt)

def tablesvqa_doc_to_text(doc):
    return doc['question']

def tablesvqa_eval(pred, gt):
    return mcq_eval(pred, gt)

def scienceqa_doc_to_text(doc):
    context, question, choices = doc["hint"], doc["question"], doc["choices"]
    choices = ast.literal_eval(choices)
    len_choices = len(choices)
    options = [arabic_letters[chr(ord("A") + i)] for i in range(len_choices)]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
    if context:
        context = f"السياق: {context}\n"

    post_prompt = "\n.أجب بحرف الخيار من الخيارات المعطاة مباشرة"
    return f"{context}{question}\n{choices_str}{post_prompt}"

def scienceqa_eval(pred, gt):
    gt = arabic_letters[chr(ord('A') + gt)]
    return mcq_eval(pred, gt)

def ocrisi_doc_to_text(doc):
    return doc['question']

def cer(pred, gt):
    d = np.zeros((len(gt) + 1, len(pred) + 1))
    for i in range(len(gt) + 1):
        d[i, 0] = i
    for j in range(len(pred) + 1):
        d[0, j] = j
    
    for i in range(1, len(gt) + 1):
        for j in range(1, len(pred) + 1):
            if gt[i-1] == pred[j-1]:
                d[i, j] = d[i-1, j-1]
            else:
                substitution = d[i-1, j-1] + 1
                insertion = d[i, j-1] + 1
                deletion = d[i-1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)
    
    error = d[len(gt), len(pred)]
    total_chars = len(gt)
    cer = error / total_chars
    
    return cer


def ocrisi_eval(pred: str, gt: str):
    pred = pred.strip()
    gt = gt.strip()
    return cer(pred, gt) <= 0.1

def evarest_doc_to_text(doc):
    return doc['question']

def evarest_eval(pred: str, gt: str):
    pred = pred.strip()
    gt = gt.strip()
    return cer(pred, gt) <= 0.1

def historicalbooks_doc_to_text(doc):
    return doc['question']

def historicalbooks_eval(pred: str, gt: str):
    pred = pred.strip()
    gt = gt.strip()
    return cer(pred, gt) <= 0.4

def khatt_doc_to_text(doc):
    return doc['question']

def khatt_eval(pred: str, gt: str):
    pred = pred.strip()
    gt = gt.strip()
    return cer(pred, gt) <= 0.3

def patsocr_doc_to_text(doc):
    return doc['question']

def patsocr_eval(pred: str, gt: str):
    pred = pred.strip()
    gt = gt.strip()
    return cer(pred, gt) <= 0.1

def arabicocr_doc_to_text(doc):
    return doc['question']

def arabicocr_eval(pred: str, gt: str):
    pred = pred.strip()
    gt = gt.strip()
    return cer(pred, gt) <= 0.4


def culturevideovqa_doc_to_text(doc):
    return doc['question']

def culturevideovqa_eval(pred, gt):
    return mcq_eval(pred, gt)


def videomme_doc_to_text(doc):
    prompt = """الترجمات النصية لهذا الفيديو مدرجة أدناه:
{subtitles}
اختر أفضل إجابة للسؤال التالي متعدد الخيارات بناءً على الفيديو. أجب فقط بالحرف (أ، ب، ج، أو د) للخيار الصحيح.
{question}
{options}
أفضل إجابة هي:"""
    subtitles = doc["subtitles"]
    question = doc["question"]
    options = doc["options"]
    options_str = "\n".join([f"{option}" for i, option in enumerate(options)])
    return prompt.format(subtitles=subtitles, question=question, options=options_str)

def videomme_eval(pred, gt):
    return mcq_eval(pred, gt)
    
def geochat_doc_to_text(doc):
    pre_prompt = "أجب على السؤال التالي بكلمة أو جملة.\n"
    return pre_prompt + doc['question']