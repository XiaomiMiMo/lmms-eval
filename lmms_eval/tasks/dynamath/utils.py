# Copyright 2025 Xiaomi Corporation.

import json
import re

import numpy as np
from lmms_eval.tasks._task_utils.math_verify_utils import MathVerifyFn
from lmms_eval.tasks._task_utils.eval_utils import extract_final_boxed_content


def preprocess(str1):
    str0 = str1
    if str0 == "":
        return str1
    while str0[0] != "{":
        str0 = str0[1:]
        if len(str0) == 1:
            return str1
    while str0[-1] != "}":
        str0 = str0[:-1]
        if len(str0) == 1:
            return str1
    str2 = str0.replace("\\", "")
    str2 = str2.replace("\\n", "\n")
    return str2


def transfer(str1):
    if "\u03c0" in str1:
        strs = str1.split('\u03c0')
        str1 = strs[0]
        return float(str1) * np.pi
    else:
        return float(str1)



def dynamath_doc_to_visual(doc):
    image = doc["decoded_image"]
    return [image]


def dynamath_doc_to_text(doc):
    guide = """
## Answer Instruction Please provide an answer to the question outlined above. Your response should adhere 
to the following JSON format, which includes two keys: 'solution' and 'short answer'. The 'solution' key can contain 
detailed steps needed to solve the question, and the 'short answer' key should provide a concise response. If the problem is a multiple choice problem, just provide the corresponing choice option, 
such as 'A', 'B', 'C', or 'D'. If the answer is a numerical value, format it as a three-digit floating-point number.

Example of expected JSON response format:

"""
    example = {
        "solution": "[Detailed step-by-step explanation]",
        "short answer": "[Concise Answer]"
    }
    text_example = json.dumps(example, indent=4)
    question = doc["question"]
    text = f"## Question\n{question}"
    if doc["answer_type"] == "float":
        text = text + "Please answer in a floating-point number."
    text = text + guide + text_example
    return text


def dynamath_boxed_doc_to_text(doc):
    question = doc["question"]
    text = f"## Question\n{question}"
    if doc["answer_type"] == "float":
        text = text + "Please answer in a floating-point number."
    guide = "\nPut your final answer within \\boxed{}."
    text = text + guide
    return text


def dynamath_process_results(doc, results):

    try:
        description = preprocess(results[0])
    except:
        description = results[0]
    try:
        dj = json.loads(description, strict=False)
    except:
        dj = {
            "solution": description,
            "short answer": description.split('\n')[-1][:-1]
        }
    temp_data = {
        "question": doc.get('question'),
        "subject": doc.get('subject'),
        "knowledge level": doc.get('level'),
        "Ground truth": doc.get('ground_truth'),
        "doc_id": doc.get('id'),
        "response": dj
    }
    try:
        answer = str(dj.get("short answer"))
        if doc.get('answer_type') == "float":
            if not answer.isdigit():
                parts = answer.split(' ')
                answer = parts[0]
                answer = transfer(answer)
            diff = float(answer) - float(temp_data.get("Ground truth"))
            if abs(diff) <= 0.001:
                temp_data["result"] = "correct"
            else:
                temp_data["result"] = "fail"
        elif doc.get('answer_type') == "multiple choice":
            # 提取被() {} []包住的选项字母
            match = re.search(r'[\(\{\[]\s*([A-Za-z])\s*[\)\}\]]', answer)
            if match:
                answer = match.group(1).upper()
            if len(answer) == 1:
                if answer == temp_data.get("Ground truth"):
                    temp_data["result"] = "correct"
                else:
                    temp_data["result"] = "fail"
            else:
                if temp_data.get("Ground truth") in answer[0:3]:
                    temp_data["result"] = "correct"
                else:
                    temp_data["result"] = "fail"
        else:
            if temp_data.get("Ground truth") in answer:
                temp_data["result"] = "correct"
            else:
                temp_data["result"] = "fail"
    except:
        temp_data["result"] = "fail"
    return {
        "dynamath_standard_eval": temp_data,
        "dynamath_worst_case_acc": temp_data,
    }


math_verify_fn = MathVerifyFn()
def dynamath_boxed_process_results(doc, results):
    math_verify_score, math_verify_ext = math_verify_fn(results[0].strip(), doc["ground_truth"])
    answer = extract_final_boxed_content(results[0])

    temp_data = {
        "question": doc.get('question'),
        "subject": doc.get('subject'),
        "knowledge level": doc.get('level'),
        "Ground truth": doc.get('ground_truth'),
        "doc_id": doc.get('id'),
        "response": answer
    }
    try:
        if doc.get('answer_type') == "float":
            if not answer.isdigit():
                parts = answer.split(' ')
                answer = parts[0]
                answer = transfer(answer)
            diff = float(answer) - float(temp_data.get("Ground truth"))
            if abs(diff) <= 0.001:
                temp_data["result"] = "correct"
            else:
                temp_data["result"] = "fail"
        elif doc.get('answer_type') == "multiple choice":
            # 提取被() {} []包住的选项字母
            match = re.search(r'[\(\{\[]\s*([A-Za-z])\s*[\)\}\]]', answer)
            if match:
                answer = match.group(1).upper()
            if len(answer) == 1:
                if answer == temp_data.get("Ground truth"):
                    temp_data["result"] = "correct"
                else:
                    temp_data["result"] = "fail"
            else:
                if temp_data.get("Ground truth") in answer[0:3]:
                    temp_data["result"] = "correct"
                else:
                    temp_data["result"] = "fail"
        else:
            if temp_data.get("Ground truth") in answer:
                temp_data["result"] = "correct"
            else:
                temp_data["result"] = "fail"
    except:
        temp_data["result"] = "fail"
    return {
        "dynamath_standard_eval": temp_data, 
        "dynamath_worst_case_acc": temp_data,
        "math_verify": {
            "score": math_verify_score,
            "extraction": math_verify_ext
        }
    }


from collections import defaultdict
def dynamath_aggregate_results_worst_case_acc(results, args):
    doc_results = defaultdict(list)
    for result in results:
        doc_results[result["doc_id"]].append(result)

    correct_docs = 0
    total_docs = len(doc_results)
    for doc_id, results in doc_results.items():
        if all(result["result"] == "correct" for result in results):
            correct_docs += 1
    return correct_docs / total_docs if total_docs > 0 else 0


def dynamath_aggregate_results(results, args):
    total = len(results)
    correct = sum(result["result"] == "correct" for result in results)
    return correct / total


def dynamath_math_verify_aggregate_results(results, args):
    total = len(results)
    score = sum(result["score"] for result in results)
    return score / total
