# Copyright 2025 Xiaomi Corporation.

from typing import List
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter


def vlmsareblind_doc_to_visual(doc):
    return [ doc["image"].convert("RGB") ]


def vlmsareblind_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc['prompt']
    return question.strip() + lmms_eval_specific_kwargs.get('post_prompt', '')

def vlmsareblind_process_results(doc, results):
    resp = results[0]
    for kw in ["boxed", "think", "bbox", "\n"]:
        resp = resp.split(kw)[-1]
    for kw in ["rows", "columns"]:
        resp = resp.replace(kw, "")
    resp = "".join([c for c in resp if (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or (c >= '0' and c <= '9')])

    answer = str(doc["groundtruth"]) 
    score = 0 
    # exact match 
    if resp.strip().lower() == answer.strip().lower().replace(",", ""):
        score = 1 
    return {"accuracy": score}  # Return 0 if no valid letter found in either prediction or answer


def vlmsareblind_aggregate_results(results):
    correct = sum(results)
    total = len(results)
    return correct / total 


