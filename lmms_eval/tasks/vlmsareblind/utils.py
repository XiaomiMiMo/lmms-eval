# Copyright 2025 Xiaomi Corporation.
from typing import List
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter


def vlmsareblind_doc_to_visual(doc):
    return [ doc["image"].convert("RGB") ]


def vlmsareblind_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc['prompt']
    return question.strip() + lmms_eval_specific_kwargs.get('post_prompt', '')

def vlmsareblind_process_results(doc, results):
    prediction = results[0]
    answer = str(doc["groundtruth"]) 
    score = 0 
    # exact match 
    if prediction.strip().lower() == answer.strip().lower():
        score = 1 
    return {"accuracy": score}  # Return 0 if no valid letter found in either prediction or answer


def vlmsareblind_aggregate_results(results):
    correct = sum(results)
    total = len(results)
    return correct / total 


