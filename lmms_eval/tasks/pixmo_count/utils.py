# Copyright 2025 Xiaomi Corporation.
import os, json
import re 
from typing import List
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter


def pixmo_count_doc_to_visual(doc):
    return [ doc["image"].convert("RGB") ]


def pixmo_count_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc['question']
    return question.strip() + lmms_eval_specific_kwargs.get('post_prompt', '')

def pixmo_count_process_results(doc, results):
    prediction = results[0]
    answer = str(doc["answer"]) 
    score = 0 
    # exact match 
    if prediction.strip().lower() == answer.strip().lower():
        score = 1 
    return {"accuracy": score}  # Return 0 if no valid letter found in either prediction or answer


def pixmo_count_aggregate_results(results):
    correct = sum(results)
    total = len(results)
    return correct / total 


