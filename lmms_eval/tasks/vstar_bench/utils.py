# Copyright 2025 Xiaomi Corporation.
import os, json
import re 
from typing import List
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter
# split test 



# Sample:
# {'text': 'What is the material of the glove?\n(A) rubber\n(B) cotton\n(C) kevlar\n(D) leather',
#  'category': 'direct_attributes',
#  'question_id': '0',
#  'label': 'A',
#  'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=2000x1500>}

def vstar_bench_doc_to_visual(doc):
    return [doc['image'].convert("RGB")]


def vstar_bench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc['text']
    return question + lmms_eval_specific_kwargs.get('post_prompt', '')


def vstar_bench_process_results(doc, results):
    prediction = results[0]
    answer = doc["label"] 
    score = 0 
    #
    pred_patterns = [
        r'Answer:?\s*[\(]?([A-Z])[\)]?',  # Pattern 1
        r'[\(]?([A-Z])[\)\.]?',           # Pattern 2
        r'option\s*([A-Z])',              # Pattern 3
        r'([A-Z])\s*option',              # Pattern 3
        r'answer\s*is\s*([A-Z])',         # Pattern 4
        r'([A-Z])\s*is\s*the\s*answer',   # Pattern 4
        r'^([A-Z])$'                      # Direct single letter
    ]
    
    # Try each pattern on the prediction
    pred_letter = None
    for pattern in pred_patterns:
        match = re.search(pattern, prediction, re.IGNORECASE)
        if match:
            pred_letter = match.group(1).upper()
            break
    
    # Extract answer letter using the same patterns
    ans_letter = None
    for pattern in pred_patterns:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            ans_letter = match.group(1).upper()
            break
    
    if pred_letter and ans_letter:
        score = 1 if pred_letter == ans_letter else 0
    return {"accuracy": score}  # Return 0 if no valid letter found in either prediction or answer


def vstar_bench_aggregate_results(results):
    correct = sum(results)
    total = len(results)
    return correct / total 


