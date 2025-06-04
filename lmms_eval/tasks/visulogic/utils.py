# Copyright 2025 Xiaomi Corporation.

import os, json
import re 
from typing import List
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter
# split test 



# Sample:
# {'label': 'A',
#  'question': 'From the four given options, select the most suitable one to fill in the question mark, so that a certain regularity is presented:\n\n\n\nA: A  \nB: B  \nC: C  \nD: D',
#  'tag': 'Quantitative Reasoning',
#  'id': '00000',
#  'image': <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=600x255>}

def visulogic_doc_to_visual(doc):
    return [doc['image'].convert("RGB")]


def visulogic_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc['question']
    return question + lmms_eval_specific_kwargs.get('post_prompt', '')


def visulogic_process_results(doc, results):
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


def visulogic_aggregate_results(results):
    correct = sum(results)
    total = len(results)
    return correct / total 


