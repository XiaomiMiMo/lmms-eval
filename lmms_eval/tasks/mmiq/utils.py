# Copyright 2025 Xiaomi Corporation.

import os, json
import re 
from typing import List
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter
# split test 
# DatasetDict({
#     test: Dataset({
#         features: ['category', 'question', 'question_en', 'question_zh', 'image', 'MD5', 'data_id', 'answer', 'split'],
#         num_rows: 2710
#     })
# })


# Sample:
# {'category': 'Temporal Movement',
#  'question': 'Choose the most appropriate option from the four given choices to fill in the question mark, so that it presents a certain regularity:',
#  'question_en': 'Choose the most appropriate option from the four given choices to fill in the question mark, so that it presents a certain regularity:',
#  'question_zh': '从所给四个选项中，选择最合适的一个填入问号处，使之呈现一定的规律性：',
#  'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=435x486>,
#  'MD5': '6ac21bde690fbc63ac30fc509a841225',
#  'data_id': 0,
#  'answer': 'B',
#  'split': 'test'}

def mmiq_doc_to_visual(doc):
    return [doc['image'].convert("RGB")]


def mmiq_en_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc['question_en']
    return question + lmms_eval_specific_kwargs.get('post_prompt', '')

def mmiq_zh_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc['question_zh']
    return question + lmms_eval_specific_kwargs.get('post_prompt', '')

def mmiq_process_results(doc, results):
    prediction = results[0]
    answer = doc["answer"] 
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


def mmiq_aggregate_results(results):
    correct = sum(results)
    total = len(results)
    return correct / total 


