# Copyright 2025 Xiaomi Corporation.

import os, json
import re 
from typing import List
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter
# split test 


# DatasetDict({
#     zerobench: Dataset({
#         features: ['question_id', 'question_text', 'question_images_decoded', 'question_answer', 'question_images', 'image_attribution'],
#         num_rows: 100
#     })
#     zerobench_subquestions: Dataset({
#         features: ['question_id', 'question_text', 'question_images_decoded', 'question_answer', 'question_images', 'image_attribution'],
#         num_rows: 334
#     })
# })

#  {'question_id': '1',
#  'question_text': 'I want to purchase all the Montellier bottles from the top three shelves. How much do I save by purchasing the bottles with a loyalty card? Give your final answer in dollars.',
#  'question_images_decoded': [<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1500x1999>],
#  'question_answer': '11.90',
#  'question_images': ['images/1_0.png'],
#  'image_attribution': 'Own'}


def zerobench_doc_to_visual(doc):
    return [ img.convert("RGB") for img in doc['question_images_decoded']]


def zerobench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc['question_text']
    return question.strip() + lmms_eval_specific_kwargs.get('post_prompt', '')

def zerobench_process_results(doc, results):
    prediction = results[0]
    answer = doc["question_answer"] 
    score = 0 
    # exact match 
    if prediction.strip().lower() == answer.strip().lower() or prediction.strip().rstrip(".").lower() == answer.strip().lower():
        score = 1 
    return {"exact_match_accuracy": score}  # Return 0 if no valid letter found in either prediction or answer


def zerobench_aggregate_results(results):
    correct = sum(results)
    total = len(results)
    return correct / total 


