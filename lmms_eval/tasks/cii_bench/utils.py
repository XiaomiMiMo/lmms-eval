# Copyright 2025 Xiaomi Corporation.

import os, json
import re 
from typing import List
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter

# split test 
# Dataset({
#     features: ['id', 'image', 'question', 'option1', 'option2', 'option3', 'option4', 'option5', 'option6', 'correct_option', 'answer', 'image_type', 'difficulty', 'domain', 'emotion', 'rhetoric', 'explanation', 'metaphorical_meaning', 'local_path'],
#     num_rows: 765
# })


# sample


# {'id': 'test-522',
#  'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=650x813>,
#  'question': '戳破了屏幕有什么意义？',
#  'option1': '手指戳破屏幕是为了发泄人们的不满。',
#  'option2': '手指戳破屏幕代表着人们在虚拟世界中找到了新的入口。',
#  'option3': '手指戳破屏幕是为了测试手机屏幕的耐用性。',
#  'option4': '手指戳破屏幕的动作象征着人们在虚拟世界中的互动可能会对现实世界产生负面影响。',
#  'option5': '手指戳破屏幕代表着人们对手机功能的不满。',
#  'option6': '手指戳破屏幕意味着手机故障，需要维修。',
#  'correct_option': '手指戳破屏幕的动作象征着人们在虚拟世界中的互动可能会对现实世界产生负面影响。',
#  'answer': 'D',
#  'image_type': '插画(Illustration)',
#  'difficulty': '中等',
#  'domain': '社会',
#  'emotion': '中性',
#  'rhetoric': "{'choices': ['隐喻', '象征']}",
#  'explanation': '这幅图像展示了两只手指通过智能手机屏幕相互指向对方，屏幕被戳破并出现裂痕，碎片散落在地。',
#  'metaphorical_meaning': '手指戳破屏幕的动作象征着人们在虚拟世界中的互动可能会对现实世界产生负面影响。整体布局简洁明了，却充满了象征意义，暗示着科技进步与人际关系之间的复杂关系。',
#  'local_path': 'images/test/test-522.jpg'}


def cii_bench_doc_to_visual(doc):
    return [doc['image'].convert("RGB")]


def cii_bench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc['question']
    options = [doc['option1'], doc['option2'], doc['option3'], doc['option4'], doc['option5'], doc['option6']]
    options_str = "\n".join([f"{chr(i+65)}. {option}" for i, option in enumerate(options)])
    return f"{question}\nOptions:\n{options_str}" + lmms_eval_specific_kwargs.get('post_prompt', '')


def cii_bench_process_results(doc, results):
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


def cii_bench_aggregate_results(results):
    correct = sum(results)
    total = len(results)
    return correct / total if total > 0 else 0


