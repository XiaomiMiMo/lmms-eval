# Copyright 2025 Xiaomi Corporation.

import os
import time
import requests
from loguru import logger as eval_logger
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter
from rouge import Rouge
import numpy as np
import re
import sys
sys.setrecursionlimit(100000)  # 设置为10000或更高

WEB_CAPTION_PROMPT = """You are given a screenshot of a webpage. Please generate the meta web description information of this webpage, i.e., content attribute in <meta name="description" content=""> HTML element.

You should use the following format, and do not output any explanation or any other contents:
<meta name="description" content="YOUR ANSWER">
"""

HEADING_OCR_PROMPT = """You are given a screenshot of a webpage. Please generate the main text within the screenshot, which can be regarded as the heading of the webpage.

You should directly tell me the main content, and do not output any explanation or any other contents.
"""

WEBQA_PROMPT = """{question}
You should directly tell me your answer in the fewest words possible, and do not output any explanation or any other contents.
"""

ELEMENT_OCR_PROMPT = """You are given a screenshot of a webpage with a red rectangle bounding box. The [x1, y1, x2, y2] coordinates of the bounding box is {bbox_ratio}.
Please perform OCR in the bounding box and recognize the text content within the red bounding box.

You should use the following format:
The text content within the red bounding box is: <YOUR ANSWER>
"""

"""You are given a screenshot of a webpage with a red rectangle bounding box. The [x1, y1, x2, y2] coordinates of the bounding box is [0.20, 0.39, 0.80, 0.41].
Please perform OCR on the bounding box and recognize the text content within the red bounding box.

You should use the following format:
The text content within the red bounding box is: <YOUR ANSWER>
"""

"output_v7/long_text_OCR/annotated_images/iheartdogs.com_start6864_annotated53.png"

ELEMENT_GROUND_PROMPT = """In this website screenshot, I have labeled IDs for some HTML elements as candicates. Tell me which one best matches the description: {element_desc}

You should directly tell me your choice in a single uppercase letter, and do not output any explanation or any other contents.
"""

ACTION_PREDICTION_PROMPT = """You are given a screenshot of a webpage with a red rectangle bounding box. The [x1, y1, x2, y2] coordinates of the bounding box is {bbox_ratio}.
Please select the best webpage description that matches the new webpage after clicking the selected element in the bounding box:
{choices_text}

You should directly tell me your choice in a single uppercase letter, and do not output any explanation or any other contents.
"""

ACTION_GROUND_PROMPT = """In this website screenshot, I have labeled IDs for some HTML elements as candicates. Tell me which one I should click to complete the following task: {instruction}

You should directly tell me your choice in a single uppercase letter, and do not output any explanation or any other contents.
"""


DEFAULT_PROMPTS = {
    "web_caption_prompt": WEB_CAPTION_PROMPT,
    "heading_ocr_prompt": HEADING_OCR_PROMPT,
    "webqa_prompt": WEBQA_PROMPT,
    "element_ocr_prompt": ELEMENT_OCR_PROMPT,
    "element_ground_prompt": ELEMENT_GROUND_PROMPT,
    "action_prediction_prompt": ACTION_PREDICTION_PROMPT,
    "action_ground_prompt": ACTION_GROUND_PROMPT,
}

def eval_web_caption(preds, golds, **kwargs):
    assert len(preds) == len(golds)
    for i in range(len(preds)):
        if not preds[i]:
            preds[i] = " "
        preds[i] = re.sub(r"<think>.*?</think>", "", preds[i], flags=re.DOTALL).strip()
        if not preds[i].strip():
            preds[i] = " "

    rouge = Rouge(metrics=['rouge-1', 'rouge-2', 'rouge-l'])
    scores = rouge.get_scores(preds, golds, avg=True)
    return dict(
        rouge_1=scores['rouge-1']['f'] * 100,
        rouge_2=scores['rouge-2']['f'] * 100,
        rouge_l=scores['rouge-l']['f'] * 100
    )

def eval_heading_ocr(preds, golds, **kwargs):
    assert len(preds) == len(golds)
    for i in range(len(preds)):
        preds[i] = re.sub(r"<think>.*?</think>", "", preds[i], flags=re.DOTALL).strip()
        if not preds[i]:
            preds[i] = " "

    rouge = Rouge(metrics=['rouge-1', 'rouge-2', 'rouge-l'])
    scores = rouge.get_scores(preds, golds, avg=True)
    return dict(
        rouge_1=scores['rouge-1']['f'] * 100,
        rouge_2=scores['rouge-2']['f'] * 100,
        rouge_l=scores['rouge-l']['f'] * 100
    )

def eval_webqa(preds, golds, **kwargs):
    f1_scores = []
    rouge = Rouge(metrics=['rouge-1'])
    for pred, gold_list in zip(preds, golds):
        pred = re.sub(r"<think>.*?</think>", "", pred, flags=re.DOTALL).strip()
        try:
            if not pred:
                pred = " "
            cur_f1 = max([rouge.get_scores([pred], [gold], avg=True)['rouge-1']['f'] for gold in gold_list])
            f1_scores.append(cur_f1)
        except:
            pass

    return dict(
        f1=sum(f1_scores) / len(f1_scores) * 100
    )
import re 
def eval_element_ocr(preds, golds, **kwargs):
    assert len(preds) == len(golds)
    for i in range(len(preds)):
        if not preds[i] or len(preds[i]) == 1:
            preds[i] = " "
        # remove <think> </think>
        preds[i] = re.sub(r"<think>.*?</think>", "", preds[i], flags=re.DOTALL).strip() 
        # remove all text before "The text content within the red bounding box is:"
        preds[i] = re.sub(r".*The text content within the red bounding box is:", "", preds[i])
        preds[i] = preds[i].strip()
        if not preds[i].strip():
            preds[i] = " "

    rouge = Rouge(metrics=['rouge-1', 'rouge-2', 'rouge-l'])
    scores = rouge.get_scores(preds, golds, avg=True)
    return dict(
        rouge_1=scores['rouge-1']['f'] * 100,
        rouge_2=scores['rouge-2']['f'] * 100,
        rouge_l=scores['rouge-l']['f'] * 100
    )

def eval_element_ground(preds, golds, **kwargs):
    results = []
    for pred, gold in zip(preds, golds):
        cur_pred = parse_multi_choice_response(pred, [chr(ord('A')+i) for i in range(8)])
        try:
            if ord('A') <= ord(cur_pred) <= ord('Z'):
                cur_pred = ord(cur_pred) - ord('A')
            else:
                cur_pred = -1
        except:
            cur_pred = -1
        results.append(cur_pred == gold)

    return dict(
        accuracy=sum(results) / len(results) * 100
    )


def eval_action_prediction(preds, golds, **kwargs):
    results = []
    for pred, gold in zip(preds, golds):
        cur_pred = parse_multi_choice_response(pred, [chr(ord('A')+i) for i in range(8)])
        try:
            if ord('A') <= ord(cur_pred) <= ord('Z'):
                cur_pred = ord(cur_pred) - ord('A')
            else:
                cur_pred = -1
        except:
            cur_pred = -1
        results.append(cur_pred == gold)

    return dict(
        accuracy=sum(results) / len(results) * 100
    )

def eval_action_ground(preds, golds, **kwargs):
    results = []
    for pred, gold in zip(preds, golds):
        cur_pred = parse_multi_choice_response(pred, [chr(ord('A')+i) for i in range(8)])
        try:
            if ord('A') <= ord(cur_pred) <= ord('Z'):
                cur_pred = ord(cur_pred) - ord('A')
            else:
                cur_pred = -1
        except:
            cur_pred = -1
        results.append(cur_pred == gold)

    return dict(
        accuracy=sum(results) / len(results) * 100
    )

eval_metric = {
    "web_caption": eval_web_caption,
    "heading_ocr": eval_heading_ocr,
    "webqa": eval_webqa,
    "element_ocr": eval_element_ocr,
    "element_ground": eval_element_ground,
    "action_prediction": eval_action_prediction,
    "action_ground": eval_action_ground,
}

# ----------- Process Multi-choice -------------
def parse_multi_choice_response(response: str, all_choices):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    # Remove <think> tags and their content
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    if len(response) == 1:
        return response.upper()
    elif not response:
        return 'a'
    elif re.match(r"[A-Z]\.", response):
        return response[0]

    for char in [',', '.', '!', '?', ';', ':', "'", '"']:
        response = response.replace(char, "")
    response = " " + response + " " # add space to avoid partial match

    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f'({choice})' in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices: # e.g., A B C D
            if f' {choice} ' in response:
                candidates.append(choice)

    if len(candidates) == 0:  # still not get answer
        # pred_index = random.choice(all_choices)
        pred_index = "z"
    elif len(candidates) > 1:
        start_indexes = []
        if ans_with_brack: 
            for can in candidates:
                index = response.rfind(f'({can})')
                start_indexes.append(index) # -1 will be ignored anyway
            # start_indexes = [generated_response.index(f'({can})') for can in candidates]
        else:
            for can in candidates:
                index = response.rfind(f" {can} ")
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else: # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index

def visualwebbench_doc_to_visual(doc):
    # Image is presented as is
    image = doc["image"].convert("RGB")
    return [image.convert("RGB")]


def visualwebbench_doc_to_text(doc):
    prompt = DEFAULT_PROMPTS[f"{doc['task_type']}_prompt"]
    task_type = doc["task_type"]
    if task_type in ["web_caption", "heading_ocr"]:
        pass # doing nothing
    elif task_type == "webqa":
        prompt = prompt.format(question=doc["question"])
    elif task_type == "element_ocr":
        prompt = prompt.format(bbox_ratio=doc["bbox"])
    elif task_type == "element_ground":
        prompt = prompt.format(element_desc=doc["elem_desc"])
    elif task_type == "action_prediction":
        option_list = []
        for idx, option in enumerate(doc["options"]):
            option_list.append(f"({chr(ord('A') + idx)}): {option}")
        prompt = prompt.format(bbox_ratio=doc["bbox"], choices_text=option_list)
    elif task_type == "action_ground":
        prompt = prompt.format(instruction=doc["instruction"])
    return prompt


def visualwebbench_process_result(doc, result):
    task_type = doc["task_type"]
    score_dict = eval_metric[task_type](result, [doc["answer"]])
    return score_dict

    # if task_type in ["webqa", "element_ground", "action_ground", "action_prediction"]:
    #     accuracy = eval_metric[task_type](result, [doc["answer"]])
    #     return {"accuracy": accuracy}
    # else:
    #     rouge_results = eval_metric[task_type](result, [doc["answer"]])
    #     return {metric: rouge_results[metric] for metric in rouge_results}

def visualwebbench_aggregate_result(results):
    return np.mean(results)