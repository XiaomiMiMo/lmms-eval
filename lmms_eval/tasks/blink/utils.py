# Copyright 2025 Xiaomi Corporation.
import os
import ast
import io
import base64
import random
import numpy as np
from PIL import Image
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter


def blink_doc_to_visual(doc):
    images = []
    for i in range(1, 5):
        image = doc[f"image_{i}"]
        if image is not None:
            images.append(image.convert("RGB"))
    return images

def blink_doc_to_text(doc):
    return doc["prompt"] + "\n" + "Answer with the option's letter from the given choices directly."

def blink_doc_to_text_boxed(doc):
    return doc["prompt"] + "\n" + "Put your final answer in \\boxed{}."

def blink_process_results(doc, results):
    parsed_preds = []
    correct = False
    for pred in results:
        if isinstance(doc['choices'], list):
            choices = doc['choices']
        else:
            choices = ast.literal_eval(doc["choices"])
        index2ans, all_choices = get_multi_choice_info(choices)
        parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
        if parsed_pred == doc["answer"][1]:
            correct = True
        parsed_preds.append(parsed_pred)

    blink_acc = {"answer": doc["answer"], "parsed_pred": parsed_preds, "score": (1 if correct else 0)}
    return {"accuracy": blink_acc}

def blink_aggregate_results(results):
    score = 0
    if len(results) > 0:
        for result in results:
            score += result["score"]
        score /= len(results)
    return score


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/data_utils.py#L54
    """

    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index
