# Copyright 2025 Xiaomi Corporation.

import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "mmbench_video.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]

# {'index': 0,
#  'video': 'wZxzBvAgqxc',
#  'video_type': 'Sports',
#  'question': 'What is the name of the player who scored the first goal in the video?',
#  'answer': 'Palmer.',
#  'dimensions': "['OCR', 'Attribute Recognition']",
#  'video_path': './video/wZxzBvAgqxc.mp4'}


from lmms_eval.tasks._task_utils.gpt_eval_utils import OpenAIClient
from loguru import logger as eval_logger
import time
from functools import partial

API_TYPE = os.getenv("API_TYPE", None)
MODEL_VERSION = os.getenv("MODEL_VERSION", None)
if API_TYPE == 'openai':
    API_URL = os.getenv("OPENAI_API_URL", "YOUR_API_URL")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    client = OpenAIClient(api_url=API_URL, api_key=API_KEY, model=MODEL_VERSION, task="mmbench_video")
else:
    raise ValueError(f"Invalid API type: {API_TYPE}")



def mmbench_video_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = os.path.join(cache_dir, doc["video_path"])
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    elif os.path.exists(video_path.replace("mp4", "mkv")):
        video_path = video_path.replace("mp4", "mkv")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def mmbench_video_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    prompt = question
    if "post_prompt" in lmms_eval_specific_kwargs:
        prompt += "\n" + lmms_eval_specific_kwargs["post_prompt"]
    return prompt


JUDGE_SYSTEM_PROMPT = """As an AI assistant, your task is to evaluate a candidate answer in comparison to a given correct answer.
The question itself, the correct 'groundtruth' answer, and the candidate answer will be provided to you.
Your assessment should range from 0 to 3, based solely on the semantic similarity between the groundtruth and the candidate answer, disregarding any grammatical differences.
A rating of 0 suggests no similarity, implying the candidate answer is entirely incorrect.
A rating of 1 suggests low similarity, meaning the candidate answer is largely incorrect.
A rating of 2 suggests high similarity, meaning the candidate answer is largely correct.
Lastly, a rating of 3 indicates complete similarity, which means the candidate answer is entirely correct.
Your response should be a single integer from 0, 1, 2, or 3.
"""


def mmbench_video_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]
    question = doc["question"]
    ground_truth = doc["answer"]
    eval_prompt = f"Question: {question}\nGround Truth Answer: {ground_truth}\nCandidate Answer: {pred}\nScore: "
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT}, 
        {"role": "user", "content": eval_prompt}
    ]
    generation_kwargs = {
        "max_tokens": 16,
        "top_p": 1,
    }
    eval_res = client.get_chat_response(
        messages, 
        default_response=None, 
        postprocess_response=lambda x: int(x.strip()), 
        generation_kwargs=generation_kwargs
    )
    if eval_res is None:
        score = 0
    else:
        score = eval_res
    data_dict = {"video_type": doc["video_type"], "dimensions": doc["dimensions"], "pred_answer": pred, "answer": ground_truth, "score": score}

    return {f"gpt_eval_score": data_dict}


def mmbench_video_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    scores = [result['score'] for result in results]
    acc = sum(scores) / len(scores) / 3 if len(scores) > 0 else 0
    return acc
