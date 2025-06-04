import datetime
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml
from decord import VideoReader, cpu
from loguru import logger as eval_logger

import lmms_eval.tasks._task_utils.file_utils as file_utils

# with open(Path(__file__).parent / "_default_template.yaml", "r") as f:
#     raw_data = f.readlines()
#     safe_data = []
#     for i, line in enumerate(raw_data):
#         # remove function definition since yaml load cannot handle it
#         if "!function" not in line:
#             safe_data.append(line)

#     config = yaml.safe_load("".join(safe_data))


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
# cache_dir = os.path.join(hf_home, cache_dir)
# base_cache_dir = config["dataset_kwargs"]["cache_dir"]
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "charades.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


# DATA_LIST = {
#     "charades": 'your_data_dir/Charades/',
# }
# Pass in video path here
# Can only work correctly with video llm
def temporal_grounding_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    video_path = doc["video"]
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = os.path.join(cache_dir, "Charades_v1_480", video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif "s3://" not in video_path:
        sys.exit(f"video path:{video_path} does not exist, please check")

    return [video_path]


# This is the place where you format your question
def temporal_grounding_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    question = doc["caption"]

    return f"{pre_prompt}{question}. {post_prompt}"


def temporal_grounding_doc_to_answer(doc):
    return doc["timestamp"]


def iou(A, B):
    max0 = max((A[0]), (B[0]))
    min0 = min((A[0]), (B[0]))
    max1 = max((A[1]), (B[1]))
    min1 = min((A[1]), (B[1]))
    # Ensure result is a regular Python float, not float16
    return float(max(min1 - max0, 0) / (max1 - min0))

    # # hacked!
    # return float(max(min1 - max0, 0) / (B[1] - B[0]))

def extract_time_from_text(paragraph):
    prompt = "A specific example is : 20.8 - 30.0 seconds".lower()
    paragraph = paragraph.lower().replace(prompt, "").replace("to", "-")
    # Split text into sentences based on common delimiters
    sentences = re.split(r"[!?\n]", paragraph)

    # Keywords that might indicate the presence of time information
    keywords = ["starts", "ends", "happens in", "start time", "end time", "start", "end", "happen"]
    # filter sentences by keywords
    candidates = []
    for sentence in sentences:
        # If sentence contains one of the keywords
        if any(keyword in sentence for keyword in keywords):
            candidates.append(sentence)

    timestamps = []
    # Check for The given query happens in m - n (seconds)
    patterns = [r"(\d+\.*\d*)\s*-\s*(\d+\.*\d*)"]

    for time_pattern in patterns:
        time_matches = re.findall(time_pattern, paragraph)
        if time_matches:
            timestamps = [[float(start), float(end)] for start, end in time_matches]

    if len(sentences) == 0:
        return []
    # check for other formats e.g.:
    # 1 .Starting time: 0.8 seconds
    # Ending time: 1.1 seconds
    # 2. The start time for this event is 0 seconds, and the end time is 12 seconds.
    if len(timestamps) == 0:
        times = []
        time_regex = re.compile(r"\b(\d+\.\d+\b|\b\d+)\b")  # time formats (e.g., 18, 18.5)
        for sentence in candidates:
            time = re.findall(time_regex, sentence)
            if time:
                time_in_sec = float(time[0])
                times.append(time_in_sec)
        times = times[: len(times) // 2 * 2]
        timestamps = [(times[i], times[i + 1]) for i in range(0, len(times), 2)]
    # Check for  examples like:
    # 3. The event 'person flipped the light switch near the door' starts at 00:00:18 and ends at 00:00:23.
    if len(timestamps) == 0:
        times = []
        time_regex = re.compile(r"\b((\d{1,2}:\d{2}:\d{2}))\b")  # time formats (e.g., 18:00, 00:18:05)
        for sentence in candidates:
            time = re.findall(time_regex, sentence)
            if time:
                t = time[0][0]  # Extract the actual time string from the tuple
            else:
                continue
            # If time is in HH:MM:SS format, convert to seconds
            if t.count(":") == 2:
                h, m, s = map(int, t.split(":"))
                time_in_sec = h * 3600 + m * 60 + s
            elif t.count(":") == 1:
                m, s = map(int, t.split(":"))
                time_in_sec = m * 60 + s
            times.append(time_in_sec)
        times = times[: len(times) // 2 * 2]
        timestamps = [(times[i], times[i + 1]) for i in range(0, len(times), 2)]
    results = []
    for start, end in timestamps:
        if end > start:
            results.append([start, end])
        else:
            results.append([end, start])
    if len(results) > 1:
        results = results[:1]
    return results

# Process result for mcq answer generation
import re
def temporal_grounding_process_results_generation(doc, result):
    pred = result[0]

    if 'boxed' in pred:
        # Extract timestamp from \boxed{[mm:ss, mm:ss]} format
        boxed_pattern = r'\\boxed\{\[(\d{1,2}):(\d{1,2}),\s*(\d{1,2}):(\d{1,2})\]\}'
        boxed_match = re.search(boxed_pattern, pred)
        
        if boxed_match:
            start_min, start_sec, end_min, end_sec = map(int, boxed_match.groups())
            start_timestamp = start_min * 60 + start_sec
            end_timestamp = end_min * 60 + end_sec
        else:
            start_timestamp = 0
            end_timestamp = 0
    elif '[' in pred:
        # Extract timestamps from [mm:ss, mm:ss] format
        timestamp_pattern = r'\[(\d{1,2}):(\d{1,2}),\s*(\d{1,2}):(\d{1,2})\]'
        timestamp_match = re.search(timestamp_pattern, pred)
        
        if timestamp_match:
            start_min, start_sec, end_min, end_sec = map(int, timestamp_match.groups())
            start_timestamp = start_min * 60 + start_sec
            end_timestamp = end_min * 60 + end_sec
        else:
            start_timestamp = 0
            end_timestamp = 0
    else:
        # Extract timestamps from natural language prediction
        extracted_timestamps = extract_time_from_text(pred)
        if len(extracted_timestamps) == 1:
            start_timestamp, end_timestamp = extracted_timestamps[0]
        else:
            # If no valid timestamp found, use default values
            start_timestamp = 0
            end_timestamp = 0

    # Convert float16 to regular Python floats to make them JSON serializable
    gt_timestamp = doc["timestamp"]
    if hasattr(gt_timestamp, 'tolist'):
        # Handle numpy arrays or tensors
        gt_timestamp = gt_timestamp.tolist()
    elif isinstance(gt_timestamp, (list, tuple)):
        # Handle lists/tuples that might contain float16 values
        gt_timestamp = [float(x) for x in gt_timestamp]
    else:
        # Handle single values
        gt_timestamp = float(gt_timestamp)
    
    result = {
        "query": f'{doc["video"]}>>>{doc["caption"]}>>>{gt_timestamp}',
        "gt": gt_timestamp,
        "pred": [start_timestamp, end_timestamp],
        'iou': iou(gt_timestamp, [start_timestamp, end_timestamp]),
    }

    return {
        "iou_0.3": result,
        "iou_0.5": result,
        "iou_0.7": result,
        "m_iou": result,
    }


def temporal_grounding_aggregate_charades_iou_threshold(results, args, threshold):
    ious = []
    for result in results:
        ious.append(result['iou'])

    success_cnt = 0
    for cur_iou in ious:
        if cur_iou >= threshold:
            success_cnt += 1

    return float(success_cnt * 100 / len(ious))

def temporal_grounding_aggregate_charades_iou_03(results, args):
    return temporal_grounding_aggregate_charades_iou_threshold(results, args, 0.3)

def temporal_grounding_aggregate_charades_iou_05(results, args):
    return temporal_grounding_aggregate_charades_iou_threshold(results, args, 0.5)

def temporal_grounding_aggregate_charades_iou_07(results, args):
    return temporal_grounding_aggregate_charades_iou_threshold(results, args, 0.7)

def temporal_grounding_aggregate_charades_m_iou(results, args):
    ious = []
    for result in results:
        ious.append(result['iou'])

    return float(sum(ious) * 100 / len(ious))


def temporal_grounding_aggregate_submissions(results, args, task):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    submission_file_name = f"inference_results_temporal_grounding_{task}_{now_date_time}.json"
    path = file_utils.generate_submission_file(submission_file_name, args)

    # results is a list of 5031 dict,
    # need to convert results into a single dict with 5031 key-value pairs
    combined_submission = {}

    for submission_dict in results:
        combined_submission.update(submission_dict)

    with open(path, "w") as f:
        json.dump(combined_submission, f, indent=4)

    eval_logger.info(f"Submission file saved to {path}")
