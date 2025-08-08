import os
from functools import partial
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter, extract_final_boxed_content, extract_after_think_content

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "vsibench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def vsibench_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["dataset"] + "/" + doc["scene_name"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        raise FileExistsError(f"video path:{video_path} does not exist.")
    return [video_path]


def vsibench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")

    if doc["question_type"] in NA_QUESTION_TYPES:
        post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "")
        return pre_prompt + question + post_prompt
    elif doc["question_type"] in MCA_QUESTION_TYPES:
        options = "\nOptions:\n" + "\n".join(doc["options"])
        post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "")
        return "".join([pre_prompt, question, options, post_prompt])
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")


def fuzzy_matching(pred):
    return pred.split(" ")[0].rstrip(".").strip()


def exact_match(pred, target):
    return 1.0 if pred.lower() == target.lower() else 0.0


def abs_dist_norm(pred, target):
    return abs(pred - target) / target


def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()


def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred


def vsibench_process_results(doc, results):
    doc["prediction"] = results[0]
    if doc["question_type"] in MCA_QUESTION_TYPES:
        doc["score"] = exact_match(fuzzy_matching(doc["prediction"]), doc["ground_truth"])
    elif doc["question_type"] in NA_QUESTION_TYPES:
        try:
            doc["score"] = mean_relative_accuracy(to_float(fuzzy_matching(doc["prediction"])), to_float(doc["ground_truth"]), 0.5, 0.95, 0.05)
        except TypeError:
            doc["score"] = 0.0
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

    return {"vsibench_score": doc}


def vsibench_process_results_boxed(doc, results):
    pred = extract_after_think_content(results[0])
    pred = extract_final_boxed_content(pred)
    doc["prediction"] = pred
    if doc["question_type"] in MCA_QUESTION_TYPES:
        doc["score"] = exact_match(fuzzy_matching(doc["prediction"]), doc["ground_truth"])
    elif doc["question_type"] in NA_QUESTION_TYPES:
        try:
            doc["score"] = mean_relative_accuracy(to_float(fuzzy_matching(doc["prediction"])), to_float(doc["ground_truth"]), 0.5, 0.95, 0.05)
        except TypeError:
            doc["score"] = 0.0
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

    return {"vsibench_score": doc}


def vsibench_aggregate_results(results):
    results = pd.DataFrame(results)

    output = {}

    for question_type, question_type_indexes in results.groupby("question_type").groups.items():
        per_question_type = results.iloc[question_type_indexes]

        if question_type in MCA_QUESTION_TYPES:
            output[f"{question_type}_score"] = per_question_type["score"].mean()
        elif question_type in NA_QUESTION_TYPES:
            output[f"{question_type}_score"] = per_question_type["score"].mean()

        else:
            raise ValueError(f"Unknown question type: {question_type}")

    output["object_rel_direction_score"] = (
        sum(
            [
                output.pop("object_rel_direction_easy_score"),
                output.pop("object_rel_direction_medium_score"),
                output.pop("object_rel_direction_hard_score"),
            ]
        )
        / 3.0
    )

    output["overall"] = sum([_ for _ in output.values()]) / len(output)
    eval_logger.info(f"Evaluation results: {output}")
    return output["overall"].item()