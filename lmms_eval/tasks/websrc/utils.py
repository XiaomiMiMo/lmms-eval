import ast
import base64
import io
import json
import os
import random
import re
from collections import defaultdict

import numpy as np
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter


def websrc_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    if lmms_eval_specific_kwargs and "post_prompt" in lmms_eval_specific_kwargs:
        question += lmms_eval_specific_kwargs["post_prompt"]
    return question


def websrc_doc_to_visual(doc):
    img_bs64 = doc["image"]
    img = Image.open(io.BytesIO(base64.b64decode(img_bs64)))
    del doc["image"]
    return [img]

import re
def compute_f1(sa, sb):

    def _normalize_str(string):
        # lower it
        string = string.lower()

        # strip leading and trailing whitespaces
        string = string.strip()
        
        # remove trailing punctuation marks
        # 处理常见的句末标点符号，包括英文和中文
        while string and string[-1] in '.。!！?？':
            string = string[:-1].strip()

        return string

    def _tokenize(text):
        # Regex pattern to match words and isolate punctuation
        pattern = r"\w+|[^\w\s]"
        tokens = re.findall(pattern, text)
        return tokens

    sa = _normalize_str(sa)
    sb = _normalize_str(sb)

    sa = _tokenize(sa)
    sb = _tokenize(sb)

    sa = set(sa)
    sb = set(sb)

    if len(sa) == 0 or len(sb) == 0:
        return 0.0

    comm = sa.intersection(sb)
    prec = len(comm) / len(sb)
    rec = len(comm) / len(sa)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
    return f1

def websrc_process_results(doc, results):
    pred = results[0]
    parsed_pred = pred
    id = doc["page_id"]
    websrc_ans = {"id": id, "domain": doc["domain"], "parsed_pred": parsed_pred}
    if "answer" in doc:
        websrc_ans["answer"] = doc["answer"]
        f1_score = compute_f1(websrc_ans["answer"], websrc_ans["parsed_pred"])
        websrc_ans["f1_score"] = f1_score

    if "id" in doc:
        websrc_ans["question_id"] = doc["id"]

    return {
        "websrc_squad_f1": websrc_ans,
        "submission": (
            {
                websrc_ans["question_id"]: pred,
            }
            if "question_id" in websrc_ans
            else None
        ),
    }


def websrc_test_aggregate_results_for_submission(results, args):
    path = generate_submission_file("websrc_test_for_submission.json", args)
    with open(path, "w") as f:
        out = {}
        for result in results:
            out.update(result)
        json.dump(out, f, indent=4)
    eval_logger.info(f"Results saved to {path}.")


def websrc_aggregate_results(results):
    evaluation_result = {}

    # Group results by domain
    subset_to_eval_samples = defaultdict(list)
    for result in results:
        subset_to_eval_samples[result["domain"]].append(result)

    # Evaluate each domain
    for subset, sub_eval_samples in subset_to_eval_samples.items():
        # judge_dict, metric_dict = evaluate_websrc(sub_eval_samples)
        metric_list = [sample['f1_score'] for sample in sub_eval_samples]
        metric_dict = {"f1": np.mean(metric_list), "num_example": len(sub_eval_samples)}
        evaluation_result[subset] = metric_dict

    # Aggregate results for all domains
    printable_results = {}
    for domain in DOMAINS:
        if domain not in evaluation_result:
            continue
        printable_results[domain] = {
            "num": int(evaluation_result[domain]["num_example"]),
            "f1": round(evaluation_result[domain]["f1"], 3),
        }
    all_ins_f1 = np.sum([cat_results["f1"] * cat_results["num_example"] for cat_results in evaluation_result.values()]) / sum([cat_results["num_example"] for cat_results in evaluation_result.values()])
    printable_results["Overall"] = {
        "num": sum([cat_results["num_example"] for cat_results in evaluation_result.values()]),
        "f1": round(all_ins_f1, 3),
    }
    print(printable_results)
    return printable_results["Overall"]["f1"]


##################
# Helper functions written by official MMMU repo.
##################
DOMAINS = [
    "auto",
    "book",
    "camera",
    "game",
    "jobs",
    "movie",
    "phone",
    "restaurant",
    "sports",
    "university",
    "hotel",
]

