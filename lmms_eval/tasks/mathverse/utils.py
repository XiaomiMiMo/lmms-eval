import json
import os
from pathlib import Path
import re
import pandas as pd
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.mathverse.mathverse_evals import MathVerseEvaluator
from lmms_eval.tasks._task_utils.math_verify_utils import MathVerifyFn

with open(Path(__file__).parent / "mathverse.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

API_TYPE = os.getenv("API_TYPE", None)
MODEL_VERSION = os.getenv("MODEL_VERSION", None)
if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "YOUR_API_URL")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
else:
    raise ValueError(f"Invalid API type: {API_TYPE}")

mathverse_evaluator = MathVerseEvaluator(api_type=API_TYPE, api_url=API_URL, api_key=API_KEY, gpt_model=MODEL_VERSION)


def mathverse_doc_to_visual(doc):
    if str(doc["image"]).strip() == "":
        return []
    return [doc["image"].convert("RGB")]


def mathverse_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    problem = {
        "question": doc["question"],
        "answer": doc["answer"] if "answer" in doc else None,
        "query_wo": doc["query_wo"],
        "query_cot": doc["query_cot"],
        "question_type": doc["question_type"],
        "problem_version": doc["problem_version"],
    }
    query_prompt = mathverse_evaluator.create_one_query(
        problem, examples=None, shot_num=0, shot_type=lmms_eval_specific_kwargs["shot_type"], hint=lmms_eval_specific_kwargs.get("hint", None), query_type=lmms_eval_specific_kwargs["query_type"]
    )
    return query_prompt


def mathverse_process_results(doc, results):
    prediction = results[0].strip()

    result = {
        "sample_index": doc["sample_index"],
        "problem_index": doc["problem_index"],
        "problem_version": doc["problem_version"],
        "question": doc["question"],
        "answer": doc["answer"] if "answer" in doc else None,
        "prediction": prediction,
        "question_type": doc["question_type"],
        "metadata": doc["metadata"],
        "query_wo": doc["query_wo"],
        "query_cot": doc["query_cot"],
    }
    result = mathverse_evaluator.eval_instance(result, config)

    return {
        "gpt_eval_score": result,
        "submission": result,
    }


math_verify_fn = MathVerifyFn()
from lmms_eval.tasks._task_utils.eval_utils import extract_final_boxed_content
def mathverse_boxed_process_results(doc, results):
    prediction = results[0].strip()
    math_verify_score, math_verify_ext = math_verify_fn(prediction, doc["answer"])

    prediction = extract_final_boxed_content(prediction).strip()
    result = {
        "sample_index": doc["sample_index"],
        "problem_index": doc["problem_index"],
        "problem_version": doc["problem_version"],
        "question": doc["question"],
        "answer": doc["answer"] if "answer" in doc else None,
        "prediction": prediction,
        "question_type": doc["question_type"],
        "metadata": doc["metadata"],
        "query_wo": doc["query_wo"],
        "query_cot": doc["query_cot"],
    }
    result = mathverse_evaluator.eval_instance(result, config)

    return {
        "gpt_eval_score": result,
        "submission": result,
        "math_verify": {
            "score": math_verify_score,
            "extraction": math_verify_ext
        }
    }


def mathverse_aggregate_results_submission(results, args, *, calculate_gain=False, random_scores=None):
    # Don't know why but this sometimes yields error so I hardcode it
    try:
        split_flag = results[0]["metadata"]["split"]
    except:
        split_flag = "testmini"
    path = generate_submission_file(f"mathverse_{split_flag}_results.json", args)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    eval_logger.info(f"Saved results to {path}")


def mathverse_aggregate_results_eval(results, args, *, calculate_gain=False, random_scores=None):
    split_flag = results[0]["metadata"]["split"]
    problem_version = results[0]["problem_version"].lower().replace(" ", "_")
    # save the result first, in case the gpt evaluation fails
    path = generate_submission_file(f"mathverse_{split_flag}_{problem_version}_results.json", args)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    # gpt evaluation
    results_dict, scores = mathverse_evaluator.eval_results(results, config)
    
    # save results
    path = generate_submission_file(f"mathverse_{split_flag}_{problem_version}_results.json", args)
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=4)
    # save scores
    path = generate_submission_file(f"mathverse_{split_flag}_{problem_version}_scores.json", args)
    with open(path, "w") as f:
        json.dump(scores, f, indent=4)
    eval_logger.info(f"Saved scores to {path}")
    if scores["average"]["accuracy"] == 0:
        return None
    return scores["average"]["accuracy"]


def mathverse_math_verify_aggregate_results(results, args):
    total = len(results)
    score = sum(result["score"] for result in results)
    return score / total