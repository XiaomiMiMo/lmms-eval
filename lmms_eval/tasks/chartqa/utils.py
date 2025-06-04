def chartqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def chartqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def chartqa_process_results(doc, results):
    pred = results[0].strip(".")
    type = doc["type"]
    score = relaxed_correctness(pred, doc["answer"])
    score = 1.0 if score else 0.0
    return_dict = {"relaxed_overall": score}
    if type == "human_test":
        return_dict["relaxed_human_split"] = score
    else:
        return_dict["relaxed_augmented_split"] = score
    return return_dict



import os
import json
import time
from lmms_eval.tasks._task_utils.gpt_eval_utils import OpenAIClient

API_TYPE = os.getenv("API_TYPE", None)
MODEL_VERSION = os.getenv("MODEL_VERSION", None)
if API_TYPE == 'openai':
    API_URL = os.getenv("OPENAI_API_URL", "YOUR_API_URL")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    client = OpenAIClient(api_url=API_URL, api_key=API_KEY, model=MODEL_VERSION, task="default-vl")
else:
    raise ValueError(f"Invalid API type: {API_TYPE}")


def postprocess_judge_response(judge_response):
    judge_response = judge_response.strip()
    score = float(extract_final_boxed_content(judge_response))
    assert score in [0.0, 1.0]
    return score, judge_response


def chartqa_process_results_judge(doc, results):
    global client

    pred = results[0]
    type = doc["type"]
    question = doc["question"]
    answer = doc["answer"].lstrip("[").rstrip("]")
    images = [doc["image"].convert("RGB")]

    score, judge_response = None, None
    if pred.lower() == answer.lower() or pred.lower().rstrip('.') == answer.lower():
        score = 1.0
    else:
        score, judge_response = client.evaluate_correctness_vl(
            question, answer, pred, images,
            postprocess_response=postprocess_judge_response,
            default_response=(0.0, "")
        )
    
    item = {"question": question, "answer": answer, "pred": pred, "score": score, "judge_response": judge_response}
    return_dict = {"relaxed_overall": item}
    if type == "human_test":
        return_dict["relaxed_human_split"] = item
    else:
        return_dict["relaxed_augmented_split"] = item
    return return_dict

def chartqa_aggregate_result_judge(results):
    score = 0.0
    for result in results:
        score += result["score"]
    score /= len(results)
    return score


from lmms_eval.tasks._task_utils.eval_utils import extract_final_boxed_content
def chartqa_process_results_boxed(doc, results):
    pred = extract_final_boxed_content(results[0])
    type = doc["type"]
    score = custom_correctness(pred, doc["answer"])
    score = 1.0 if score else 0.0
    return_dict = {"relaxed_overall": score}
    if type == "human_test":
        return_dict["relaxed_human_split"] = score
    else:
        return_dict["relaxed_augmented_split"] = score
    return return_dict


def relaxed_correctness(prediction, target, max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    This funcion is taken from https://github.com/QwenLM/Qwen-VL/blob/34b4c0ee7b07726371b960911f249fe61b362ca3/eval_mm/evaluate_vqa.py#L113
    Args:
      target: List of target string.
      prediction: List of predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str):
        try:
            if text.endswith("%"):
                # Convert percentages to floats.
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


from loguru import logger as eval_logger
def custom_correctness(prediction, target, max_relative_change: float = 0.05):
    parsed_preds = []
    if prediction.startswith("\\text{") and prediction.endswith("}"):
        prediction = prediction[6:-1]
    if prediction.endswith("%"):
        try:
            parsed_preds.append(str(float(prediction.rstrip("\%").rstrip("%")) / 100.0))
            parsed_preds.append(str(float(prediction.rstrip("\%").rstrip("%"))))
        except ValueError:
            parsed_preds.append(prediction)
    else:
        try:
            parsed_preds.append(str(float(prediction) * 100.0))
            parsed_preds.append(str(float(prediction)))
        except ValueError:
            parsed_preds.append(prediction)
    
    if target.endswith("%"):
        try:
            target_float = float(target.rstrip("\%").rstrip("%") / 100.0)
        except ValueError:
            target_float = None
    else:
        try:
            target_float = float(target)
        except ValueError:
            target_float = None
    
    for pred in parsed_preds:
        try:
            pred_float = float(pred)
        except ValueError:
            pred_float = None
        if prediction.endswith("%"):
            eval_logger.info(f"prediction: {prediction}, pred: {pred}, target: {target}, pred_float: {pred_float}, target_float: {target_float}")
        if target_float is not None and pred_float is not None:
            if abs(target_float) * max_relative_change >= abs(pred_float - target_float):
                return True
        else:
            if target.lower() in pred.lower():
                return True
    return False

