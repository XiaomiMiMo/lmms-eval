import json
import os
import time
from loguru import logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks._task_utils.gpt_eval_utils import OpenAIClient
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter

API_TYPE = os.getenv("API_TYPE", None)
MODEL_VERSION = os.getenv("MODEL_VERSION", None)
if API_TYPE == 'openai':
    API_URL = os.getenv("OPENAI_API_URL", "YOUR_API_URL")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    client = OpenAIClient(api_url=API_URL, api_key=API_KEY, model=MODEL_VERSION, task="default")
else:
    raise ValueError(f"Invalid API type: {API_TYPE}")


def docvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def docvqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"

def postprocess_judge_response(judge_response):
    judge_response = judge_response.strip()
    score = float(extract_final_boxed_content(judge_response))
    assert score in [0.0, 1.0]
    return score, judge_response

from lmms_eval.tasks._task_utils.eval_utils import extract_final_boxed_content
def get_gpt_eval_score(question, answer, pred):
    global client
    
    if isinstance(answer, list):
        answer_list = answer
    else:
        answer_list = [answer]
    score, judge_response = None, None
    for answer in answer_list:
        if pred.lower() == answer.lower() or pred.lower().rstrip('.') == answer.lower():
            score = 1.0
    
    if score is None:
        score, judge_response = client.evaluate_correctness(
            question, answer, pred,
            postprocess_response=postprocess_judge_response,
            default_response=(0.0, "")
        )
    
    return score, judge_response


def docvqa_process_results_judge(doc, results):
    question = doc["question"]
    answer = doc["answers"]
    pred = results[0]
    
    score, judge_response = get_gpt_eval_score(question, answer, pred)
    return {"gpt_eval_score": {"questionId": int(doc["questionId"]), "question": question, "answer": answer, "pred": pred, "score": score, "judge_response": judge_response}}



def docvqa_aggregate_result_judge(results):
    scores = []
    for result in results:
        scores.append(result["score"])
    if len(scores) > 0:
        return sum(scores) / len(scores)
    else:
        return 0.0


def docvqa_test_process_results(doc, results):
    pred = results[0]
    questionId = doc["questionId"]
    return {"anls": {"questionId": int(questionId), "answer": pred}, "submission": {"questionId": int(questionId), "answer": pred}}


def docvqa_test_aggregate_results(results, args):
    # save results as json
    path = generate_submission_file("docvqa_test_for_submission.json", args)
    with open(path, "w") as f:
        json.dump(results, f)
    logger.info(f"Results saved to {path}")
