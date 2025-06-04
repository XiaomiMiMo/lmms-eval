
import os
import json
import time
from loguru import logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter, extract_final_boxed_content

API_TYPE = os.getenv("API_TYPE", None)
MODEL_VERSION = os.getenv("MODEL_VERSION", None)
if API_TYPE == 'openai':
    from lmms_eval.tasks._task_utils.gpt_eval_utils import OpenAIClient
    API_URL = os.getenv("OPENAI_API_URL", "YOUR_API_URL")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    client = OpenAIClient(api_url=API_URL, api_key=API_KEY, model=MODEL_VERSION, task="ocrbench")
else:
    raise ValueError(f"Invalid API type: {API_TYPE}")

# Add the following functions to your existing utils.py file
OCRBench_score = {
    "Regular Text Recognition": 0,
    "Irregular Text Recognition": 0,
    "Artistic Text Recognition": 0,
    "Handwriting Recognition": 0,
    "Digit String Recognition": 0,
    "Non-Semantic Text Recognition": 0,
    "Scene Text-centric VQA": 0,
    "Doc-oriented VQA": 0,
    "Key Information Extraction": 0,
    "Handwritten Mathematical Expression Recognition": 0,
}


def ocrbench_doc_to_visual(doc):
    # Assuming the 'doc' dictionary has a key 'image' with image data
    return [doc["image"].convert("RGB")]


def ocrbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    # Assuming the 'doc' dictionary has a key 'question' with the question text
    question = doc["question"].strip()
    if "post_prompt" in lmms_eval_specific_kwargs:
        question += lmms_eval_specific_kwargs["post_prompt"]
    return question


def ocrbench_process_results(doc, results):
    pred = results[0]
    gt_ans = doc["answer"]
    dataset_name = doc["dataset"]

    score = 0
    if dataset_name == "HME100k":
        if type(gt_ans) == list:
            for j in range(len(gt_ans)):
                answer = gt_ans[j].strip().replace("\n", " ").replace(" ", "")
                predict = pred.strip().replace("\n", " ").replace(" ", "")
                if answer in predict:
                    score = 1
        else:
            answer = gt_ans.strip().replace("\n", " ").replace(" ", "")
            predict = pred.strip().replace("\n", " ").replace(" ", "")
            if answer in predict:
                score = 1
    else:
        if type(gt_ans) == list:
            for j in range(len(gt_ans)):
                answer = gt_ans[j].lower().strip().replace("\n", " ")
                predict = pred.lower().strip().replace("\n", " ")
                if answer in predict:
                    score = 1
        else:
            answer = gt_ans.lower().strip().replace("\n", " ")
            predict = pred.lower().strip().replace("\n", " ")
            if answer in predict:
                score = 1
    return {
        "ocrbench_accuracy": {"question_type": doc["question_type"], "score": score, "prediction": pred, "ground_truth": gt_ans},
    }

def postprocess_judge_response(judge_response):
    judge_response = judge_response.strip()
    score = float(extract_final_boxed_content(judge_response))
    assert score in [0.0, 1.0]
    return score, judge_response

def get_gpt_eval_score(question, answer, pred):
    global client
    
    if isinstance(answer, list):
        answer_list = answer
    else:
        answer_list = [answer]
    score, judge_response = None, None
    for answer in answer_list:
        if pred.strip().lower() == answer.strip().lower():
            score = 1.0
    
    if score is None:
        score, judge_response = client.evaluate_correctness(
            question, answer, pred,
            postprocess_response=postprocess_judge_response,
            default_response=(0.0, "")
        )
    
    return score, judge_response


def ocrbench_process_results_judge(doc, results):
    pred = results[0]
    gt_ans = doc["answer"]
    if not isinstance(gt_ans, list):
        gt_ans = [gt_ans]
    dataset_name = doc["dataset"]

    score = 0
    if dataset_name == "HME100k":
        predict = pred.strip().replace("\n", " ").replace(" ", "")
        answers = [_.strip().replace("\n", " ").replace(" ", "") for _ in gt_ans]
    else:
        predict = pred.lower().strip().replace("\n", " ")
        answers = [_.lower().strip().replace("\n", " ") for _ in gt_ans]
    score, judge_response = get_gpt_eval_score(doc["question"], answers, predict)
    return {
        "ocrbench_accuracy": {"question_type": doc["question_type"], "score": score, "prediction": pred, "ground_truth": gt_ans, "judge_response": judge_response},
    }


def ocrbench_aggregate_accuracy(results, args):
    for result in results:
        OCRBench_score[result["question_type"]] += result["score"]
    recognition_score = (
        OCRBench_score["Regular Text Recognition"]
        + OCRBench_score["Irregular Text Recognition"]
        + OCRBench_score["Artistic Text Recognition"]
        + OCRBench_score["Handwriting Recognition"]
        + OCRBench_score["Digit String Recognition"]
        + OCRBench_score["Non-Semantic Text Recognition"]
    )
    Final_score = recognition_score + OCRBench_score["Scene Text-centric VQA"] + OCRBench_score["Doc-oriented VQA"] + OCRBench_score["Key Information Extraction"] + OCRBench_score["Handwritten Mathematical Expression Recognition"]
    # file_name = generate_submission_file("ocrbench_results.txt", args, subpath="results")
    # with open(file_name, "w") as f:
    #     print("######################### OCRBench #############################", file=f)
    #     print(f"Text Recognition(Total 300): {recognition_score}", file=f)
    #     print("---------------- Details of Recognition Score ------------------", file=f)
    #     print(f"Regular Text Recognition(Total 50): {OCRBench_score['Regular Text Recognition']}", file=f)
    #     print(f"Irregular Text Recognition(Total 50): {OCRBench_score['Irregular Text Recognition']}", file=f)
    #     print(f"Artistic Text Recognition(Total 50): {OCRBench_score['Artistic Text Recognition']}", file=f)
    #     print(f"Handwriting Recognition(Total 50): {OCRBench_score['Handwriting Recognition']}", file=f)
    #     print(f"Digit String Recognition(Total 50): {OCRBench_score['Digit String Recognition']}", file=f)
    #     print(f"Non-Semantic Text Recognition(Total 50): {OCRBench_score['Non-Semantic Text Recognition']}", file=f)
    #     print("----------------------------------------------------------------", file=f)
    #     print(f"Scene Text-centric VQA(Total 200): {OCRBench_score['Scene Text-centric VQA']}", file=f)
    #     print("----------------------------------------------------------------", file=f)
    #     print(f"Doc-oriented VQA(Total 200): {OCRBench_score['Doc-oriented VQA']}", file=f)
    #     print("----------------------------------------------------------------", file=f)
    #     print(f"Key Information Extraction(Total 200): {OCRBench_score['Key Information Extraction']}", file=f)
    #     print("----------------------------------------------------------------")
    #     print(f"Handwritten Mathematical Expression Recognition(Total 100): {OCRBench_score['Handwritten Mathematical Expression Recognition']}", file=f)
    #     print("--------------------- Final Score ------------------------------", file=f)
    #     print(f"Final Score(Total 1000): {Final_score}", file=f)
    # logger.info(f"OCR Bench results saved to {file_name}")
    # return {"Final Score":Final_score,"Text Recognition":recognition_score,'Scene Text-centric VQA':OCRBench_score['Scene Text-centric VQA'],'Doc-oriented VQA':OCRBench_score['Doc-oriented VQA'],'Key Information Extraction':OCRBench_score['Key Information Extraction'],'Handwritten Mathematical Expression Recognition':OCRBench_score['Handwritten Mathematical Expression Recognition']}
    return Final_score / 1000  # return the final score as accuracy
