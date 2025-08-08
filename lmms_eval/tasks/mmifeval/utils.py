import asyncio
import os
import time
import re

import aiohttp
import requests
from loguru import logger as eval_logger
from openai import AzureOpenAI, OpenAI
from tqdm.asyncio import tqdm
from PIL import Image
import io, base64
import json

from lmms_eval.tasks._task_utils.gpt_eval_utils import OpenAIClient
from lmms_eval.tasks._task_utils.math_verify_utils import MathVerifyFn
from lmms_eval.tasks._task_utils.eval_utils import extract_final_boxed_content
from lmms_eval.tasks.mmifeval.function_and_compare import *

API_TYPE = os.getenv("API_TYPE", None)
MODEL_VERSION = os.getenv("MODEL_VERSION", None)
if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "YOUR_API_URL")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    client = OpenAIClient(api_url=API_URL, api_key=API_KEY, model=MODEL_VERSION, task="mmifeval")
else:
    raise ValueError(f"Invalid API type: {API_TYPE}")

def mmifeval_doc_to_visual(doc):
    image = Image.open(io.BytesIO(base64.b64decode(doc["image"])))
    return [image]


def mmifeval_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    prompt = doc["question"]
    if lmms_eval_specific_kwargs and "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != '':
        prompt = f"{prompt}\n{lmms_eval_specific_kwargs['post_prompt']}"
    return prompt


def extract_score_from_direct_gpt_resp(raw_score):
    """从gpt返回的评分文本中提取每个约束的得分，返回dict。"""
    score_pattern = re.compile(r"Score\s+of\s+([a-zA-Z0-9_\-]+):\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
    cleaned_score = re.sub(r"\s+", " ", raw_score).strip()
    cleaned_score = re.sub(r"\*", "", cleaned_score)
    score_matches = score_pattern.findall(cleaned_score)
    if not score_matches:
        raise ValueError(f"raw_score format is incorrect, cannot parse scores: {raw_score}")
    score_dict = {}
    for match in score_matches:
        component_name = match[0].strip().lower().replace(" ", "_")
        numerator = int(match[1])
        denominator = int(match[2])
        score = numerator / denominator
        score_dict[component_name] = score
    return score_dict

def extract_score_from_cmp_gpt_resp(response_text):
    summary_idx = response_text.lower().rfind("summary")
    if summary_idx == -1:
        raise ValueError("No 'summary' found in response.")
    after_summary = response_text[summary_idx + len("summary"):]
    match = re.search(r"\b(true|false)\b", after_summary, re.IGNORECASE)
    if match:
        value = match.group(1).lower()
        return 1 if value == "true" else 0
    raise ValueError("No valid 'True' or 'False' found after 'summary'.")

def generate_eval_pt(constraints, prediction):
    constraints_str = "\n".join(
        [f"Constraint_{i+1}: {constraint['value']}" for i, constraint in enumerate(constraints)]
    )
    pt = f"""\
Your task is to evaluate whether the response from an AI assistant adheres to all of the given constraints. \
Please follow the requirements below to make the judgment:
1. Be strict and consistent in your assessment.
2. You should refer to the content of image to make the judgment.
3. For each constraint, if the response fails to fully meet the constraint, give it a score of 0. Otherwise, give it a score of 1.

<start of response>
{prediction}
<end of response>

<start of constraint list>
{constraints_str}
<end of constraint list>

You must evaluate and provide an explanation for each constraint listed, ensuring no constraint is omitted. \
At the end, summarize the scores for all constraints in one sentence.

Your output should strictly follow the format below:
Judgement: ...
Summary: Score of constraint_1: x/1, Score of constraint_2: x/1, Score of constraint_3: x/1, ..., Score of constraint_n: x/1.
"""
    return pt

def generate_cmp_pt(constraint, pred_with_constraint, pred_without_constraint):
    pt = f"""\
You are an expert in judging whether the respone follow the given constraint. Your task is to assess whether the model's response satisfies the given constraint and return True or False. I will provide you with the constraint and the model's response under this constraint. To assist with your evaluation, I will also provide you with the model's response to the same question without the constraint.

<start of constraint>
{constraint}
<end of constraint>

<start of response under the constraint>
{pred_with_constraint}
<end of response under the constraint>

<start of response without the constraint>
{pred_without_constraint}
<end of response without the constraint>

**Please follow the steps below to evaluate**:
Step 1. Compare the model's response under the constraint with its response without the constraint. If you believe these two answers are very similar, it means the model has not fully considered the impact of the constraint on the answer. Please return False.
Step 2. Compare the model's response under the constraint with the content of the constraint. If you believe the model's response does not meet the requirements specified in the constraint, return False. Otherwise, if the response effectively satisfies the constraint, return True.

Start by briefly explaining your reasoning based on the above steps. At the end, provide a one-sentence summary of your evaluation.

Your output must strictly follow this format:  
Reasoning: ...  
Summary: "True" / "False".
"""
    return pt

def generate_eval_pt_vision(question, prediction, ground_truth):
    pt = f"""\
You are an expert evaluator. Your task is to extract the answer from the model output and compare it with the ground truth list to determine whether the model answer covers all the points in the ground truth list. \
The ground truth list is provided as a JSON array of strings, and the model answer is a text string. \
An answer is considered correct if every element from the ground truth list appears in the model answer (substring matching is acceptable). \
The order does not matter. \

Your response should only be 'right' if the model answer fully covers the ground truth, or 'wrong' if it does not. \
Do not provide any additional commentary.

Question: {question}
Response from the model: {prediction}
Ground Truth List: {ground_truth}
"""
    return pt

def extract_score_from_vision_gpt_resp(raw_score):
    if raw_score == "right":
        return 1
    elif raw_score == "wrong":
        return 0
    else:
        if re.search(r"right", raw_score, re.IGNORECASE):
            return 1
        elif re.search(r"wrong", raw_score, re.IGNORECASE):
            return 0
        else:
            raise ValueError("raw_score format is incorrect, cannot parse scores")

def mmifeval_process_results(doc, results):
    """
    按照score.py中judge_one_item的逻辑实现评分
    支持P-Level (vision prompt) 和其他类型 (多约束评分)
    对于 cmp_gpt 类型的约束，暂时跳过，在聚合阶段处理
    """
    response = results[0]
    if "</think>" in response:
        response = response.split("</think>")[-1]
    
    # P-Level题型：使用vision prompt
    if doc.get("tag", None) == "P-Level":
        pt = generate_eval_pt_vision(doc["question"], response, doc["answer"])
        gpt_resp = client.get_chat_response(
            [{"role": "user", "content": pt}],
            default_response="wrong",
            generation_kwargs={"temperature": 0.0, "max_tokens": 4096}
        )
        try:
            score = extract_score_from_vision_gpt_resp(gpt_resp)
            return {
                "mmifeval_standard_eval": {
                    "id": doc.get("id", ""),
                    "question": doc.get("question", ""),
                    "answer": doc.get("answer", ""),
                    "response": response,
                    "score": {
                        "total_score": score,
                        "gpt_resp": gpt_resp,
                    },
                    "tag": doc.get("tag", ""),
                    "constraints": doc.get("constraints", []),
                },
                "mmifeval_c_level_eval": {
                    "id": doc.get("id", ""),
                    "question": doc.get("question", ""),
                    "answer": doc.get("answer", ""),
                    "response": response,
                    "score": {
                        "total_score": score,
                        "gpt_resp": gpt_resp,
                    },
                    "tag": doc.get("tag", ""),
                    "constraints": doc.get("constraints", []),
                },
                "mmifeval_p_level_eval": {
                    "id": doc.get("id", ""),
                    "question": doc.get("question", ""),
                    "answer": doc.get("answer", ""),
                    "response": response,
                    "score": {
                        "total_score": score,
                        "gpt_resp": gpt_resp,
                    },
                    "tag": doc.get("tag", ""),
                    "constraints": doc.get("constraints", []),
                },
            }
        except Exception as e:
            eval_logger.error(f"\nError in P-Level evaluation:\n{e}\nDoc:\n{doc}\nGPT Response:\n{gpt_resp}")
            return {
                "mmifeval_standard_eval": {
                    "id": doc.get("id", ""),
                    "question": doc.get("question", ""),
                    "answer": doc.get("answer", ""),
                    "response": response,
                    "score": {
                        "total_score": 0,
                        "gpt_resp": str(e),
                    },
                    "tag": doc.get("tag", ""),
                    "constraints": doc.get("constraints", []),
                    "del_cons": doc.get("del_cons", ""),
                },
                "mmifeval_p_level_eval": {
                    "id": doc.get("id", ""),
                    "question": doc.get("question", ""),
                    "answer": doc.get("answer", ""),
                    "response": response,
                    "score": {
                        "total_score": 0,
                        "gpt_resp": str(e),
                    },
                    "tag": doc.get("tag", ""),
                    "constraints": doc.get("constraints", []),
                    "del_cons": doc.get("del_cons", ""),
                },
                "mmifeval_c_level_eval": {
                    "id": doc.get("id", ""),
                    "question": doc.get("question", ""),
                    "answer": doc.get("answer", ""),
                    "response": response,
                    "score": {
                        "total_score": 0,
                        "gpt_resp": str(e),
                    },
                    "tag": doc.get("tag", ""),
                    "constraints": doc.get("constraints", []),
                    "del_cons": doc.get("del_cons", ""),
                },
            }
    
    # 其他题型：多约束评分
    constraints = json.loads(doc["constraints"])
    
    # 分类约束：direct_gpt 和其他
    constraint_direct_gpt = []
    constraint_other = []
    constraint_cmp_gpt = []  # 单独记录 cmp_gpt 约束
    
    for constraint in constraints:
        method = constraint["judge"]["method"]
        if method == "direct_gpt":
            constraint_direct_gpt.append(constraint)
        elif method == "cmp_gpt":
            constraint_cmp_gpt.append(constraint)
        else:
            constraint_other.append(constraint)
    
    score_dict = {}
    
    # 1. 批量处理 direct_gpt 约束
    if len(constraint_direct_gpt) > 0:
        pt_direct_gpt = generate_eval_pt(constraint_direct_gpt, response)
        gpt_resp = client.get_chat_response(
            [{"role": "user", "content": pt_direct_gpt}],
            default_response="Score of constraint_1: 0/1.",
            generation_kwargs={"temperature": 0.0, "max_tokens": 4096}
        )
        try:
            direct_gpt_score_dict = extract_score_from_direct_gpt_resp(gpt_resp)
            score_dict["gpt_resp_direct_gpt"] = gpt_resp
            # 将解析出的分数映射到对应的约束key
            for i, constraint in enumerate(constraint_direct_gpt):
                constraint_key = f"constraint_{i+1}"
                if constraint_key in direct_gpt_score_dict:
                    score_dict[constraint["key"]] = direct_gpt_score_dict[constraint_key]
                else:
                    score_dict[constraint["key"]] = 0.0
        except Exception as e:
            eval_logger.error(f"\nError in direct_gpt evaluation:\n{e}\nDoc:\n{doc}\nGPT Response:\n{gpt_resp}")
            score_dict["gpt_resp_direct_gpt"] = str(e)
            # 对所有direct_gpt约束都给0分
            for constraint in constraint_direct_gpt:
                score_dict[constraint["key"]] = 0.0
    
    # 2. 处理 rule_based 约束
    for constraint in constraint_other:
        if constraint["judge"]["method"] == "rule_based":
            score = 1.0
            for func_dict in constraint["judge"].get("verify_funcs", []):
                func = globals().get(func_dict["func"])
                if func is None:
                    score = 0.0
                    break
                judge_result = func(response, *func_dict.get("params", []))
                if not judge_result:  # False -> score = 0
                    score = 0.0
                    break
            score_dict[constraint["key"]] = score
    
    # 3. 对于 cmp_gpt 约束，暂时标记为待处理
    for constraint in constraint_cmp_gpt:
        score_dict[f"_cmp_gpt_pending_{constraint['key']}"] = {
            "constraint": constraint,
            "response": response,
            "doc_id": doc.get("id", ""),
        }
    
    # 4. 计算 total_score (跳过以 gpt_resp_ 和 _cmp_gpt_pending_ 开头的键)
    total_score = 0.0
    cnt = 0
    for key, value in score_dict.items():
        if key.startswith("gpt_resp_") or key.startswith("_cmp_gpt_pending_"):
            continue
        total_score += value
        cnt += 1
    
    # 如果有 cmp_gpt 约束，total_score 暂时设为 None，在聚合阶段重新计算
    if len(constraint_cmp_gpt) > 0:
        score_dict["total_score"] = None
        score_dict["_has_cmp_gpt"] = True
    else:
        score_dict["total_score"] = total_score / cnt if cnt > 0 else 0.0
        score_dict["_has_cmp_gpt"] = False
    
    return {
        "mmifeval_standard_eval": {
            "id": doc.get("id", ""),
            "question": doc.get("question", ""),
            "answer": doc.get("answer", ""),
            "response": response,
            "score": score_dict,
            "tag": doc.get("tag", ""),
            "constraints": doc.get("constraints", []),
            "infer_type": doc.get("infer_type", "main"),
            "del_cons": doc.get("del_cons", ""),
        },
        "mmifeval_c_level_eval": {
            "id": doc.get("id", ""),
            "question": doc.get("question", ""),
            "answer": doc.get("answer", ""),
            "response": response,
            "score": score_dict,
            "tag": doc.get("tag", ""),
            "constraints": doc.get("constraints", []),
            "infer_type": doc.get("infer_type", "main"),
            "del_cons": doc.get("del_cons", ""),
        },
        "mmifeval_p_level_eval": {
            "id": doc.get("id", ""),
            "question": doc.get("question", ""),
            "answer": doc.get("answer", ""),
            "response": response,
            "score": score_dict,
            "tag": doc.get("tag", ""),
            "constraints": doc.get("constraints", []),
            "infer_type": doc.get("infer_type", "main"),
            "del_cons": doc.get("del_cons", ""),
        },
    }

def _process_cmp_gpt_in_aggregate(results):
    """
    在聚合阶段处理 cmp_gpt 约束的评分
    Args:
        results: 所有结果的列表
    Returns:
        处理后的 results 列表
    """
    # 1. 构建 aux_data_dict
    aux_data_dict = {}
    for result in results:
        if result.get("infer_type", "main") == "aux_cmp_gpt":
            del_cons = result.get("del_cons", "")
            item_id = result.get("id", "")
            if item_id not in aux_data_dict:
                aux_data_dict[item_id] = {}
            aux_data_dict[item_id][del_cons] = result.get("response", "")
    
    eval_logger.info(f"Built aux_data_dict with {len(aux_data_dict)} items for cmp_gpt evaluation")
    
    # 2. 处理所有包含 cmp_gpt 的结果
    processed_results = []
    for result in results:
        if not result.get("_has_cmp_gpt", False):
            processed_results.append(result)
            continue
        
        score_dict = result["score"].copy()
        
        # 处理所有 cmp_gpt 待处理项
        cmp_gpt_keys = [key for key in score_dict.keys() if key.startswith("_cmp_gpt_pending_")]
        
        for pending_key in cmp_gpt_keys:
            pending_data = score_dict[pending_key]
            constraint = pending_data["constraint"]
            response = pending_data["response"]
            doc_id = pending_data["doc_id"]
            
            constraint_key = constraint["key"]
            
            # 获取无约束预测结果
            if doc_id in aux_data_dict and constraint_key in aux_data_dict[doc_id]:
                del_cons_prediction = aux_data_dict[doc_id][constraint_key]
                
                pt = generate_cmp_pt(constraint["value"], response, del_cons_prediction)
                gpt_resp = client.get_chat_response(
                    [{"role": "user", "content": pt}],
                    default_response="Summary: False.",
                    generation_kwargs={"temperature": 0.0, "max_tokens": 4096}
                )
                try:
                    score = extract_score_from_cmp_gpt_resp(gpt_resp)
                    score_dict[constraint_key] = score
                    score_dict[f"gpt_resp_cmp_gpt_{constraint_key}"] = gpt_resp
                except Exception as e:
                    eval_logger.error(f"\nError in cmp_gpt evaluation:\n{e}\nDoc ID:\n{doc_id}\nGPT Response:\n{gpt_resp}")
                    score_dict[constraint_key] = 0.0
                    score_dict[f"gpt_resp_cmp_gpt_{constraint_key}"] = str(e)
            else:
                eval_logger.warning(f"Missing del_cons_prediction for id={doc_id}, key={constraint_key}")
                score_dict[constraint_key] = 0.0
                score_dict[f"gpt_resp_cmp_gpt_{constraint_key}"] = "Missing del_cons_prediction"
            
            # 移除待处理标记
            del score_dict[pending_key]
        
        # 重新计算 total_score
        total_score = 0.0
        cnt = 0
        for key, value in score_dict.items():
            if key.startswith("gpt_resp_") or key.startswith("_") or key == "total_score":
                continue
            total_score += value
            cnt += 1
        
        score_dict["total_score"] = total_score / cnt if cnt > 0 else 0.0
        score_dict["_has_cmp_gpt"] = False  # 标记已处理
        
        result["score"] = score_dict
        processed_results.append(result)
    
    return processed_results

def mmifeval_aggregate_results(results, args):
    # 首先处理 cmp_gpt 约束
    results = _process_cmp_gpt_in_aggregate(results)
    
    scores = []
    for result in results:
        if result.get("infer_type", "main") == "main":  # 只统计 main 数据
            score = result["score"]["total_score"] if isinstance(result["score"], dict) else result["score"]
            if score is not None:  # 跳过 None 值
                scores.append(score)
    score = sum(scores) / len(scores) if len(scores) > 0 else 0
    return score

def mmifeval_aggregate_c_level_results(results, args):
    # 首先处理 cmp_gpt 约束
    results = _process_cmp_gpt_in_aggregate(results)
    
    scores = []
    for result in results:
        # 只统计C-Level 且是 main 数据
        if result.get("tag", None) == "C-Level" and result.get("infer_type", "main") == "main":
            score = result["score"]["total_score"] if isinstance(result["score"], dict) else result["score"]
            if score is not None:  # 跳过 None 值
                scores.append(score)
    acc = sum(scores) / len(scores) if len(scores) > 0 else 0
    return acc

def mmifeval_aggregate_p_level_results(results, args):
    # 首先处理 cmp_gpt 约束（如果有的话）
    results = _process_cmp_gpt_in_aggregate(results)
    
    scores = []
    for result in results:
        # 只统计P-Level 且是 main 数据
        if result.get("tag", None) == "P-Level" and result.get("infer_type", "main") == "main":
            score = result["score"]["total_score"] if isinstance(result["score"], dict) else result["score"]
            if score is not None:  # 跳过 None 值
                scores.append(score)
    acc = sum(scores) / len(scores) if len(scores) > 0 else 0
    return acc

