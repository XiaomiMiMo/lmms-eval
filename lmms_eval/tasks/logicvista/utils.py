# Copyright 2025 Xiaomi Corporation.

import asyncio
import os
import time

import aiohttp
import requests
from loguru import logger as eval_logger
from openai import AzureOpenAI, OpenAI
from tqdm.asyncio import tqdm

from lmms_eval.tasks._task_utils.gpt_eval_utils import OpenAIClient
from lmms_eval.tasks._task_utils.math_verify_utils import MathVerifyFn
from lmms_eval.tasks._task_utils.eval_utils import extract_final_boxed_content

API_TYPE = os.getenv("API_TYPE", None)
MODEL_VERSION = os.getenv("MODEL_VERSION", None)
if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "YOUR_API_URL")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    client = OpenAIClient(api_url=API_URL, api_key=API_KEY, model=MODEL_VERSION, task="logicvista")
else:
    raise ValueError(f"Invalid API type: {API_TYPE}")

JUDGE_RULES = """You are a information extractor that extracts multiple choice letter answer choices from a paragraph that contains the answer choice and sometimes explaination of why that choice is correct to the given question.
What letter did the following answer choose? If the answer did not select a letter answer choice, first try to infer the answer based off the given choices.
If it does not seem like the given answer corresponds to an answer choice OR if there is no selected answer, please just respond with Z.
Make sure you answer with ONLY the letters chosen.
Example 1: 
Question: <start>
What is the main object in image?
Options: A. teddy bear B. rabbit C. cat D. dog
<end>
Answer: <start>
a cute teddy bear
<end>
Your output: A
Example 2: 
Question: <start>
What is the main object in image?
Options: A. teddy bear B. rabbit C. cat D. dog
<end>
Answer: <start>
Spider
<end>
Your output: Z
Example 3: 
Question: <start>
Which figure is a rotation of the object?
<end>
Answer: <start>
The figure on the right, labeled "D," is a rotation of the object shown in the top left corner.
<end>
Your output: D
Example 4: 
Question: <start>
Which of the boxes comes next in the sequence? Select from A-E
<end>
Answer: <start>
The sequence of the boxes is A, B, C, D, E.
<end>
Your output: ABCDE
Example 5: 
Question: <start>
{question}
<end>
Answer: <start>
{answer}
<end>
Your output: """



def logicvista_doc_to_visual(doc):
    return [doc["image"]]


def logicvista_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    prompt = doc["question"]
    if lmms_eval_specific_kwargs and "post_prompt" in lmms_eval_specific_kwargs:
        prompt = f"{prompt}\n{lmms_eval_specific_kwargs['post_prompt']}"
    return prompt


def logicvista_process_results(doc, results):
    def postprocess_response(response):
        response = response.strip().lower()
        assert all(c in "12345abcdefghijz" for c in response), response
        return response
    
    math_verify_fn = MathVerifyFn()
    math_verify_score, math_verify_ext = math_verify_fn(results[0].strip(), doc["answer"])
    gt_answer = "".join(sorted(doc["answer"].lower().split(", ")))

    question = doc["question"]
    response = extract_final_boxed_content(results[0])
    if response.lower().strip() == gt_answer:
        gpt_eval_res = response.strip()
        score = 1
    else:
        gpt_eval_res = client.get_chat_response(
            [{"role": "user", "content": JUDGE_RULES.format(question=question, answer=response)}], 
            postprocess_response=postprocess_response,
            default_response="Z",
            generation_kwargs={"temperature": 0.8, "max_tokens": 64}
        )
        score = 1 if gpt_eval_res == gt_answer else 0

    return {
        "logicvista_standard_eval": {
            "question": question,
            "answer": gt_answer,
            "response": response,
            "extracted_answer": gpt_eval_res,
            "score": score,
        },
        "math_verify": {
            "score": math_verify_score,
            "extraction": math_verify_ext
        }
    }



def logicvista_aggregate_results(results, args):
    scores = []
    for result in results:
        scores.append(result["score"])
    score = sum(scores) / len(scores) if len(scores) > 0 else 0
    return score

