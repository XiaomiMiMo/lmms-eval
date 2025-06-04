# Copyright 2025 Xiaomi Corporation.

import os
import time
import requests
from openai import OpenAI, AzureOpenAI
from loguru import logger as eval_logger
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter


JUDGE_RULES = """You are a strict evaluator assessing answer correctness. You must output 1 for fully correct answers and 0 for any other case.
# Input
Question:
```
{question}
```
Ground Truth Answer:
```
{answer}
```
Model Prediction:
```
{pred}
```

# Evaluation Rules
- The model prediction may contain the reasoning process, you should spot the final answer from it.
- For multiple-choice questions: Score 1 if the predicted answer matches the ground truth answer, it can be directly in option letters or the content of the options.
- For open-ended questions:
  * Score 1 if the prediction matches the answer semantically, it can be in different format.
  * Score 0 for partially correct answers or answers with extra incorrect information, even if the reasoning process is correct.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision
- For questions requiring units, both value and unit must be correct

# Strict Output format
0 or 1"""

API_TYPE = os.getenv("API_TYPE", None)
MODEL_VERSION = os.getenv("MODEL_VERSION", None)
if API_TYPE == 'openai':
    from lmms_eval.tasks._task_utils.gpt_eval_utils import OpenAIClient
    API_URL = os.getenv("OPENAI_API_URL", "YOUR_API_URL")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    client = OpenAIClient(api_url=API_URL, api_key=API_KEY, model=MODEL_VERSION, task="visualsimpleqa")
else:
    raise ValueError(f"Invalid API type: {API_TYPE}")


def get_chat_response(content: str, max_tokens: int):
    global API_TYPE
    global MODEL_VERSION
    global client

    if API_TYPE == 'openai':
        messages = [
            {
                "role": "system",
                "content": "You are a helpful and precise assistant for checking the correctness of the answer.",
            },
            {"role": "user", "content": content},
        ]
        generation_kwargs = {
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        response = client.get_chat_response(
            messages, 
            default_response="",
            postprocess_response=lambda x: x.strip().strip('[]'),
            generation_kwargs=generation_kwargs
        )
        return response
    else:
        raise ValueError(f"Invalid API type: {API_TYPE}")



def visualsimpleqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]

def visualsimpleqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    prompt = doc["multimodal_question"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        prompt += lmms_eval_specific_kwargs["post_prompt"]
    return prompt

def visualsimpleqa_process_results(doc, results):
    pred = results[0]
    llm_judge_prompt = JUDGE_RULES.format(question=doc["multimodal_question"], answer=doc["answer"], pred=pred)
    llm_judge_score = get_chat_response(llm_judge_prompt, max_tokens=20)
    score = 1 if "1" in llm_judge_score or "[1]" in llm_judge_score else 0

    item = {
        "question": doc["multimodal_question"],
        "answer": doc["answer"],
        "prediction": pred,
        "score": score,
    }

    return {"accuracy": item}


def visualsimpleqa_aggregate_results(results):
    total_score = 0
    for result in results:
        try:
            item_score = result["score"]
            total_score += item_score
        except:
            eval_logger.warning(f"Failed to get score {result['id']}: {result['score']}")
            total_score += 0
    return total_score / len(results)


