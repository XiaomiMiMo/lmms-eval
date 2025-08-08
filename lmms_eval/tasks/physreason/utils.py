import os
import json
from functools import partial
from lmms_eval.tasks._task_utils.eval_utils import AfterThinkFilter, extract_final_boxed_content
from lmms_eval.tasks._task_utils.gpt_eval_utils import OpenAIClient

API_TYPE = os.getenv("API_TYPE", None)
MODEL_VERSION = os.getenv("MODEL_VERSION", None)
if API_TYPE == 'openai':
    API_URL = os.getenv("OPENAI_API_URL", "YOUR_API_URL")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    client = OpenAIClient(api_url=API_URL, api_key=API_KEY, model=MODEL_VERSION, task="physreason")
else:
    raise ValueError(f"Invalid API type: {API_TYPE}")


# {'captions': '',
#  'image_list': [],
#  'context': "To rescue a patient, an ambulance urgently departs, sounding its siren, and begins to move with uniform acceleration along a horizontal straight road from rest at $t=0$, with an acceleration magnitude of $a=2\\mathsf{m}/\\mathsf{s}^{2}$. At $t_{1}=10s$, it ceases acceleration and begins to move with uniform velocity. At some later time, the ambulance stops sounding its siren. The last sound of the siren is heard by a person at the ambulance's starting point at $t_{2}=41s$. The speed of sound is given as $v_{0}=340m/s$.",
#  'questions': ["What is the magnitude of the ambulance's velocity during its uniform motion?",
#   'What is the distance of the ambulance from its starting point at the moment it stops sounding its siren?'],
#  'answers': ['20 m/s', '680 m'],
#  'difficulty': 'easy',
#  'problem_id': 'cal_problem_00002',
#  'info': ''}

def physreason_doc_to_visual(doc):
    return [img.convert("RGB") for img in doc["image_list"]]

def physreason_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    prompt = doc["context"]
    for i, question in enumerate(doc["questions"]):
        prompt += f"\nQuestion {i+1}: {question}"
    if lmms_eval_specific_kwargs and lmms_eval_specific_kwargs.get("post_prompt"):
        prompt += lmms_eval_specific_kwargs["post_prompt"]
    return prompt

def physreason_doc_to_target(doc):
    return json.dumps(doc["answers"])

EXTRACT_SYSTEM_PROMPT = """You are a professional answer extraction assistant. Please only return the extracted answer without adding any additional explanations."""

EXTRACT_PROMPT = """Please extract the answer for the specific question from the following output text.
## Specific question
{question}

## Output text
{solution}

## Output Format
Please return the answer directly without any explanation or additional text."""

SCORE_SYSTEM_PROMPT = "You are a professional mathematical problem answer evaluation assistant."

SCORE_PROMPT = """Based on the following information, please determine whether the two answers are semantically equivalent:

## Question
{question}

## Ground truth
{gt}

## Prediction
{pred}

## Output Format
Please only answer "true" or "false" to indicate whether these two answers express the same meaning. When evaluating, please consider whether mathematical expressions, units, and other details are equivalent."""


def physreason_process_results(doc, results):
    pred = results[0]
    steps = json.loads(doc['info'])['explanation_steps']
    num_questions = len(steps)
    num_steps = [len(v) for k, v in steps.items()]
    extracted_answers = []
    result = {"num_questions": num_questions, "num_steps": num_steps, "num_correct": 0}
    
    for i in range(num_questions):
        question = doc["context"] + "\nQuestion " + str(i+1) + ": " + doc["questions"][i]
        prompt = EXTRACT_PROMPT.format(question=question, solution=pred)
        messages = [{"role": "system", "content": EXTRACT_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        generation_kwargs = {
            "max_tokens": 2048,
            "temperature": 0,
            "top_p": 1,
        }
        answer = client.get_chat_response(
            messages, 
            default_response="", 
            postprocess_response=lambda x: x.strip(),
            generation_kwargs=generation_kwargs
        )
        extracted_answers.append(answer)

        gt = doc["answers"][i]
        if answer.strip().lower() == gt.strip().lower():
            result["num_correct"] += 1
        else:
            messages = [
                {"role": "system", "content": SCORE_SYSTEM_PROMPT},
                {"role": "user", "content": SCORE_PROMPT.format(question=question, gt=gt, pred=answer)}
            ]
            generation_kwargs = {
                "max_tokens": 2048,
                "temperature": 0,
                "top_p": 1,
            }
            response = client.get_chat_response(
                messages,
                check_response=lambda x: extract_final_boxed_content(x).strip().lower() in ["true", "false"],
                postprocess_response=lambda x: extract_final_boxed_content(x).strip().lower() == "true",
                default_response=False,
            )
            if response:
                result["num_correct"] += 1
            else:
                break
    
    result["extracted_answers"] = extracted_answers
    result["score"] = sum(result["num_steps"][:result["num_correct"]]) / sum(result["num_steps"])
    return {"physreason_score": result}


def physreason_aggregate_results(results):
    scores = [_["score"] for _ in results]
    return sum(scores) / len(scores) if len(scores) > 0 else 0