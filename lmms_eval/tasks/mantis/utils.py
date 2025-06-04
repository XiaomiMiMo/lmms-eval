# Copyright 2025 Xiaomi Corporation.

import os, json
import re 
from typing import List
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter
MULTIPLE_CHOICE_TEMPLATE = "{question}\nAnswer with the option's letter from the given choices directly."
SHORT_ANSWER_TEMPLATE = "{question}\nAnswer with the answer directly."
TEMPLATE_MAPPING = {
    "multi-choice": MULTIPLE_CHOICE_TEMPLATE,
    "short-answer": SHORT_ANSWER_TEMPLATE
}
# DatasetDict({
#     test: Dataset({
#         features: ['id', 'question_type', 'question', 'images', 'options', 'answer', 'data_source', 'category'],
#         num_rows: 217
#     })
# })

# Code adapted from official eval script: https://github.com/TIGER-AI-Lab/Mantis/blob/fa5a57bd07a32558ff95356c7f76c0a926b4bb4d/mantis/benchmark/eval.py 

def parse_answer(raw_answer):
    if "final answer:" in raw_answer.lower():
        answer = raw_answer[raw_answer.lower().index("final answer:") + len("final answer:"):].strip()
    elif "the answer is" in raw_answer.lower():
        answer = raw_answer[raw_answer.lower().index("the answer is") + len("the answer is"):].strip()
    elif "answer:" in raw_answer.lower():
        answer = raw_answer[raw_answer.lower().index("answer:") + len("answer:"):].strip()
    else:
        answer = raw_answer
    return answer

def get_option(final_answer):
    if re.match(r'Answer: [A-Z]', final_answer):
        return final_answer[8]
    for s in final_answer:
        if s.isalpha():
            return s.upper()
    return None


def get_prediction(question_type: str, raw_answer: str, ref_answer: str):
    answer = parse_answer(raw_answer)
    correct = False 
    if question_type == 'multi-choice':
        assert len(ref_answer) == 1, f"Ref answer is not a single character: {ref_answer}"
        selected_option = get_option(answer)
        correct = selected_option == ref_answer.upper()
        parsed_answer = selected_option
    elif question_type == 'short-answer':
        correct = ref_answer.lower() == answer.lower()
        parsed_answer = answer
    
    return {
        "accuracy": 1 if correct else 0,
        "raw_answer": raw_answer,
        "parsed_answer": parsed_answer
    }

def mantis_doc_to_visual(doc):
    return [img.convert("RGB") for img in doc["images"]]


def get_question(doc):
    qtype = doc["question_type"]
    question = doc["question"]
    if qtype == "multi-choice":
        option_idx = 'A'
        for option in doc['options']:
            if not any([x in option.upper() for x in [f"{option_idx})", f"{option_idx}:", f"{option_idx}."]]):
                question += f'\n ({option_idx}) {option}'
            else:
                question += f'\n {option}'
            option_idx = chr(ord(option_idx) + 1)
    return question

def mantis_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = get_question(doc)
    qtype = doc["question_type"]
    template = TEMPLATE_MAPPING[qtype]
    return template.format(question=question)

def mantis_doc_to_text_boxed(doc, lmms_eval_specific_kwargs=None):
    question = get_question(doc)
    return question + "\n" + "Put your final answer in \\boxed{}."

def mantis_process_results(doc, results):
    prediction = results[0]
    ret_dict = get_prediction(doc["question_type"], prediction, doc["answer"])
    return ret_dict 
    

def mantis_aggregate_results(results):
    correct = sum(results)
    total = len(results)
    return correct / total 


