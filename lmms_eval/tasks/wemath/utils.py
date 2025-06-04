# Copyright 2025 Xiaomi Corporation.

import re
from lmms_eval.tasks.wemath.eval_utils import evaluate_models, evaluate_models_boxed
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter


def wemath_doc_to_visual(doc):
    image = doc["image_path"]
    return [image]


def wemath_doc_to_text(doc):
    question = doc["question"]
    options = doc["option"]

    template = """Now, we require you to solve a multiple-choice math question. Please briefly describe your thought process and provide the final answer(option).
Question: {question}
Option: {options}
Regarding the format, please answer following the template below, and be sure to include two <> symbols:
<Thought process>: <<your thought process>> <Answer>: <<your option>>"""

    final_question = template.format(question=question, options=options)
    return final_question


def wemath_boxed_doc_to_text(doc):
    question = doc["question"]
    options = doc["option"]

    template = """Now, we require you to solve a multiple-choice math question. Please briefly describe your thought process and provide the final answer(option).
Question: {question}
Option: {options}
Answer the question with option letter from given choices. Put your final answer within \\boxed{{}}."""

    final_question = template.format(question=question, options=options)
    return final_question


def wemath_process_results(doc, results):
    """
    {
        "ID": "3steps_165",
        "split": "testmini",
        "knowledge concept": "Area of Circles",
        "question": "As shown in the figure, there is a circular flower bed. Mary walked from the northernmost point of the flower bed along the edge to the easternmost point, taking a total of 80 steps. It is known that Mary's average step length is 0.628 cm, what is the area of the flower bed (  ) m²?(π = 3.14)",
        "option": "A. 200.96;B. 3215.36;C. 6280;D. 32; E. No correct answer",
        "answer": "B",
        "image_path": "3steps/image/165-3.png",
        "key": "3steps_3",
        "question number": 1575,
        "knowledge concept description": "Area of ...",
        "response": "<Thought process>: ... <Answer>: ..."
    }
    """
    res = {
        "ID": doc["ID"],
        "split": doc["split"],
        "knowledge concept": doc["knowledge concept"],
        "question": doc["question"],
        "option": doc["option"],
        "answer": doc["answer"],
        "key": doc["key"],
        "question number": doc["question number"],
        "knowledge concept description": doc["knowledge concept description"],
        "response": results[0].strip()
    }
    return {"wemath_standard_eval": res}


def wemath_aggregate_results(results, args):
    res = evaluate_models(results)
    return float(res["Score (Strict)"].iloc[0].strip('%')) / 100


def wemath_boxed_aggregate_results(results, args):
    res = evaluate_models_boxed(results)
    return float(res["Score (Strict)"].iloc[0].strip('%')) / 100
