# Copyright 2025 Xiaomi Corporation.

import os, json
from copy import deepcopy
from collections import defaultdict
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter

from lmms_eval.tasks.charxiv.constants import DESCRIPTIVE_RESP_INST, DESCRIPTIVE_GRADING_PREFIX, \
            DESCRIPTIVE_GRADING_QMAP, DESCRIPTIVE_GRADING_ICL
from lmms_eval.tasks.charxiv.constants import REASONING_RESP_INST, REASONING_GRADING_PREFIX, \
                REASONING_GRADING_INST
# sample format 
# 'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1024x496>,
#  'question': 'For the subplot at row 1 and column 2, what is the spatially highest labeled tick on the y-axis?\n    * Your final answer should be the tick value on the y-axis that is explicitly written, including the case when y-axis is shared across multiple subplots. When the y-axis is present on both the left and right of the plot, based on the axis at the left. Ignore units or scales that are written separately from the tick, such as units and scales from the axis label or the corner of the plot.',
#  'answer': '60',
#  'qtype': 'descriptive'}
# metadata

from loguru import logger as eval_logger
import time
from functools import partial
from lmms_eval.tasks._task_utils.gpt_eval_utils import OpenAIClient

API_TYPE = os.getenv("API_TYPE", None)
MODEL_VERSION = os.getenv("MODEL_VERSION", None)
if API_TYPE == 'openai':
    API_URL = os.getenv("OPENAI_API_URL", "YOUR_API_URL")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    client = OpenAIClient(api_url=API_URL, api_key=API_KEY, model=MODEL_VERSION, task="charxiv")
else:
    raise ValueError(f"Invalid API type: {API_TYPE}")



def charxiv_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def charxiv_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    prompt = doc["question"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        prompt += lmms_eval_specific_kwargs["post_prompt"]
    return prompt


def charxiv_process_results(doc, results):
    global client

    pred = results[0]
    qtype = doc["qtype"]
    return_dict = {}
    if qtype == "descriptive":
        # build grading queries
        queries = build_descriptive_grading_queries(doc, pred)[0]['grading_query']
        return_dict = get_descriptive_result(client, queries, length=1)
        score = return_dict["score_T1"]
        category = "descriptive"
    elif qtype == "reasoning":
        queries = build_reasoning_grading_queries(doc, pred)
        assert len(queries) == 1, "Only one query is supported for reasoning"
        for figure_path, query in queries.items():
            ext, scr = get_reasoning_result(client, query['grading_query'])
            score = scr 
            category =  "reasoning"
    return {"overall": score,
            "reasoning": score if category == "reasoning" else -1, # use -1 to indicate invalid reasoning example
            "descriptive": score if category == "descriptive" else -1, # use -1 to indicate invalid descriptive example
            }

def charxiv_aggregate_results(results):
    correct, total = 0, 0 
    valid_results = [ret for ret in results if ret != -1]
    correct = sum(valid_results)
    total = len(valid_results)
    return correct / total if total > 0 else 0

## original help functions from https://github.com/princeton-nlp/CharXiv/blob/main/src/descriptive_utils.py
def get_rubric(qid):
    instruction = None
    if qid in [1]:
        instruction = DESCRIPTIVE_GRADING_ICL['title']
    if qid in [2, 3, 4, 5, 6, 7]:
        instruction = DESCRIPTIVE_GRADING_ICL['ocr']
    if qid in [8, 9, 10, 12, 14, 15, 17, 19]:
        instruction = DESCRIPTIVE_GRADING_ICL['quant']
    if qid in [11]:
        instruction = DESCRIPTIVE_GRADING_ICL['bool']
    if qid in [13]:
        instruction = DESCRIPTIVE_GRADING_ICL['enum']
    if qid in [16]:
        instruction = DESCRIPTIVE_GRADING_ICL['trend']
    if qid in [18]:
        instruction = DESCRIPTIVE_GRADING_ICL['layout']
    assert instruction is not None, f"Instruction for qid {qid} is not found."
    return instruction



def get_descriptive_result(client, prompt, length):
    generation_kwargs = {
        "response_format": {"type": "json_object"},
        "max_tokens": 2048,
        "temperature": 0,
        "top_p": 1,
    }
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = client.get_chat_response(messages, default_response=None, check_response=partial(verify_descriptive_grading_output, length_data=length), generation_kwargs=generation_kwargs)
    if response is None:
        response = build_dummy_output(length)
    elif isinstance(response, str):
        response = json.loads(response)
    return response

def get_reasoning_result(client, prompt):
    generation_kwargs = {
        "response_format": {"type": "json_object"},
        "max_tokens": 2048,
        "temperature": 0,
        "top_p": 1,
    }
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = client.get_chat_response(messages, default_response=None, check_response=verify_reasoning_grading_output, generation_kwargs=generation_kwargs)
    if response is None:
        ext, scr = 'Failed to parse response', -1
    else:
        if isinstance(response, str):
            response = json.loads(response)
        if "extracted_answer" in response:
            ext, scr = response['extracted_answer'], response['score']
        else:
            ext, scr = response['extract_answer'], response['score']
    return ext, scr



def build_json_keys(length):
    keys = []
    # specify the keys for gpt-4o's json response
    for i in range(1, length+1):
        keys.append(f"extract_answer_T{i}")
        keys.append(f"score_T{i}")
    return str(keys)

def populate_grading_inputs(batch):
    query = ""
    for i, (_, response, answer) in enumerate(batch):
        # index, response, answer
        curr_query = "T{}:\nResponse {}: {}\nGround Truth {}: {}\n\n"\
            .format(i+1, i+1, response, i+1, answer)
        query += curr_query
    return query

def verify_descriptive_grading_output(data, length_data):
    data = json.loads(data.strip())

    # check the integrity of keys and values
    for i in range(1, length_data+1):
        assert f"extract_answer_T{i}" in data, f"extract_answer_T{i} is not found in {data}"
        assert f"score_T{i}" in data, f"score_T{i} is not found in {data}"
        assert data[f"score_T{i}"] in [0, 1], f"score_T{i} is not in [0, 1]"
    return True

def verify_reasoning_grading_output(data):
    data = json.loads(data.strip())
    assert "extracted_answer" in data or "extract_answer" in data, f"extracted_answer is not found in {data}"
    assert "score" in data, f"score is not found in {data}"
    return True

def build_dummy_output(length_data):
    # if failed to parse the response, return dummy data
    data = {}
    for i in range(1, length_data+1):
        data[f"extract_answer_T{i}"] = "Failed to parse response"
        data[f"score_T{i}"] = -1
    return data

def preprocess_descriptive_grading_queries(input, resp, num_templates=19):
    # group the responses based on the template id instead of figure id
    groups = {i: [] for i in range(1, num_templates + 1)}
    for _, data in input.items():
        figure_path = data['figure_path']
        qids = data['qids']
        for i, qid in enumerate(qids):
            # figure_path with question index
            resp_key = f"{figure_path}_{i}"
            response = resp[resp_key]['response']
            answer = data['answers'][i]
            groups[qid].append((resp_key, response, answer))
    return groups

def build_descriptive_grading_queries(doc, pred, nq_per_query=1):
    queries = []
        # batched evaluation based on number of questions per query (nq_per_query)
        # batch: list of tuples (resp_key, response, answer)
        # question based on the template id
    qid = doc['qid']
    question = DESCRIPTIVE_GRADING_QMAP[qid].split('\n')[0]
        # build the json keys for GPT-4o's response
    json_keys = build_json_keys(1)
        # populate batch size, question, and json keys spec
    prefix = DESCRIPTIVE_GRADING_PREFIX\
            .replace("<|NUM_TRIPLETS|>", str(1))\
            .replace("<|OVERARCHING_QUESTION|>", question)\
            .replace("<|JSON_KEYS|>", json_keys)
    # add in-context grading example based on the template id
    rubric_icl = get_rubric(qid)
    # prompt + example + model responses
    triple = (doc['metadata']['original_id'], doc['answer'], pred)
    key = f"{doc['metadata']['original_id']}_{doc['metadata']['figure_path']}_{doc['qid']}"
    grading_query = prefix + rubric_icl + populate_grading_inputs([triple])
    curr_query = {
        'resp_keys': [key],
        'grading_query': grading_query,
    }
    queries.append(curr_query)
    return queries

def postprocess_descriptive_grading_queries(queries):
    scores = {}
    for query in queries:
        # query contains resp_keys, grading_query, extract_answer and score
        resp_keys = query['resp_keys']
        for i, resp_key in enumerate(resp_keys):
            # extract the answer and score for each response key
            extracted_answer = query[f"extract_answer_T{i+1}"]
            score = query[f"score_T{i+1}"]
            # store the extracted answer and score
            scores[resp_key] = {
                'resp_id': resp_key,
                'extracted_answer': extracted_answer,
                'score': score,
            }
    return scores


def get_number_instruction(answer):
    base = answer.split('.')
    whole, decimal = base[0], None if len(base) == 1 else base[1]
    # check if it contains decimal places
    if whole is not None and decimal is None:
        inst = "* Your final answer must be an exact integer."
    elif whole is not None and decimal is not None:
        num_decimal = len(decimal)
        inst = f"* Your final answer must be a number with {num_decimal} decimal places."
    else:
        raise ValueError(f"Invalid answer: {answer}")
    return inst


def build_reasoning_grading_queries(doc, pred):
    queries = {}
    figure_path = str(doc['metadata']['figure_path'])
    # question without instruction, response
    query, response = doc['question'].split('\n')[0], pred
    # get query for answer type (inst_category), then
    # populate the query with the question, ground truth, and response
    grading_query = REASONING_GRADING_PREFIX + deepcopy(\
            REASONING_GRADING_INST[doc['metadata']['reasoning_a_type']])\
            .replace("<|question|>", query)\
            .replace("<|ground_truth|>", doc['answer'])\
            .replace("<|response|>", response)
    query = {
            'figure_path': figure_path,
            'grading_query': grading_query,
        }
    queries[figure_path] = query
    return queries
