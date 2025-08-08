import json
from lmms_eval.tasks._task_utils.eval_utils import AfterThinkFilter
from lmms_eval.tasks.mmlongbench.eval_utils import calculate_metrics, parse_output


def mmlongbench_doc_to_visual(doc):
    return [img.convert("RGB") for img in doc["image_list"]]

def mmlongbench_doc_to_text(doc):
    return doc["prompt"]


class DefaultAnswerGenerator():
    def __init__(self):
        self.idx = 0
        self.default_answer_list = ["0", "1", "1", "0"]
    def get_default_answer(self):
        default_answer  = self.default_answer_list[self.idx]
        self.idx = (self.idx + 1) % len(self.default_answer_list)
        return default_answer

default_answer_generator = DefaultAnswerGenerator()

metric_name_map = {
    "sub_em": "sub_em",
    "binary_acc": "binary_acc",
    "soft_acc": "soft_acc",
    "mc_acc": "mc_acc",
    "cls_acc": "cls_acc",
    "rouge": "rougeL_f1",
    "doc_qa": "doc_qa"
}

def mmlongbench_process_results(doc, results):
    dataset = doc["dataset"]
    pred = results[0]
    answer = json.loads(doc["answer"])

    default_answer = None
    answer_format = None
    if ("infoseek" in dataset or "viquae" in dataset) or "triviaqa" in dataset or "occ_vrag" in dataset:
        system_template = "Answer:"
        metrics = "sub_em"
    elif "vh_single" in dataset or "vh_multi" in dataset:
        default_answer = default_answer_generator.get_default_answer()
        system_template = "Answer:"
        metrics = "binary_acc"
    elif "mm_niah" in dataset:
        system_template = "Answer:"
        if "text" in dataset:
            if "retrieval" in dataset or "reasoning" in dataset:
                metrics = "sub_em"
            else:
                metrics = "soft_acc"
        else:
            if "retrieval" in dataset or "reasoning" in dataset:
                metrics = "mc_acc"
            else:
                metrics = "soft_acc"
    elif dataset == "text-haystack_retrieval-image": # our ablation of removing other text
        system_template = "Answer:"
        metrics = "mc_acc"
    elif any(key in dataset for key in ["cars196", "food101", "inat2021", "sun397"]):
        system_template = "label:"
        metrics = "cls_acc"
    elif "gov-report" in dataset or "lexsum" in dataset:
        system_template = "Summary:"
        metrics = "rouge"
    elif any(key in dataset for key in ["longdocurl", "mmlongdoc", "slidevqa"]) or dataset == "text_doc":
        answer_format = json.loads(doc["raw_item"])["answer_format"]
        system_template = "Answer:"
        metrics = "doc_qa"
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    
    if answer_format is not None:
        answer = [answer, answer_format]
    
    parsed_pred = parse_output(pred, prefix=system_template)
    if metrics == "doc_qa":
        if parsed_pred is None:
            preds = [pred]
        else:
            preds = [parsed_pred]
    elif default_answer is not None:
        preds = [pred, parsed_pred, default_answer]
    else:
        preds = [pred, parsed_pred]
    
    mets = {}
    for pred in preds:
        new_mets = calculate_metrics(pred, answer, metrics)
        for k, v in new_mets.items():
            if k not in mets:
                mets[k] = v
            else:
                mets[k] = max(mets[k], v)
    
    ret = {
        "pred": pred,
        "parsed_pred": parsed_pred,
        "default_answer": default_answer,
        "answer": answer,
        "answer_format": answer_format,
        "score": mets[metric_name_map[metrics]]
    }

    return {"mmlongbench_score": ret}


def mmlongbench_aggregate_results(results):
    scores = []
    for result in results:
        scores.append(result["score"])
    score = sum(scores) / len(scores) * 100 if len(scores) > 0 else 0
    return score
