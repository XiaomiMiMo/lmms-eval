import json
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter


default_model_specific_kwargs = {
    "img_token": "<image>",
    "post_prompt": "Answer with the option's letter from the given choices directly.",
}


def seed_doc_to_visual(doc):
    return [image.convert("RGB") for image in doc["image"]]


def parse_choice_img(choice: str, img_token: str):
    if "jpg" in choice or "png" in choice:
        return img_token
    return choice


def seed_doc_to_text(doc, model_specific_kwargs=None):
    img_token = model_specific_kwargs.get("img_token", default_model_specific_kwargs["img_token"])
    post_prompt = model_specific_kwargs.get("post_prompt", default_model_specific_kwargs["post_prompt"])
    question = doc["question"]
    question.replace("<img>", img_token)
    question += "\n" + f"A. {parse_choice_img(doc['choice_a'], img_token)}\n"
    question += f"B. {parse_choice_img(doc['choice_b'], img_token)}\n"
    question += f"C. {parse_choice_img(doc['choice_c'], img_token)}\n"
    question += f"D. {parse_choice_img(doc['choice_d'], img_token)}"
    if doc["data_type"] == "Image Generation":
        num_img_in_question = len(doc["data_id"]) - 4
        prepend_tokens = [img_token] * num_img_in_question
        question = " ".join(prepend_tokens) + "\n" + question
    return f"{question}\n{post_prompt}"


def seed_process_result(doc, result):
    pred = result[0].strip()
    if len(pred) > 1:
        pred = pred[0]
    answer = doc["answer"]
    data_type = doc["data_type"].split(" ")
    data_type = "_".join(data_type)

    return {f"seed_{data_type}": {"pred": pred, "answer": answer, "question_id": doc["question_id"]}, f"seed_all": {"pred": pred, "answer": answer, "question_id": doc["question_id"]}}


def seed_aggregation_result(results):
    total_count = 0
    total_correct = 0
    for result in results:
        if result["pred"] == result["answer"]:
            total_correct += 1
        total_count += 1
    return total_correct / total_count if total_count != 0 else 0


def seed_aggregation_result_all(results):
    score = seed_aggregation_result(results)
    stored_results = []
    for result in results:
        stored_results.append({"question_id": result["question_id"], "prediction": result["pred"]})
    with open("./seed_submission.json", "w") as f:
        json.dump(stored_results, f, indent=4)
    print("Storing files for seed_submission ...")

    return score
