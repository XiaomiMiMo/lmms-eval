# Copyright 2025 Xiaomi Corporation.

import re

from datasets import Dataset
from loguru import logger as eval_logger

REC_METRICS = ["IoU", "ACC@0.1", "ACC@0.3", "ACC@0.5", "ACC@0.7", "ACC@0.9", "Center_ACC"]


def screenspot_rec_doc_to_visual(doc):
    # Image is presented as is
    image = doc["image"].convert("RGB")
    return [image.convert("RGB")]


PROMPT_EN = "Bounding box coordinates are specified in the format (top-left x, top-left y, bottom-right x, bottom-right y). Please provide the bounding box coordinates of the region that corresponds to the command: {instruction}"

PROMPT_MIMO_EN = "Locate UI components that match the command: \"{instruction}\". Output a JSON in the format [{{\"bbox_2d\": [...], \"label\": \"{{the_whole_command}}\"}}, ...]."

PROMPT_CN = "请按照[左上角x, 左上角y, 右下角x, 右下角y]的格式提供与命令相对应的区域边界框坐标：{instruction}"

PROMPT_MIMO_CN = "定位如下命令所指定的UI元素：\"{instruction}\"。按照如下格式输出JSON：[{{\"bbox_2d\": [...], \"label\": \"{{the_whole_command}}\"}}, ...]。"


def screenspot_pro_en_rec_doc_to_text(doc):
    return PROMPT_EN.format(instruction=doc["instruction"])


def screenspot_pro_en_rec_doc_to_text_mimo(doc):
    return PROMPT_MIMO_EN.format(instruction=doc["instruction"])


def screenspot_pro_cn_rec_doc_to_text(doc):
    return PROMPT_CN.format(instruction=doc["instruction_cn"])


def screenspot_pro_cn_rec_doc_to_text_mimo(doc):
    return PROMPT_MIMO_CN.format(instruction=doc["instruction_cn"])


from lmms_eval.tasks._task_utils.eval_utils import parse_bbox, normalize_bbox, parse_bbox_from_point
import os


def screenspot_pro_rec_process_result(doc, result, inst_key="instruction"):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = result[0] if len(result) > 0 else ""
    pred1 = parse_bbox(pred)
    if pred1 == [0,0,0,0]:
        pred = parse_bbox_from_point(pred)
    else:
        pred = pred1
    pred = normalize_bbox(pred, doc["image_width"], doc["image_height"], resize_max_pixels=int(os.getenv("QWEN_RESIZE_MAX_PIXELS", 0)))
    bbox = normalize_bbox(doc["bbox"], doc["image_width"], doc["image_height"])
    iou = compute_iou(bbox, pred)
    center_acc = compute_center_accuracy(bbox, pred)
    ann_id = doc["img_filename"]
    data_dict = {"instruction": doc[inst_key], "pred": pred, "ann_id": ann_id, "bbox": bbox, "iou": iou, "center_acc": center_acc}
    return {f"screenspot_{metric}": data_dict for metric in REC_METRICS}


def screenspot_pro_en_rec_process_result(doc, result):
    return screenspot_pro_rec_process_result(doc, result, inst_key="instruction")


def screenspot_pro_cn_rec_process_result(doc, result):
    return screenspot_pro_rec_process_result(doc, result, inst_key="instruction_cn")


def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
    - float: IoU of box1 and box2.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Compute the area of intersection
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the area of the union
    union_area = box1_area + box2_area - intersection_area

    # Compute the Intersection over Union
    iou = intersection_area / union_area

    return iou


def compute_accuracy(iou, threshold=0.5):
    """
    Compute the accuracy of two bounding boxes based on a specified threshold.

    Parameters:
    - iou (float): IoU of the two bounding boxes.
    - threshold (float): Threshold for the IoU to consider the prediction correct.

    Returns:
    - float: Accuracy of the prediction based on the IoU threshold.
    """
    return iou >= threshold


def compute_center_accuracy(box1, box2):
    """
    Compute if the center point of box 2 is within box 1.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
    - bool: True if the center point of box 2 is within box 1, False otherwise.
    """
    # Compute the center point of box 2
    center_x = (box2[0] + box2[2]) / 2
    center_y = (box2[1] + box2[3]) / 2

    # Check if the center point is within box 1
    return box1[0] <= center_x <= box1[2] and box1[1] <= center_y <= box1[3]


def screenspot_rec_aggregation_result(results, metric):
    """
    Aggregate the results of the screenspot evaluation task using the specified metric.

    Args:
    - results (list of dict): List of result dictionaries.
    - metric (str): Metric to use for aggregation.

    Returns:
    - dict: Dictionary containing the aggregated results for the specified metric.
    """
    iou_scorers = {
        "IoU": lambda x: x,
        "ACC@0.1": lambda x: compute_accuracy(x, 0.1),
        "ACC@0.3": lambda x: compute_accuracy(x, 0.3),
        "ACC@0.5": lambda x: compute_accuracy(x, 0.5),
        "ACC@0.7": lambda x: compute_accuracy(x, 0.7),
        "ACC@0.9": lambda x: compute_accuracy(x, 0.9),
    }
    scorers = {
        "Center_ACC": lambda x: x,
    }
    results_dict = {
        metric: [],
    }
    for result in results:
        # Extract the ground truth and predicted bounding boxes
        gt = result["bbox"]
        pred = result["pred"]

        if metric in iou_scorers:
            # Extract the IoU
            iou = result["iou"]
            # Compute the specified metric between the ground truth and predicted bounding boxes
            score = iou_scorers[metric](iou)
        elif metric in scorers:
            # Extract the center accuracy
            center_acc = result["center_acc"]
            # Compute the specified metric between the ground truth and predicted bounding boxes
            score = scorers[metric](center_acc)

        results_dict[metric].append(score)

    for key in results_dict:
        if len(results_dict[key]) == 0:
            results_dict[key] = 0
        else:
            results_dict[key] = sum(results_dict[key]) / len(results_dict[key])

        print(f"{key}: {results_dict[key]:0.4f}")
    return results_dict[metric]


def screenspot_rec_iou(results):
    return screenspot_rec_aggregation_result(results, "IoU")


def screenspot_rec_acc01(results):
    return screenspot_rec_aggregation_result(results, "ACC@0.1")


def screenspot_rec_acc03(results):
    return screenspot_rec_aggregation_result(results, "ACC@0.3")


def screenspot_rec_acc05(results):
    return screenspot_rec_aggregation_result(results, "ACC@0.5")


def screenspot_rec_acc07(results):
    return screenspot_rec_aggregation_result(results, "ACC@0.7")


def screenspot_rec_acc09(results):
    return screenspot_rec_aggregation_result(results, "ACC@0.9")


def screenspot_rec_center_acc(results):
    return screenspot_rec_aggregation_result(results, "Center_ACC")
