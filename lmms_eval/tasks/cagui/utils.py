from lmms_eval.tasks._task_utils.gui_utils import *
import os
import time
from loguru import logger as eval_logger
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter
import numpy as np
import re
from PIL import Image
import io
import base64
import math
import json
from typing import Union, List

_TAP_DISTANCE_THRESHOLD = 0.14  # Fraction of the screen
_TAP_DISTANCE_THRESHOLD_AC = 0.04  # for android control, align with qwen's code.
_SWIPE_DISTANCE_THRESHOLD = 0.04 # Interval determining if an action is a tap or a swipe.
ANNOTATION_WIDTH_AUGMENT_FRACTION= 1.2 # aitw set it to 1.4, aitz and qwen 2.5 vl set it to 1.2.
ANNOTATION_HEIGHT_AUGMENT_FRACTION= 1.2 # We follow qwen setting.

def _resize_annotation_bounding_boxes(
    annotation_position: Union[List[float], List[List[float]]],
    width_factor: float = 1.2,
    height_factor: float = 1.2,
):
    """Uniformly enlarge bbox(es) by the given factors."""

    def _resize(box: List[float]):
        y, x, h, w = box
        h_delta = (height_factor - 1) * h
        w_delta = (width_factor - 1) * w
        y = max(0, y - h_delta / 2)
        x = max(0, x - w_delta / 2)
        h = min(1, h + h_delta)
        w = min(1, w + w_delta)
        return [y, x, h, w]

    if not annotation_position:
        return []
    if isinstance(annotation_position[0], list):
        return [_resize(b) for b in annotation_position]
    return _resize(annotation_position)

def check_inside(x, y, bbox_list):
    bbox_array = np.array(bbox_list)
    y_min, x_min, height, width = bbox_array[:, 0], bbox_array[:, 1], bbox_array[:, 2], bbox_array[:, 3]
    y_max, x_max = y_min + height, x_min + width

    # Check whether (x, y) is inside any of the bounding boxes
    within_x = (x_min <= x) & (x <= x_max)
    within_y = (y_min <= y) & (y <= y_max)
    within_bbox = within_x & within_y

    if np.any(within_bbox):
        within_bbox_coords = bbox_array[within_bbox]
        return True, within_bbox_coords
    else:
        return False, None

def obtain_gt_bbox(coordinate, bbox_list, eval_android_control=False):
    x, y = coordinate['x'], coordinate['y']
    if len(bbox_list) == 0:
        return []

    if not eval_android_control:
        is_inside, bbox_inside = check_inside(x, y, bbox_list)
        if is_inside:
            return bbox_inside.tolist()
        else:
            return []
    else:
        def get_center_distance(box):
            ymin, xmin, h, w = box
            center_y = ymin + h/2
            center_x = xmin + w/2
            return ((center_y - y) ** 2 + (center_x - x) ** 2) ** 0.5

        sorted_boxes = sorted(bbox_list, key=get_center_distance)
        # return the 5 nearest bboxes
        return sorted_boxes[:5]

def cagui_doc_to_visual(doc):
    # Image is presented as is
    image = Image.open(io.BytesIO(base64.b64decode(doc["image_base64"])))
    return [image.convert("RGB")]

def cagui_doc_to_text(doc, llm_eval_specific_kwargs):
    model_name = llm_eval_specific_kwargs.get("model_name", "mimo_agent")
    goal = doc["instruction"]
    history_list = doc["history_list"]
    prompt = ""
    if model_name == "uitars":
        prompt = prompt_uitars.format(goal=goal)
        new_prompt = f"{prompt}<think>"
        return new_prompt
    elif model_name == "mimo_agent":
        history_actions = []
        for history_item in history_list:
            history_actions.append(history_item["action"])
        image_height, image_width = doc["image_height"], doc["image_width"]
        history_actions = transfer_history_data_into_absolute_style(history_actions, image_width, image_height, convert_scroll=True)
        history_action_str = ""
        for idx, history_action in enumerate(history_actions):
            history_action_str += f"Step {idx}: {history_action}."
        history_action_str = "None" if history_action_str == "" else history_action_str

        # action_space = ['{"action": "click", "start_point": [x,y]}', '{"action": "input", "text": "text"}', '{"action": "open", "app": "app_name"}', '{"action": "longpress", "start_point": [x,y]}', '{"action": "scroll", "direction": "direction"}', '{"action": "press", "keys": [key1, key2, ...]}', '{"action": "wait"}', '{"action": "finished", "status": "status"}']
        action_space = ['{"action": "click", "start_point": [x,y]}', '{"action": "drag", "start_point": [x,y], "end_point": [x,y]}', '{"action": "input", "text": "text"}', '{"action": "press", "keys": [key1, key2, ...]}', '{"action": "wait"}', '{"action": "finished", "status": "status"}']
        
        action_space_str = "\n".join(action_space)
        prompt = prompt_mimo_wo_instruction.format(action_space=action_space_str, goal=goal, previous_actions=history_action_str)   
    elif model_name == "qwen25_vl":
        pass
    return prompt

def cagui_process_result(doc, result):
    gt_action = doc["action"]
    gt_ui_positions = json.loads(doc["ui_positions"])
    result = result[0]

    metric_dict = {
        "action_type_acc": 0,
        "step_wise_acc": 0,
        "parse_error": 0,
    }

    try:
        pred_action = json.loads(result)
        if pred_action["action_name"] == "":
            metric_dict["parse_error"] = 1
            return metric_dict
    except Exception as e:
        metric_dict["parse_error"] = 1
        return metric_dict
    
    gt_action_type = gt_action["action_name"]
    pred_action_type = pred_action["action_name"]
    type_match = (pred_action_type is not None and gt_action_type == pred_action_type)
    if type_match:
        metric_dict["action_type_acc"] = 1

    exact_match = False

    if type_match and (gt_action_type == "click" or gt_action_type == "longpress"):
        metric_dict["gt_coord"] = (gt_action["start_point"][0], gt_action["start_point"][1])
        metric_dict["pred_coord"] = (pred_action["start_point"][0], pred_action["start_point"][1])
        if pred_action["start_point"] is None:
            metric_dict["parse_error"] = 1
            return metric_dict
        
        try:
            gt_cand_nodes = _resize_annotation_bounding_boxes(gt_ui_positions, width_factor=ANNOTATION_WIDTH_AUGMENT_FRACTION, height_factor=ANNOTATION_HEIGHT_AUGMENT_FRACTION)
        except Exception as e:
            eval_logger.warning(f"Error in _resize_annotation_bounding_boxes: {e}")
            # gt_cand_nodes = gt_ui_positions
            # print(gt_ui_positions)
            metric_dict["parse_error"] = 1
            return metric_dict
        
        gt_start_point = gt_action["start_point"]
        pred_start_point = pred_action["start_point"]
        gt_bbox = obtain_gt_bbox({"x": gt_start_point[0], "y": gt_start_point[1]}, gt_cand_nodes, eval_android_control=False)

        if gt_bbox == []:
            x_gt, y_gt = gt_start_point[0], gt_start_point[1]
            x_pd, y_pd = pred_start_point[0], pred_start_point[1]
            distance = np.linalg.norm(np.array([x_gt, y_gt]) - np.array([x_pd, y_pd]))
            exact_match = bool(distance <= (_TAP_DISTANCE_THRESHOLD))
        else:
            reference_point = gt_start_point[0], gt_start_point[1]
            x_pd, y_pd = pred_start_point[0], pred_start_point[1]
            for bbox in gt_bbox:
                ymin, xmin, h, w = bbox
                ymax, xmax = ymin + h, xmin + w
                exact_match = bool(
                    (xmin <= x_pd <= xmax)
                    and (ymin <= y_pd <= ymax)
                )
                if exact_match:
                    break
            if not exact_match:
                x_gt, y_gt = gt_start_point[0], gt_start_point[1]
                distance = np.linalg.norm(np.array([x_gt, y_gt]) - np.array([x_pd, y_pd]))
                exact_match = bool(distance <= (_TAP_DISTANCE_THRESHOLD))

        metric_dict["step_wise_acc"] = int(exact_match)
    elif type_match and (gt_action_type == "scroll"):
        if pred_action["direction"] is None:
            metric_dict["parse_error"] = 1
            return metric_dict
        
        # map_dict = {
        #     "up": "down",
        #     "down": "up",
        #     "left": "left",
        #     "right": "right",
        # }
        # if pred_action["direction"] in map_dict:
        #     exact_match = (map_dict[pred_action["direction"]] == gt_action["direction"])
        # else:
        #     exact_match = False
        exact_match = (pred_action["direction"] == gt_action["direction"])
        metric_dict["step_wise_acc"] = int(exact_match)
    elif type_match and (gt_action_type == "input"):
        if pred_action["text"] is None:
            metric_dict["parse_error"] = 1
            return metric_dict
        
        pd_text_norm = pred_action["text"].lower().strip()
        gt_text_norm = gt_action["text"].lower().strip()

        # align with Qwen‑2.5‑VL eval
        exact_match = (pd_text_norm in gt_text_norm or \
                    gt_text_norm in pd_text_norm)
        
        metric_dict["step_wise_acc"] = int(exact_match)
    elif type_match and (gt_action_type == "press"):
        if pred_action["keys"] is None:
            metric_dict["parse_error"] = 1
            return metric_dict
        
        exact_match = (pred_action["keys"] == gt_action["keys"])
        metric_dict["step_wise_acc"] = int(exact_match)
    elif type_match and (gt_action_type == "open"):
        if pred_action["app_name"] is None:
            metric_dict["parse_error"] = 1
            return metric_dict
        
        exact_match = (pred_action["app_name"] == gt_action["app_name"])
        metric_dict["step_wise_acc"] = int(exact_match)
    elif type_match and (gt_action_type == "wait"):
        metric_dict["step_wise_acc"] = 1
    elif type_match and (gt_action_type == "finished"):
        metric_dict["step_wise_acc"] = 1
    
    # if gt_action_type == "finished" and pred_action_type == "wait":
    #     metric_dict["step_wise_acc"] = 1
    #     metric_dict["action_type_acc"] = 1
    
    # metric_dict["step_wise_acc"] = 1
    return metric_dict


def cagui_aggregate_result(results):
    return np.mean(results)


