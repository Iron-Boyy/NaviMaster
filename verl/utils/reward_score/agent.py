import re
from typing import Dict
import json
import math

def calculate_f1_score(predicted_str, ground_truth_str):
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())
    
    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    if len(predicted_tokens) == 0:
        precision = 0
    else:
        precision = len(common_tokens) / len(predicted_tokens)
    if len(ground_truth_tokens) == 0:
        recall = 0
    else:
        recall = len(common_tokens) / len(ground_truth_tokens)
    
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def grounding_reward(predict_str: str, ground_truth: str) -> float:
    answer_tag_pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
    reward = 0.0
    theta = 20
    try:
        content_answer_match = re.search(answer_tag_pattern, predict_str, re.DOTALL)
        if content_answer_match:
            content_answer = content_answer_match.group(2).strip()
            action = json.loads(content_answer)[0]
            gt_action = json.loads(ground_truth)
            if action['action'] == gt_action['action']:
                if action['action']=='click' or action['action']=='long_press':
                    action['x']= action['point'][0]
                    action['y']= action['point'][1]

                    if math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= theta: 
                        reward=1.0
                    elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 2*theta: 
                        reward=0.9
                    elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 3*theta: 
                        reward=0.8
                    elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 4*theta: 
                        reward=0.7
                    elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 5*theta: 
                        reward=0.6
                    elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 6*theta: 
                        reward=0.5
                    elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 7*theta: 
                        reward=0.4
                    elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 8*theta: 
                        reward=0.3
                    elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 9*theta: 
                        reward=0.2
                    elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 10*theta: 
                        reward=0.1
                    else:
                        reward=0.0
                elif action['action']=='input_text' and gt_action['task_type']!='aguvis' :
                    action['x']= action['point'][0]
                    action['y']= action['point'][1]
                    if gt_action["text"]==action["text"] or calculate_f1_score(gt_action["text"], action["text"])>0.5: 
                        if math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= theta: 
                            reward=1.0
                        elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 2*theta: 
                            reward=0.9
                        elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 3*theta: 
                            reward=0.8
                        elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 4*theta: 
                            reward=0.7
                        elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 5*theta: 
                            reward=0.6
                        elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 6*theta: 
                            reward=0.5
                        elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 7*theta: 
                            reward=0.4
                        elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 8*theta: 
                            reward=0.3
                        elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 9*theta: 
                            reward=0.2
                        elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 10*theta: 
                            reward=0.1
                        else:
                            reward=0.0
                    else:
                        reward=0.0
                elif action['action']=='scroll' and gt_action['task_type']!='aguvis':
                    action['start_x']= action['start_point'][0]
                    action['start_y']= action['start_point'][1]
                    action['end_x']= action['end_point'][0]
                    action['end_y']= action['end_point'][1]
                    if math.sqrt((action['start_x'] - gt_action['gt_bbox'][0])**2 + (action['start_y'] - gt_action['gt_bbox'][1])**2) <= theta and math.sqrt((action['end_x'] - gt_action['gt_bbox'][2])**2 + (action['end_y'] - gt_action['gt_bbox'][3])**2) <= theta: 
                        reward=1.0
                    elif math.sqrt((action['start_x'] - gt_action['gt_bbox'][0])**2 + (action['start_y'] - gt_action['gt_bbox'][1])**2) <= 2*theta and math.sqrt((action['end_x'] - gt_action['gt_bbox'][2])**2 + (action['end_y'] - gt_action['gt_bbox'][3])**2) <= 2*theta: 
                        reward=0.9
                    elif math.sqrt((action['start_x'] - gt_action['gt_bbox'][0])**2 + (action['start_y'] - gt_action['gt_bbox'][1])**2) <= 3*theta and math.sqrt((action['end_x'] - gt_action['gt_bbox'][2])**2 + (action['end_y'] - gt_action['gt_bbox'][3])**2) <= 3*theta: 
                        reward=0.8
                    elif math.sqrt((action['start_x'] - gt_action['gt_bbox'][0])**2 + (action['start_y'] - gt_action['gt_bbox'][1])**2) <= 4*theta and math.sqrt((action['end_x'] - gt_action['gt_bbox'][2])**2 + (action['end_y'] - gt_action['gt_bbox'][3])**2) <= 4*theta: 
                        reward=0.7
                    elif math.sqrt((action['start_x'] - gt_action['gt_bbox'][0])**2 + (action['start_y'] - gt_action['gt_bbox'][1])**2) <= 5*theta and math.sqrt((action['end_x'] - gt_action['gt_bbox'][2])**2 + (action['end_y'] - gt_action['gt_bbox'][3])**2) <= 5*theta: 
                        reward=0.6
                    elif math.sqrt((action['start_x'] - gt_action['gt_bbox'][0])**2 + (action['start_y'] - gt_action['gt_bbox'][1])**2) <= 6*theta and math.sqrt((action['end_x'] - gt_action['gt_bbox'][2])**2 + (action['end_y'] - gt_action['gt_bbox'][3])**2) <= 6*theta: 
                        reward=0.5
                    elif math.sqrt((action['start_x'] - gt_action['gt_bbox'][0])**2 + (action['start_y'] - gt_action['gt_bbox'][1])**2) <= 7*theta and math.sqrt((action['end_x'] - gt_action['gt_bbox'][2])**2 + (action['end_y'] - gt_action['gt_bbox'][3])**2) <= 7*theta: 
                        reward=0.4
                    elif math.sqrt((action['start_x'] - gt_action['gt_bbox'][0])**2 + (action['start_y'] - gt_action['gt_bbox'][1])**2) <= 8*theta and math.sqrt((action['end_x'] - gt_action['gt_bbox'][2])**2 + (action['end_y'] - gt_action['gt_bbox'][3])**2) <= 8*theta: 
                        reward=0.3
                    elif math.sqrt((action['start_x'] - gt_action['gt_bbox'][0])**2 + (action['start_y'] - gt_action['gt_bbox'][1])**2) <= 9*theta and math.sqrt((action['end_x'] - gt_action['gt_bbox'][2])**2 + (action['end_y'] - gt_action['gt_bbox'][3])**2) <= 9*theta: 
                        reward=0.2
                    elif math.sqrt((action['start_x'] - gt_action['gt_bbox'][0])**2 + (action['start_y'] - gt_action['gt_bbox'][1])**2) <= 10*theta and math.sqrt((action['end_x'] - gt_action['gt_bbox'][2])**2 + (action['end_y'] - gt_action['gt_bbox'][3])**2) <= 10*theta: 
                        reward=0.1
                    else:
                        reward=0.0
                elif action['action'] in ['moveto']:
                    action['x']= action['point'][0]
                    action['y']= action['point'][1]
                    if math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= theta: 
                        reward=1.0
                    elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 2*theta: 
                        reward=0.9
                    elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 3*theta: 
                        reward=0.8
                    elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 4*theta: 
                        reward=0.7
                    elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 5*theta: 
                        reward=0.6
                    elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 6*theta: 
                        reward=0.5
                    elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 7*theta: 
                        reward=0.4
                    elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 8*theta: 
                        reward=0.3
                    elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 9*theta: 
                        reward=0.2
                    elif math.sqrt((action['x'] - gt_action['gt_bbox'][0])**2 +(action['y'] - gt_action['gt_bbox'][1])**2) <= 10*theta: 
                        reward=0.1
                    else:
                        reward=0.0
                else:
                    if action['action'] in ['navigate_home', 'navigate_back', 'turn_around', 'turn_left', 'turn_right', 'stop', 'wait']:
                        reward=1.0
                    elif action['action']=='scroll':
                        if gt_action["text"]==action["direction"]: 
                            reward=1.0
                        else:
                            reward=0.0
                    elif action['action']=='open_app':
                        if gt_action["text"]==action["app_name"]: 
                            reward=1.0
                        else:
                            reward=0.0
                    else:
                        reward=0.0
    except Exception:
        reward = 0.0
    return reward

def type_reward(predict_str: str, ground_truth: str) -> float:
    answer_tag_pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
    reward = 0.0
    try:
        content_answer_match = re.search(answer_tag_pattern, predict_str, re.DOTALL)

        if content_answer_match:
            content_answer = content_answer_match.group(2).strip()
            action = json.loads(content_answer)[0]
            gt_action = json.loads(ground_truth)
            if action['action'] == gt_action['action']:
                if action['action']=='input_text':
                    if gt_action["text"]==action["text"] or calculate_f1_score(gt_action["text"], action["text"])>0.5: 
                        reward=1.0
                    else:
                        reward=0.5
                else:
                    reward=1.0
            else:
                reward=0.0
    except Exception:
        reward = 0.0
    return reward

def format_reward(predict_str: str) -> float:
    answer_tag_pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'
    content_answer_match = re.search(answer_tag_pattern, predict_str, re.DOTALL)
    reward = 0.0
    if content_answer_match:
        content_answer = content_answer_match.group(2).strip()
        try:
            action = json.loads(content_answer)[0]
            if "action" not in action.keys():
                reward = tag_reward
            elif action['action'] in ["click","long_press", "moveto"]:
                if len(action['point'])==2 :
                    reward = 1.0
                else:
                    reward = 0.0
            elif action['action'] == "input_text":
                if "text" in action.keys():
                    reward = 1.0
                else:
                    reward = 0.0
            elif action['action'] == "open_app":
                if "app_name" in action.keys():
                    reward = 1.0
                else:
                    reward = 0.0
            elif action['action'] == "scroll":
                if 'direction' in action.keys():
                    if action['direction'] in ['up', 'down', 'left', 'right']:
                        reward = 1.0
                    else:
                        reward = 0.0
                else:
                    if len(action['start_point'])==2 and len(action['end_point'])==2:
                        reward = 1.0
                    else:
                        reward = 0.0
            elif action['action'] == "navigate_home":
                reward = 1.0
            elif action['action'] == "navigate_back":
                reward = 1.0
            elif action['action'] == "turn_left":
                reward = 1.0
            elif action['action'] == "turn_right":
                reward = 1.0
            elif action['action'] == "turn_around":
                reward = 1.0
            elif action['action'] == "look_down":
                reward = 1.0
            elif action['action'] == "stop":
                reward = 1.0
            elif action['action'] == "wait":
                reward = 1.0
            else:
                reward = 0.0  # 如果是JSON格式，给予奖励
        except:
            reward = 0.0 # 如果不是JSON格式，不给予奖励
    return reward

def agent_compute_score(predict_str: str, ground_truth: str, format_weight: float = 0.1) -> Dict[str, float]:
    format_score = format_reward(predict_str)
    type_score = type_reward(predict_str, ground_truth)
    grounding_score = grounding_reward(predict_str, ground_truth)

    return {
        "overall": grounding_score+ 0.1*format_score+ type_score,
        "format": format_score,
        "grounding": grounding_score,
        "type": type_score,
    }
