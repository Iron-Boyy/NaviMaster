import re
import json
from PIL import Image

def calculate_f1_score(predicted_str, ground_truth_str):
    predicted_str=predicted_str.replace("[","").replace("]","")
    ground_truth_str=ground_truth_str.replace("[","").replace("]","")
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())

    if len(predicted_tokens)==1 and len(ground_truth_tokens)==1:
        predicted_token=list(predicted_tokens)[0]
        ground_truth_token=list(ground_truth_tokens)[0]
        if predicted_token in ground_truth_token or ground_truth_token in predicted_token:
            return 1
    
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

def extract_action(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    action_pattern = r"'action':\s*'(\w+)'"
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        action_match = re.search(action_pattern, content_answer)
        if action_match:
            return action_match.group(1)
    return "no action"

def extract_input_text(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    action_pattern = r"'input_text':\s*'(.*?)'"
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        action_match = re.search(action_pattern, content_answer)
        if action_match:
            return action_match.group(1)
    return "no input text"

def extract_coord(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\{.*\[(\d+),\s*(\d+)]\s*.*\}'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    try:
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            coord_match = re.search(bbox_pattern, content_answer)
            if coord_match:
                coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                return coord, True
        else:
            coord_pattern = r'\{.*\((\d+),\s*(\d+))\s*.*\}'
            coord_match = re.search(coord_pattern, content)
            if coord_match:
                coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                return coord, True
        return [0, 0, 0, 0], False
    except:
        return [0, 0, 0, 0], False

def extract_box(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\{.*\[\[(\d+),\s*(\d+)\],\s*\[(\d+),\s*(\d+)\]\]\s*.*\}'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    try:
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            coord_match = re.search(bbox_pattern, content_answer)
            if coord_match:
                coord = [int(coord_match.group(1)), int(coord_match.group(2)), int(coord_match.group(3)), int(coord_match.group(4))]
                return coord, True
        # else:
        #     coord_pattern = r'\{.*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+))\s*.*\}'
        #     coord_match = re.search(coord_pattern, content)
        #     if coord_match:
        #         coord = [int(coord_match.group(1)), int(coord_match.group(2)), int(coord_match.group(3)), int(coord_match.group(4))]
        #         return coord, True
        return [0, 0, 0, 0], False
    except:
        return [0, 0, 0, 0], False

def extract_robopoint(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'^\((\d+)\,\s*(\d+)\)$'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    try:
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            coord_match = re.search(bbox_pattern, content_answer)
            if coord_match:
                coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                return coord, True
        return [0, 0, 0, 0], False
    except:
        return [0, 0, 0, 0], False
    
import numpy as np

def compute_iou(box1, box2):
    """
    计算两个矩形框的交并比（IOU）。

    参数：
    - box1: 第一个矩形框的坐标，格式为 [x1, y1, x2, y2]。
    - box2: 第二个矩形框的坐标，格式为 [x1, y1, x2, y2]。

    返回：
    - iou: 交并比（IOU）值。
    """

    # 计算两个矩形框的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算两个矩形框的交集
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 如果两个矩形框没有交集，返回0
    if x1 >= x2 or y1 >= y2:
        return 0

    # 计算交集的面积
    intersection_area = (x2 - x1) * (y2 - y1)

    # 计算并集的面积
    union_area = area1 + area2 - intersection_area

    # 计算交并比（IOU）
    iou = intersection_area / union_area
    if iou >0.8:
        return 1.0
    else:
        return 0.0


def compute_dis(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    
    # 计算欧氏距离
    distance = np.sqrt(np.sum((point1[:-1] - point2[:-1]) ** 2))
    depth_dis = np.abs(point1[-1] - point2[-1])
    if distance <80 and depth_dis<0.2:
        return 1.0
    else:
        return 0.0

def compute_2ddis(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    
    # 计算欧氏距离
    distance = np.sqrt(np.sum((point1 - point2) ** 2))
    if distance <50:
        return 1.0
    else:
        return 0.0
def r1gui_format_reward(predict_str: str, ground_truth: str) -> float:
    """
    检查 predict_str 是否符合 <think></think><answer></answer> 的格式，
    并验证 <answer> 中的内容是否符合 [{'action': 'action', 'point': '[x,y]', 'input_text': 'no input text'}] 的格式要求。
    """
    # 检查 <think> 和 <answer> 的外部结构
    ground_truth=json.loads(ground_truth)
    outer_pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    if not re.fullmatch(outer_pattern, predict_str):
        return 0.0

    # 提取 <answer> 中的内容
    answer_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
    if not answer_match:
        return 0.0

    # 提取 <answer> 内的内容并解析为 JSON 格式
    answer_content = answer_match.group(1).strip()
    try:
        if ground_truth['task_type'] == 'robopoint':
            pattern = r'^\((\d+)\,\s*(\d+)\)$'
            try:
                coord_match = re.search(pattern, answer_content)
                if coord_match:
                    return 1.0
                return 0.0
            except:
                return 0.0
            
        actions = eval(answer_content)  # 尝试将 <answer> 的内容解析为 JSON

        # 验证 actions 是否为列表
        if not isinstance(actions, list):
            return 0.0

        # 验证每个 action 的格式
        if ground_truth['task_type'] == 'affordance':
            for action in actions:
                if not isinstance(action, dict):
                    return 0.0
                # 检查 action 字典是否包含所需的键
                if "box" not in action:
                    return 0.0
                # 验证 action 的值是否符合要求
                if not (isinstance(action["box"][0][0],int) and isinstance(action["box"][0][1],int)) and isinstance(action["box"][1][0],int) and isinstance(action["box"][1][1],int) and action["box"].size():  # 匹配形如 [x,y] 的点
                    return 0.0
        elif ground_truth['task_type'] == 'navigation':
            for action in actions:
                if not isinstance(action, dict):
                    return 0.0
                # 检查 action 字典是否包含所需的键
                if "action" not in action:
                    return 0.0
                # 验证 action 的值是否符合要求
                if action["action"] == "moveto":
                    if not (isinstance(action["point"][0],int) and isinstance(action["point"][1],int)):  # 匹配形如 [x,y] 的点
                        return 0.0
                else:
                    if len(action.keys())>1:
                        return 0.0
            if len(actions) > 1:
                return 0.0
        else:
            for action in actions:
                if not isinstance(action, dict):
                    return 0.0
                # 检查 action 字典是否包含所需的键
                if "action" not in action or "point" not in action or "input_text" not in action:
                    return 0.0
                # 验证 action 的值是否符合要求
                if not isinstance(action["action"], str):
                    return 0.0
                if not (isinstance(action["point"][0],int) and isinstance(action["point"][1],int)):  # 匹配形如 [x,y] 的点
                    return 0.0
                if not isinstance(action["input_text"], str):
                    return 0.0
                if action["action"] in ['type', 'select','open_app'] and action["input_text"] in ['no input text']:
                    return 0.0
                if action["action"] in ['scroll'] and action["input_text"] not in ['left','right','up','down']:
                    return 0.0

        # 如果所有检查均通过，返回 1.0
        return 1.0
    except:
        return 0.0

def r1gui_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    """
    比较 predict_str 和 ground_truth 中的动作和参数是否一致。
    """
    try:
        # 提取 ground_truth 的动作和参数
        ground_truth=json.loads(ground_truth)
        if ground_truth['task_type'] == 'affordance':
            gt_bbox=ground_truth['gt_bbox']
            pred_bbox,_=extract_box(predict_str)
            return compute_iou(gt_bbox, pred_bbox)
        elif ground_truth['task_type'] == 'robopoint':
            gt_point=ground_truth['action']
            pred_point,_=extract_robopoint(predict_str)
            return compute_2ddis(gt_point, pred_point)
        elif ground_truth['task_type'] == 'navigation':
            pred_action=extract_action(predict_str).lower()
            if pred_action in("turn_left", "turn_right", "turn_around", "look_down", "stop"):
                if pred_action == ground_truth['action']:
                    return 1.0
                else:
                    return 0.0
            else:
                pred_point,_=extract_coord(predict_str)
                coord_pattern = r'\((\d+),\s*(\d+)\)'
                gt_point=ground_truth['action']
                matches = re.findall(coord_pattern, gt_point)
                point2 = []
                for match in matches:
                    x, y = map(int, match)
                    point2.append(x)
                    point2.append(y)
                depth_img = Image.open(ground_truth['gt_depth_path'])
                depth_array = np.array(depth_img, dtype=np.uint8)
                restored_depth = depth_array.astype(np.float32) / 255.0 * 10.0
                point1 = [pred_point[0], pred_point[1],restored_depth[pred_point[1]][pred_point[0]]]
                point2.append(restored_depth[point2[1]][point2[0]])
                return compute_dis(point1, point2)
        else:
            gt_action=ground_truth['action'].lower()
            gt_bbox=ground_truth['gt_bbox']
            gt_input_text=ground_truth['input_text']
            pred_action=extract_action(predict_str).lower()
            pred_input_text=extract_input_text(predict_str)
            pred_bbox,_=extract_coord(predict_str)
            
            if pred_action!=gt_action:
                return 0.0
            
            if gt_action in ["click"]:
                if len(gt_bbox)==2:
                    if (pred_bbox[0]-gt_bbox[0])**2+(pred_bbox[1]-gt_bbox[1])**2<140**2:
                        return 1.0
                    else:
                        return 0.0
                elif len(gt_bbox)==4:
                    if (gt_bbox[0]<pred_bbox[0]<gt_bbox[2]) and (gt_bbox[1]<pred_bbox[1]<gt_bbox[3]):
                        return 1.0
                    else:
                        return 0.0
                else:
                    return 0.0
            elif gt_action in ['type', 'select','scroll']:
                if calculate_f1_score(pred_input_text,gt_input_text)>=0.5:
                    return 1.0
                else:
                    return 0.0
            else:
                return 1.0

    except Exception as e:
        return 0.0
    
def r1gui_compute_score(predict_str: str, ground_truth: str):
    format = r1gui_format_reward(predict_str, ground_truth)
    accuracy = r1gui_accuracy_reward(predict_str, ground_truth)
    log_path = "/nas/shared/kilab/luozhihao/test.txt"
    try:
        with open(log_path, "a") as f:
            f.write(f"Content: {predict_str}\n")
            f.write(f"Solution: {ground_truth}\n")
            f.write(f"Score: {format},{accuracy}\n")
    except:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"Content: {predict_str}\n")
            f.write(f"Solution: {ground_truth}\n")
            f.write(f"Score: {format},{accuracy}\n")
    return {
        "overall": 0.8 * accuracy + 0.2 * format,
        "format": format,
        "accuracy": accuracy,
    }

# pr=("<think> The command 'What's on the menu at IHOP?' suggests a search for information about the menu at an IHOP restaurant. However, "
# "the current UI screenshot is a calendar application displaying holidays and significant dates for the month of October and November. There is no direct way to per"
# "form a web search or access an IHOP menu from this calendar app. Therefore, the appropriate action would be to exit the current application and open a web browser"
# "or a dedicated app for searching the IHOP menu. "                                                                                                               
# "Since the action history is 'None', the first step is to navigate away from the current app to a web browser or a search engine.</think> "
# " <answer>[{'action': 'scroll', 'point': [123, 401], 'input_text': 'left'}]</answer>")
# gt=json.dumps({"action": "scroll", "gt_bbox": [103.0, 409.18800000000005], "input_text": "LEFT"})
# print(gr_iou_accuracy_reward(pr,gt))
# print(gr_format_reward(pr))