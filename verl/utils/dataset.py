# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import re
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF
import json
import io

def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: Union[Dict[str, Any], ImageObject], max_pixels: int, min_pixels: int) -> ImageObject:
    if isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, str):
        image = Image.open(image)
    else:
        image = Image.open(BytesIO(image))
    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image

def extract_robopoint(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    bbox_pattern = r'^\((\d+)\,\s*(\d+)\)$'
    coord_match = re.search(bbox_pattern, content)
    if coord_match:
        coord = [int(coord_match.group(1)), int(coord_match.group(2))]
        return coord, True
    return [0, 0, 0, 0], False
class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        system_prompt: str = None,
        max_pixels: int = None,
        min_pixels: int = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"
            # print(data_path)

        if os.path.isdir(data_path):
            self.dataset = load_dataset("parquet", data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            self.dataset = load_dataset("parquet", data_files=data_path, split="train")
        else:  # remote dataset
            self.dataset = load_dataset(data_path, split=data_split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row_dict: dict = self.dataset[index]
        

        # prompt_str: str = row_dict[self.prompt_key]
        text=row_dict['instruction']
        history=row_dict['history']
        task_type=row_dict['task_type']
        try:
            image_x, image_y = Image.open(io.BytesIO(row_dict['image'])).size
            image_size = [image_x, image_y]
        except:
            image_x, image_y = Image.open(io.BytesIO(row_dict['image']['bytes'])).size
            image_size = [image_x, image_y]
        row_dict.pop('verify_bbox', None)
        row_dict.pop('success_rate', None)
        row_dict.pop('scale', None)
        images=[row_dict['image']]
      
        if task_type=='high':
            prompt_str=  (
                f"You are a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue executing the command '{text}', with the action history being '{history}'.\n"
                "Please provide the action to perform (enumerate from ['complete', 'close/delete', 'press_home', 'click', 'press_back', 'type', 'select', 'scroll', 'enter']), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.\n"
                "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
                "<think> ... </think> <answer>[{'action': enum['complete', 'close/delete', 'press_home', 'click', 'press_back', 'type', 'select', 'scroll', 'enter'], 'point': [x, y], 'input_text': 'no input text [default]'}]</answer>\n"
                "Note:\n specific input text (no default) is necessary for actions enum['type', 'select', 'scroll'] \n Example:\n"
                "[{'action': enum['complete', 'close/delete', 'press_home', 'press_back', 'enter'], 'point': [-100, -100], 'input_text': 'no input text'}]\n"
                "[{'action': enum['click'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
                "[{'action': enum['type', 'select'], 'point': [-100, -100], 'input_text': 'shanghai shopping mall'}]\n"
                "[{'action': enum['scroll'], 'point': [-100, -100], 'input_text': enum['up', 'left', 'right', 'down']}]"
            )
        elif task_type=='low':
            prompt_str=(
                f"You are a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue executing the command '{text}', with the action history being '{history}'.\n"
                "Please provide the action to perform (enumerate from ['click']), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.\n"
                "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
                '<think> ... </think> <answer>[{"action": enum["click"], "point": [x, y], "input_text": "no input text"}]</answer>\n'
                "Example:\n"
                '[{"action": "click", "point": [123, 300], "input_text": "no input text"}]\n'
            )
        elif task_type=='affordance':
            prompt_str=(
                f"You are a reasoning Agent Assistant using joint control. In this photo <image>, the task is '{text}'.\n"
                "Please predict all possible affordance areas of the end effector, the box represent where you need to focus on if an operation is performed.\n"
                "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
                "<think> ... </think> <answer>{'box': [[x1, y1], [x2, y2]]}</answer>\n"
                "Note: The 'box' should contain the coordinates of the top-left corner [x1, y1] and the bottom-right corner [x2, y2] of the area of interest.\n"
                "Example:\n"
                "[{'box': [[123, 300], [156, 387]]]\n"
            )
        elif task_type=='navigation':
            prompt_str=(
                f"You are a Navigation Robot in an unfamiliar environment. In this photo <image>, the task is '{text}', with the history being '{history}'\n"
                "You need to use your prior knowledge about where items are typically located within a home. "
                "Please predict next action to find target item.\n"
                "Your action can be in the following list: \n"
                "Basic Action(move to a point on the ground in the picture):\n"
                '- Based on the image, predict the optimal location to move next to finish the task, use the coordinates (x, y)(x is the pixel from left to right and y is the pixel from top to bottom) to indicate where you want to move to:'
                ' [{"action": "moveto", "point": [x(position in horizontal(width)), y(position in vertical(height))]}].\n'
                'View Adjustment Actions(adjust view as current photo does not have suitable position):\n'
                '- Executes a 90-degree rotation to the left from the current facing direction. Ideal for navigating around obstacles on the right, aligning with a leftward path, or adjusting the view to inspect the left side of the environment. Use this when the task requires a lateral shift to the left: [{"action": "turn_left"}].\n'
                '- Rotates the perspective 90 degrees to the right. This action is useful when the target object or destination is positioned on the right, or when you need to change the direction to follow a rightward route: [{"action": "turn_right"}].\n'
                '- Performs a 180-degree rotation, flipping the orientation to face the opposite direction. This is valuable for finding a possible way if there is no path in front of you: [{"action": "turn_around"}].\n'
                '- Adjusts the camera view to look downwards, without physically moving the position. This is particularly useful for examining details on the ground, such as identifying objects, reading markings, or inspecting lower-level structures: [{"action": "look_down"}].\n'
                'Stop Action:\n'
                '- Carefully inspect the environment and judge from history, if you find your target in your view and has been close enough for about 1 meter, stop at current position: [{"action": "stop"}].\n'
                "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
                "<think> ... </think> <answer> answer here </answer>\n"
                "Note: The 'point' should contain the coordinates of the next destination. Coordinates are absolute coordinates(a center point defined by top-left and bottom-right coordinates). Ensure the predicted location is navigable(i.e., on the ground).\n"
                "Example:\n"
                '[{"action": "moveto", "point": [123, 300]}]\n'
            )
        elif task_type=='robopoint':
            prompt_str=(
                f'{text}\n'
                "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
                '<think> ... </think> <answer>[{"action": "moveto", "point": [x, y]}]</answer>\n'
                "Note: The coordinates should be between the size of picture, indicating the absolute pixel locations of the points in the image."
                "Example:\n"
                '[{"action": "moveto", "point": [123, 300]}]\n'
                
            )
        elif task_type=='mygui':
            text = text.replace('<image>', '')
            prompt_str=(
                    "In this photo <image>, A conversation between User and Assistant. The user asks a question, and the Assistant solves it step by step. The assistant "
                    "first thinks about the reasoning process in the mind and then provides the user with the answer. "
                    "At each step, you will be given the current screenshot and the history of the conversation(include sceenshot and action in each step)."
                    "Based on these pieces of information and the goal, you must give the whole content of what you think and then choose to perform one of the action in the following list "
                    "(action description followed by the JSON format) by outputing the action in the correct JSON format.\n"
                    '- Click/tap on an element on the screen. We have defined the width and height of the screenshot'
                    ', use the coordinates (x, y)(x is the pixel from left to right and y is the pixel from top to bottom)'
                    ' to indicate which element you want to click, both x and y are integers:'
                    ' [{"action": "click", "point": [x(position in horizontal(width)), y(position in vertical(height))]}].\n'
                    '- Long press on an element on the screen, similar with the click action'
                    ' above, use the coordinates (x, y)(x is the pixel from left to right and y is the pixel from top to bottom) to indicate which'
                    ' element you want to long press:'
                    ' [{"action": "long_press", "point": [x(position in horizontal(width)), y(position in vertical(height))]}].\n'
                    '- Type text into a text field (this action contains clicking on the target field, typing in the text and pressing the enter)'
                    ', use the coordinates (x, y)(x is the pixel from left to right and y is the pixel from top to bottom)'
                    ' to indicate which element you want to click, both x and y are integers:'
                    ' [{"action": "input_text", "text": <text_input>, "point": [x(position in horizontal(width)), y(position in vertical(height))]}]\n'
                    '- Navigate to the home screen: [{"action": "navigate_home"}]\n'
                    '- Navigate back: [{"action": "navigate_back"}]\n'
                    '- Scroll the screen or a scrollable UI element from start point to end point, use the coordinates (x, y)(x is the pixel from left to right and y is the pixel from top to bottom)'
                    ' to indicate the two points you want to scroll, both x and y are integers:'
                    ' [{"action": "scroll", "start_point": [<start position in horizontal(width)>, <start position in vertical(height)>], "end_point": [<end position in horizontal(width)>, <end position in vertical(height)>]}]\n'
                    f'The task and history is:{text}\n'
                    "NOTICES:1.Coordinates are absolute coordinates (a center point defined by top-left and bottom-right coordinates).\n"
                    "2.Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
                    '<think>...</think><answer>[{"action": ...}]</answer>\n'
                    "Example:\n"
                    '[{"action": "click", "point": [378, 871]}]\n'
                    '[{"action": "long_press", "point": [1082, 2001]}]\n'
                    '[{"action": "input_text", "text": "shanghai shopping mall", "point": [235, 1320]}]\n'
                    '[{"action": "navigate_home"}]\n'
                    '[{"action": "navigate_back"}]\n'
                    '[{"action": "scroll", "start_point": [80, 122], "end_point": [1203, 122]}]\n'
                    )


        if task_type=='aguvis':
            messages = [{"role": "user", "content": text}]
            images=[process_image(image, self.max_pixels, self.min_pixels) for image in images]
        else:
            messages = [{"role": "user", "content": prompt_str}]
            images=[process_image(image, self.max_pixels, self.min_pixels) for image in images]

        if task_type=='navigation':
            if row_dict['gt_action'] in ['turn_left', 'turn_right', 'stop', 'turn_back', 'look_down', 'turn_around', 'stop']:
                if row_dict['gt_action']=='turn_back':
                    gt={'action': 'turn_around', 'gt_bbox': [-100, -100], 'gt_depth_path': row_dict['depth_path'], 'task_type': task_type}
                else:
                    gt={'action': row_dict['gt_action'], 'gt_bbox': [-100, -100], 'gt_depth_path': row_dict['depth_path'], 'task_type': task_type}
            else:
                gt_bbox, _ = extract_robopoint(row_dict['gt_action'])
                gt={'action': 'moveto', 'gt_bbox': gt_bbox, 'gt_depth_path': row_dict['depth_path'], 'image_size': image_size, 'task_type': task_type}
        elif task_type=='robopoint':
            gt={'action':'moveto', 'gt_bbox': row_dict['gt'], 'image_size': image_size, 'task_type': task_type}
        elif task_type=='mygui' or 'aguvis':
            gt={'action': row_dict['gt_action'], 'gt_bbox': row_dict['gt_bbox'], 'text': row_dict['gt_input_text'], 'image_size': image_size, 'task_type': task_type}
        else:
            scalex,scaley=images[0].size
            gt_bbox=row_dict['gt_bbox']
            gt_bbox[0]*=scalex
            gt_bbox[1]*=scaley
            if len(gt_bbox)>2:
                gt_bbox[2]*=scalex
                gt_bbox[3]*=scaley

            gt={'action': row_dict['gt_action'],'gt_bbox': gt_bbox,'input_text': row_dict['gt_input_text'], 'image_size': image_size, 'task_type': task_type}

        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        row_dict["multi_modal_data"] = {
            "image": images
        }
        model_inputs = self.processor(row_dict["multi_modal_data"]["image"], prompt, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        row_dict["multi_modal_inputs"] = dict(model_inputs)
        position_ids = get_rope_index(
            self.processor,
            input_ids=input_ids,
            image_grid_thw=model_inputs["image_grid_thw"],
            attention_mask=attention_mask,
        )

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        row_dict["ground_truth"] = json.dumps(gt)
        return row_dict
