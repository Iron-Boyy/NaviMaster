import os
import json
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import ray
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import re
from datasets import load_dataset
from datasets import Dataset as hf_dataset
from PIL import Image
from io import BytesIO

ray.init()

MODEL_PATH = ""

# 推理参数
SAMPLING_PARAMS = SamplingParams(
    temperature=0.0,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=1024,  
    stop_token_ids=[], 
)

DATA_PATH = ""

MICRO_BATCH = 24

def extract_action(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    action_pattern = r"'action':\s*'(\w+)'"
    action_pattern_2 = r'"action":\s*"(\w+)"'
    action_pattern_3 = r'"action_type":\s*"(\w+)"'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        action_match = re.search(action_pattern, content_answer)
        action_match_2 = re.search(action_pattern_2, content_answer)
        action_match_3 = re.search(action_pattern_3, content_answer)
        if action_match:
            return action_match.group(1)
        if action_match_2:
            return action_match_2.group(1)
        if action_match_3:
            return action_match_3.group(1)
    else:
        try:
            action_match = re.search(action_pattern, content)
            action_match_2 = re.search(action_pattern_2, content)
            action_match_3 = re.search(action_pattern_3, content)
            if action_match:
                return action_match.group(1)
            if action_match_2:
                return action_match_2.group(1)
            if action_match_3:
                return action_match_3.group(1)
        except:
            return None
    return None

def extract_input_text(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    action_pattern = r"'input_text':\s*'(.*?)'"
    action_pattern_2 = r'"text":\s*"(.*?)"'
    action_pattern_3 = r'"app_name":\s*"(.*?)"'
    action_pattern_4 = r'"direction":\s*"(.*?)"'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        action_match = re.search(action_pattern, content_answer)
        action_match_2 = re.search(action_pattern_2, content_answer)
        action_match_3 = re.search(action_pattern_3, content_answer)
        action_match_4 = re.search(action_pattern_4, content_answer)
        if action_match:
            return action_match.group(1)
        if action_match_2:
            return action_match_2.group(1)
        if action_match_3:
            return action_match_3.group(1)
        if action_match_4:
            return action_match_4.group(1)
    else:
        try:
            action_match = re.search(action_pattern, content)
            action_match_2 = re.search(action_pattern_2, content)
            action_match_3 = re.search(action_pattern_3, content)
            action_match_4 = re.search(action_pattern_4, content)
            if action_match:
                return action_match.group(1)
            if action_match_2:
                return action_match_2.group(1)
            if action_match_3:
                return action_match_3.group(1)
            if action_match_4:
                return action_match_4.group(1)
        except:
            return "no input text"
    return "no input text"


def extract_coord(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\{.*\[(\d+),\s*(\d+)]\s*.*\}'
    bbox_pattern_2 = [r'"x":\s*(\d+)', r'"y":\s*(\d+)']
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    try:
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            scroll_pattern  = r"\{.*'start_point':\s*\[(\d+),\s*(\d+)\],\s*'end_point':\s*\[(\d+),\s*(\d+)\].*\}"
            scroll_pattern_2  = r'\{.*"start_point":\s*\[(\d+),\s*(\d+)\],\s*"end_point":\s*\[(\d+),\s*(\d+)\].*\}'
            coord_match = re.search(scroll_pattern, content_answer)
            if coord_match:
                start_x, start_y, end_x, end_y = coord_match.groups()
                return [int(start_x), int(start_y), int(end_x), int(end_y)], True
            coord_match = re.search(scroll_pattern_2, content_answer)
            if coord_match:
                start_x, start_y, end_x, end_y = coord_match.groups()
                return [int(start_x), int(start_y), int(end_x), int(end_y)], True
            coord_match = re.search(bbox_pattern, content_answer)
            coord_match_2 = [re.search(bbox_pattern_2[0], content_answer),re.search(bbox_pattern_2[1], content_answer)] 
            if coord_match:
                coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                return coord, True
            if coord_match_2[0] and coord_match_2[1]:
                coord = [int(coord_match_2[0].group(1)), int(coord_match_2[1].group(1))]
                return coord, True
        else:
            try:
                scroll_pattern  = r"\{.*'start_point':\s*\[(\d+),\s*(\d+)\],\s*'end_point':\s*\[(\d+),\s*(\d+)\].*\}"
                scroll_pattern_2  = r'\{.*"start_point":\s*\[(\d+),\s*(\d+)\],\s*"end_point":\s*\[(\d+),\s*(\d+)\].*\}'
                coord_match = re.search(scroll_pattern, content)
                if coord_match:
                    start_x, start_y, end_x, end_y = coord_match.groups()
                    return [int(start_x), int(start_y), int(end_x), int(end_y)], True
                coord_match = re.search(scroll_pattern_2, content)
                if coord_match:
                    start_x, start_y, end_x, end_y = coord_match.groups()
                    return [int(start_x), int(start_y), int(end_x), int(end_y)], True
                coord_match = re.search(bbox_pattern, content)
                coord_match_2 = [re.search(bbox_pattern_2[0], content),re.search(bbox_pattern_2[1], content)] 
                if coord_match:
                    coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                    return coord, True
                if coord_match_2[0] and coord_match_2[1]:
                    coord = [int(coord_match_2[0].group(1)), int(coord_match_2[1].group(1))]
                    return coord, True
            except:
                coord_pattern = r'\{.*\((\d+),\s*(\d+))\s*.*\}'
                coord_match = re.search(coord_pattern, content)
                if coord_match:
                    coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                    return coord, True
        return [0, 0, 0, 0], False
    except:
        return [0, 0, 0, 0], False

class MultiModalDataset(Dataset):
    def __init__(self, data, processor, tokenizer, model_name):
        self.data = data
        self.processor = processor
        self.tokenizer = tokenizer
        self.model_name = model_name

    def __len__(self):
        return len(self.data)
    
    

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample["image"]
        image = Image.open(BytesIO(image['bytes'])).convert("RGB")
        text = sample["instruction"]
        history="None" if 'history' not in sample else sample['history']

        message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": 'test'},
                    ],
                }
            ]

        image_inputs, video_inputs, video_kwargs = process_vision_info(message, return_video_kwargs=True)

        inputs = self.processor(
                    text=["test"],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
        
        resized_height = inputs['image_grid_thw'][0][1] * self.processor.image_processor.patch_size
        resized_width = inputs['image_grid_thw'][0][2] * self.processor.image_processor.patch_size

        inst = "Task:"+text+"\n\nAction history: "+history
        system_text=(
            "A conversation between User and Assistant. The user asks a question, and the Assistant solves it step by step. The assistant "
            "first thinks about the reasoning process in the mind and then provides the user with the answer. "
            "At each step, you will be given the current screenshot and the history of the conversation(include sceenshot and action in each step)."
            "Based on these pieces of information and the goal, you must give the whole content of what you think and then choose to perform one of the action in the following list "
            "(action description followed by the JSON format) by outputing the action in the correct JSON format.\n"
            '- Click/tap on an element on the screen. We have defined the width and height of the screenshot'
            ', use the coordinates (x, y)(x is the pixel from left to right and y is the pixel from top to bottom)'
            ' to indicate which element you want to click, both x and y are integers:'
            ' [{"action": "click", "point": [x(position in horizontal(width)), y(position in vertical(height))]}].\n'
            '- Type text into a text field (this action contains typing in the text and pressing the enter):'
            ' [{"action": "input_text", "text": <text_input>}]\n'
            '- Navigate to last page or last step, if you find current page is not desired: [{"action": "navigate_back"}]\n'
            '- Scroll the screen in one of the four directions, scroll start from the center of the screen to the one of four edges of the screen:'
            ' [{"action": "scroll", "direction": enum["up", "left", "right", "down"] }]\n'
            '- Scroll the screen or a scrollable UI element from start point to end point, use the coordinates (x, y)(x is the pixel from left to right and y is the pixel from top to bottom)'
            ' to indicate the two points you want to scroll, both x and y are integers:'
            ' [{"action": "scroll", "start_point": [<start position in horizontal(width)>, <start position in vertical(height)>], "end_point": [<end position in horizontal(width)>, <end position in vertical(height)>]}]\n'
            '- Open an app, if you find you can not open app in current screenshot, you can use this action to directly open it through its name: [{"action": "open_app", "app_name": <name>}]\n'
            '- Wait for the screen to update: [{"action": "wait"}]\n'
            "NOTICES:1.Coordinates are absolute coordinates (a center point defined by top-left and bottom-right coordinates).\n"
            "2.The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
            '<think> reasoning process here </think><answer>[{"action": "click", "point": [123, 300]}]</answer>\n'
            "Example:\n"
            '[{"action": "wait"}]\n'
            '[{"action": "click", "point": [123, 300]}]\n'
            '[{"action": "scroll", "direction": up}]\n'
            '[{"action": "scroll", "start_point": [123, 300], "end_point": [320, 456]}]\n'
            '[{"action": "input_text", "text": "shanghai shopping mall"}]\n'
            '[{"action": "open_app", "app_name": "Chrome"}]\n'
            '[{"action": "navigate_back"}]\n'
            )
        
        message = [
            {"role": "system", "content": system_text},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "<image>\n"+inst},
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(message, return_video_kwargs=True)

        inputs = self.processor(
                    text=[prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
        
        resized_height = inputs['image_grid_thw'][0][1] * self.processor.image_processor.patch_size
        resized_width = inputs['image_grid_thw'][0][2] * self.processor.image_processor.patch_size
              
        origin_height = image_inputs[0].size[1]
        origin_width = image_inputs[0].size[0]
        scale_x = origin_width / resized_width
        scale_y = origin_height / resized_height

        del inputs

        sample["scale"]=[scale_x.item(),scale_y.item()]
        sample["image_size"]=[origin_width,origin_height]

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        return {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
            "original_sample": sample,
        }


def custom_collate_fn(batch):
    collated_batch = {
        "prompts": [],
        "multi_modal_data": [],
        "mm_processor_kwargs": [],
        "original_samples": [],
    }
    for item in batch:
        collated_batch["prompts"].append(item["prompt"])
        collated_batch["multi_modal_data"].append(item["multi_modal_data"])
        collated_batch["mm_processor_kwargs"].append(item["mm_processor_kwargs"])
        collated_batch["original_samples"].append(item["original_sample"])
    return collated_batch


@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, model_path, sampling_params):
        self.llm = LLM(
            model=model_path,
            limit_mm_per_prompt={"image": 1, "video": 1},
        )
        self.sampling_params = sampling_params

    def process_data(self, dataloader):
        results = []

        for batch in tqdm(dataloader):
            prompts = batch["prompts"]
            multi_modal_data = batch["multi_modal_data"]
            mm_processor_kwargs = batch["mm_processor_kwargs"]
            original_samples = batch["original_samples"]

            llm_inputs = [
                {
                    "prompt": prompt,
                    "multi_modal_data": mm_data,
                    "mm_processor_kwargs": mm_kwargs,
                }
                for prompt, mm_data, mm_kwargs in zip(prompts, multi_modal_data, mm_processor_kwargs)
            ]

            outputs = self.llm.generate(llm_inputs, sampling_params=self.sampling_params)
            
            for original_sample, output in zip(original_samples, outputs):
                generated_text = output.outputs[0].text
                gt_bbox = original_sample["gt_bbox"]
                original_sample["pred"] = generated_text
                pred_coord, _ = extract_coord(generated_text)
                if len(pred_coord) == 4 and pred_coord[0]+pred_coord[1]+pred_coord[2]+pred_coord[3]!=0:
                    original_sample["pred_coord"] = [pred_coord[0]*original_sample["scale"][0],pred_coord[1]*original_sample["scale"][1], pred_coord[2]*original_sample["scale"][0],pred_coord[3]*original_sample["scale"][1]]
                else:
                    original_sample["pred_coord"] = [pred_coord[0]*original_sample["scale"][0],pred_coord[1]*original_sample["scale"][1]]
                pred_action = extract_action(generated_text)
                original_sample["pred_action"] = pred_action
                original_sample["pred_input_text"]=extract_input_text(generated_text)
                original_sample["scale"]=[]
                original_sample["image"]=''
                results.append(original_sample)

        return results






def main(args):
    MODEL_PATH=args.model_path
    DATA_PATH=args.data_path
    if DATA_PATH.endswith('parquet'):
        data=load_dataset("parquet", data_files=DATA_PATH, split="train")
    else:
        data = [json.loads(s) for s in open(DATA_PATH, "r")] if DATA_PATH.endswith(".jsonl") else json.load(open(DATA_PATH,"r"))

    OUTPUT_DIR = args.output_path
    num_actors = args.num_actor
    OUTPUT_DIR = os.path.join(OUTPUT_DIR,MODEL_PATH.split('/')[-1])
    NEW_FILE = os.path.join(OUTPUT_DIR, DATA_PATH.split("/")[-1].replace(".jsonl", "_pred.jsonl").replace('.parquet','.json'))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    data_chunks = [hf_dataset.from_dict(data[i::num_actors]) for i in range(num_actors)]

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    workers = [Worker.remote(MODEL_PATH, SAMPLING_PARAMS) for _ in range(num_actors)]

    futures = []
    for i, chunk in enumerate(data_chunks):
        dataset = MultiModalDataset(chunk, processor, tokenizer, args.model_path.split('/')[-1])
        dataloader = DataLoader(dataset, batch_size=MICRO_BATCH, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
        futures.append(workers[i].process_data.remote(dataloader))

    all_results = ray.get(futures)

    with open(NEW_FILE, "w") as ans_file:
        for worker_results in all_results:
            for sample in worker_results:
                ans_file.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='<model_path>')
    parser.add_argument('--data_path', type=str, default="<data_path>")
    parser.add_argument('--output_path', type=str, default='./outputs')
    parser.add_argument('--num_actor', type=int, default=8)
    args = parser.parse_args()
    main(args)
