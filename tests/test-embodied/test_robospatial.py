import json, os, io, base64
import requests
import numpy as np
from PIL import Image
import logging

JSONL_FILE_PATH = "./data/RoboSpatial/annotations.json"

def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def read_json_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        content = json.load(file)
    return content

def _encode_image(image):
    pil_image = Image.fromarray(image[:, :, :3])
    
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()
    
    return base64.b64encode(img_bytes).decode('utf-8')
def send_request_to_model(task, image, processor, llm):
    image = Image.open(image).convert("RGB")
    message = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": task},
            ],
        }
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=1024
    )
    prompt = processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
    image_inputs, video_inputs, video_kwargs = process_vision_info(message, return_video_kwargs=True)
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    llm_input=[{
        "prompt": prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }]

    outputs = llm.generate(llm_input, sampling_params)
    response = outputs[0].outputs[0].text

    return response

def main():
    MODEL_PATH = args.model_path
    llm = LLM(model=MODEL_PATH, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    data = read_json_file(JSONL_FILE_PATH)
    
    results = []
    for item in data:
        task = item.get("question").replace("Your answer should be formatted as a list of tuples, i.e. [(x1, y1), ...], where each tuple contains the x and y coordinates of several points", "Your answer should be formatted as a list, i.e. [x, y], where it contains the x and y coordinates of a point")
        if item.get('mask') == None:
            continue
        task = (
            f'{task}\n'
            "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
            '<think> ... </think> <answer>[{"action": "moveto", "point": [x, y]}]</answer>\n'
            "Example:\n"
            '[{"action": "moveto", "point": [123, 300]}]\n'
        )
        image = os.path.join("./data/RoboSpatial",item.get("img"))
        result = send_request_to_model(task, image, processor, llm)
        if result:
            results.append(result)
    with open(args.output_path, 'w', encoding='utf-8') as output_file:
        for result in results:
            json.dump({"result": result}, output_file)
            output_file.write('\n')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='<model_path>')
    parser.add_argument('--output_path', type=str, default='./outputs')
    args = parser.parse_args()
    main(args)