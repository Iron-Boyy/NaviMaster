from PIL import Image
from tqdm import tqdm
import argparse
import json
import numpy as np
import re


def text2pts(text, width=640, height=480):
    pattern = r'\{.*\[(\d+),\s*(\d+)]\s*.*\}'
    matches = re.findall(pattern, text)
    points = []
    for match in matches:
        vector = [
            int(num) if '.' in num else int(num) for num in match
        ]
        if len(vector) == 2:
            x, y = vector
            points.append((x, y))
        elif len(vector) == 4:
            x0, y0, x1, y1 = vector
            if isinstance(x0, float):
                x0 = int(x0 * width)
                y0 = int(y0 * height)
                x1 = int(x1 * width)
                y1 = int(y1 * height)
            mask = np.zeros((height, width), dtype=bool)
            mask[y0:y1, x0:x1] = 1
            y, x = np.where(mask)
            points.extend(list(np.stack([x, y], axis=1)))
    return np.array(points)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer_file", default="/mnt/shared-storage-user/luozhihao/codebase/code/RoboPoint/result/navimaster_guiodyssey_7000+7000_embodied_81_robospatial.jsonl")
    parser.add_argument("--data_dir", default="/mnt/shared-storage-user/luozhihao/codebase/dataset/RoboSpatial")
    parser.add_argument("--question_file", default="/mnt/shared-storage-user/luozhihao/codebase/dataset/RoboSpatial/annotations.json")
    parser.add_argument("--num_questions", type=int)
    args = parser.parse_args()

    with open(f"{args.question_file}", 'r') as file:
        questions = json.load(file)
    if args.num_questions is None:
        args.num_questions = len(questions)
    with open(args.answer_file, 'r') as file:
        answers = [json.loads(line) for line in file]

    accuracy = []
    for idx, question in enumerate(questions[:args.num_questions]):
        if question['mask'] is None:
            continue
        points = text2pts(answers[idx]['result'])
        mask = np.array(Image.open(f"{args.data_dir}/{question['mask']}")) / 255.
        acc = 0
        if len(points) > 0:
            in_range = (points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) \
                     & (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0])
            acc = np.concatenate([
                mask[points[in_range, 1], points[in_range, 0]],
                np.zeros(points.shape[0] - in_range.sum())
            ]).mean()
        accuracy.append(acc)
    print(f"RoboSpitial Accuracy: {np.mean(accuracy)}")
