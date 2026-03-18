#!/usr/bin/env bash
MODEL_NAME= #Your model name 

python ./eval/summarize_vqa_where2place.py --answer_file ./result/${MODEL_NAME}_where2place.jsonl
python ./eval/summarize_vqa_referspatial.py --answer_file ./result/${MODEL_NAME}_refspatial.jsonl
python ./eval/summarize_vqa_robospatial.py --answer_file ./result/${MODEL_NAME}_robospatial.jsonl
python ./eval/summarize_vqa_roboreflt.py --answer_file ./result/${MODEL_NAME}_roboreflt.jsonl
