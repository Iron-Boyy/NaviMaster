MODEL_PATH= # Your model path
MODEL_NAME= #The model name

python ./test_where2place.py --model_path ${MODEL_PATH} --output_path ./result/${MODEL_NAME}_where2place.jsonl
python ./test_refspatial.py --model_path ${MODEL_PATH} --output_path ./result/${MODEL_NAME}_refspatial.jsonl
python ./test_robospatial.py --model_path ${MODEL_PATH} --output_path ./result/${MODEL_NAME}_robospatial.jsonl
python ./test_roboreflt.py --model_path ${MODEL_PATH} --output_path ./result/${MODEL_NAME}_roboreflt.jsonl
