MODEL_NAME= #The model name
DATA_DIR=./outputs/${MODEL_NAME}
python eval/eval_omni.py --model_id ${MODEL_NAME} --prediction_file_path  ${DATA_DIR}/androidcontrol_high_test.json
python eval/eval_omni.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/androidcontrol_low_test.json
