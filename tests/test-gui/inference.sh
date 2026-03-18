MODEL_PATH= #Your model path
DATA_DIR= #Android Control path

python inference/inference_vllm_android.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/androidcontrol_high_test.parquet
python ./inference/inference_vllm_android.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/androidcontrol_low_test.parquet

