export WANDB_MODE=offline
set -x

MODEL_PATH=/mnt/shared-storage-user/luozhihao/codebase/huoshan/models/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

SYSTEM_PROMPT=""""""

python3 -m verl.trainer.main \
    config=scripts/config.yaml \
    data.train_files=/mnt/shared-storage-user/luozhihao/codebase/huoshan/dataset/robopoint/train_myagent_guiodyssey10000+10000 \
    data.val_files=/mnt/shared-storage-user/luozhihao/evolveai_oss/luozhihao/huoshanoss/dataset/GUI-R1/test.parquet \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    worker.reward.compute_score=agent \
    trainer.experiment_name=navimaster_no_delete_val_4 \
    trainer.n_gpus_per_node=8 \
    trainer.total_episodes=3 \
    data.max_pixels=1258291 \
    data.max_prompt_length=7000 \
    data.max_response_length=1024 \
    data.val_batch_size=256 \
