echo "GPU $1"

export CUDA_VISIBLE_DEVICES="$1"
export WANDB_PROJECT="CS846"
python train.py ./config/hf_config.json