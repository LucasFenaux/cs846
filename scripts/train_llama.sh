echo "GPU $1"

export CUDA_VISIBLE_DEVICES="$1"
export WANDB_PROJECT="CS846"
python train_llama.py ./config/hf_config_llama.json