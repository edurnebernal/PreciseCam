export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="./preciseCam_1024_bs1_g32_lr1e-5_70k_1GPU"
export DATASET="./dataset"
export VAE="madebyollin/sdxl-vae-fp16-fix"
export NOW=$( date '+%F_%H:%M:%S' )

export BATCH_SIZE=1
export LEARNING_RATE=1e-5
export RESOLUTION=1024
export STEPS=70000
export GRADIENT=32

accelerate config default

accelerate launch ./diffusers-adapted/train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --resume_from_checkpoint="latest" \
 --logging_dir=$NOW \
 --proportion_empty_prompts=0.5 \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=$DATASET \
 --mixed_precision="fp16" \
 --resolution=$RESOLUTION \
 --learning_rate=$LEARNING_RATE \
 --max_train_steps=$STEPS \
 --checkpointing_steps=1000 \
 --checkpoints_total_limit=5 \
 --train_batch_size=$BATCH_SIZE \
 --gradient_accumulation_steps=$GRADIENT \
 --pretrained_vae_model_name_or_path=$VAE \
 --gradient_checkpointing \
 --set_grads_to_none \
 --seed=13 