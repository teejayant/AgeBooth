accelerate launch \
  --mixed_precision="fp16" \
  --machine_rank 0 \
  --num_machines 1 \
  --main_process_port 11131 \
  --num_processes 1 \
  train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"  \
  --instance_data_dir="data/10-20" \
  --output_dir="models/ageLoRA/jgr/10_20_b1" \
  --instance_prompt="A person in sbu age." \
  --rank=16 \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=5e-5 \
  --report_to="tensorboard" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --validation_prompt="A person at the age of sbu, portrait, in white background" \
  --validation_epochs=200 \
  --seed="0" \
  --gradient_checkpointing \
  --variant="fp16" \
  --checkpointing_steps=200 \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention
  # -push_to_hub \
