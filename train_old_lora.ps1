# Training script for Old LoRA (70-80 years)
# Optimized for RTX 4050 6GB GPU

$env:MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
$env:OUTPUT_DIR = "models/ageLoRA/old_70_80"
$env:INSTANCE_DIR = "dataset_small/training/old_70_80"
$env:VALIDATION_DIR = "dataset_small/validation/old_70_80"

$BATCH_SIZE = 1
$env:PROMPT = "A person at old age"
$env:VALID_PROMPT = "A person at the age of 75, portrait, high quality"

Write-Host "Starting Old LoRA Training (Age 70-80)..." -ForegroundColor Green
Write-Host "This will take approximately 20-30 minutes on RTX 4050" -ForegroundColor Yellow

$MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
$INSTANCE_DIR = "dataset_small/training/old_70_80"
$OUTPUT_DIR = "models/ageLoRA/old_70_80"
$INSTANCE_PROMPT = "A person at old age"

accelerate launch train_dreambooth_lora_sdxl.py `
  --pretrained_model_name_or_path="$MODEL_NAME" `
  --instance_data_dir="$INSTANCE_DIR" `
  --output_dir="$OUTPUT_DIR" `
  --instance_prompt="$INSTANCE_PROMPT" `
  --resolution=512 `
  --train_batch_size=1 `
  --gradient_accumulation_steps=4 `
  --learning_rate=1e-4 `
  --lr_scheduler="constant" `
  --lr_warmup_steps=0 `
  --max_train_steps=400 `
  --rank=16 `
  --validation_prompt="A person at the age of 75, portrait, high quality" `
  --validation_epochs=100 `
  --checkpointing_steps=100 `
  --seed=42 `
  --gradient_checkpointing `
  --use_8bit_adam `
  --mixed_precision="fp16" `
  --report_to="tensorboard"

Write-Host "`nOld LoRA training completed!" -ForegroundColor Green
Write-Host "Model saved to: $OUTPUT_DIR" -ForegroundColor Cyan
