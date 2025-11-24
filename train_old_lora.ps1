# Training script for Old LoRA (70-80 years)
# Optimized for RTX 4050 6GB GPU
Write-Host "=== Old LoRA Training (Age 70-80) - FIXED ===" -ForegroundColor Green
Write-Host "Optimized for RTX 4050 6GB VRAM" -ForegroundColor Yellow
Write-Host ""

# Force GPU usage
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"

# Configuration
$MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
$OUTPUT_DIR = "models/ageLoRA/old_70_80"
$INSTANCE_DIR = "dataset_small/training/old_70_80"
$PROMPT = "A person at old age"

Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Model: $MODEL_NAME"
Write-Host "  Dataset: $INSTANCE_DIR"
Write-Host "  Output: $OUTPUT_DIR"
Write-Host ""

# GPU Check
Write-Host "GPU Verification:" -ForegroundColor Yellow
python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}'); print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB'); print(f'  CUDA: {torch.cuda.is_available()}')"
Write-Host ""

Write-Host "Starting training in 3 seconds..." -ForegroundColor Green
Write-Host "Close all other apps to save RAM!" -ForegroundColor Red
Start-Sleep -Seconds 3

accelerate launch `
  --machine_rank=0 `
  --num_machines=1 `
  --main_process_port=11131 `
  --num_processes=1 `
  --mixed_precision="fp16" `
  --gpu_ids="0" `
  train_dreambooth_lora_sdxl.py `
  --pretrained_model_name_or_path="$MODEL_NAME" `
  --instance_data_dir="$INSTANCE_DIR" `
  --output_dir="$OUTPUT_DIR" `
  --instance_prompt="$PROMPT" `
  --rank=4 `
  --resolution=512 `
  --train_batch_size=1 `
  --gradient_accumulation_steps=1 `
  --learning_rate=1e-4 `
  --lr_scheduler="constant" `
  --lr_warmup_steps=0 `
  --max_train_steps=200 `
  --checkpointing_steps=100 `
  --seed=42 `
  --gradient_checkpointing `
  --mixed_precision="fp16"

Write-Host ""
Write-Host "=== Training Complete! ===" -ForegroundColor Green
Write-Host "Model saved to: $OUTPUT_DIR" -ForegroundColor Cyan