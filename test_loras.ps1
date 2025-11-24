Write-Host "=== AgeBooth SDXL Simple Inference ===" -ForegroundColor Green
Write-Host "Testing Young and Old LoRAs separately" -ForegroundColor Yellow
Write-Host ""

# Set HuggingFace cache to D: drive
$env:HF_HOME = "D:\huggingface_cache"
$env:TRANSFORMERS_CACHE = "D:\huggingface_cache"
$env:DIFFUSERS_CACHE = "D:\huggingface_cache"

# Configuration
$YOUNG_LORA = "models/ageLoRA/young_10_20/checkpoint-200"
$OLD_LORA = "models/ageLoRA/old_70_80/checkpoint-200"
$INPUT_IMAGE = "example_inputs/hinton.jpeg"
$OUTPUT_DIR = "outputs/simple_test"
$GENDER = "m"
$SEED = 42

Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Young LoRA: $YOUNG_LORA"
Write-Host "  Old LoRA: $OLD_LORA"
Write-Host "  Input Image: $INPUT_IMAGE"
Write-Host "  Gender: $GENDER"
Write-Host "  Output: $OUTPUT_DIR"
Write-Host ""

# Validation
if (-not (Test-Path "$YOUNG_LORA/pytorch_lora_weights.safetensors")) {
    Write-Host "❌ Young LoRA not found!" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "$OLD_LORA/pytorch_lora_weights.safetensors")) {
    Write-Host "❌ Old LoRA not found!" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $INPUT_IMAGE)) {
    Write-Host "❌ Input image not found!" -ForegroundColor Red
    exit 1
}

Write-Host "✓ All files validated" -ForegroundColor Green
Write-Host ""

# GPU check
Write-Host "GPU Status:" -ForegroundColor Yellow
python -c "import torch; print(f'  Device: {torch.cuda.get_device_name(0)}'); print(f'  CUDA Available: {torch.cuda.is_available()}')"
Write-Host ""

Write-Host "Starting simple inference..." -ForegroundColor Green
Write-Host "This will generate 2 images (young + old) without interpolation" -ForegroundColor Yellow
Write-Host "Estimated time: ~3-4 minutes" -ForegroundColor Yellow
Write-Host ""

# Run inference
python inference_simple.py `
  --young_lora="$YOUNG_LORA" `
  --old_lora="$OLD_LORA" `
  --image_path="$INPUT_IMAGE" `
  --output_dir="$OUTPUT_DIR" `
  --resolution=512 `
  --guidance_scale=7.5 `
  --num_inference_steps=50 `
  --strength=0.75 `
  --seed=$SEED `
  --sex="$GENDER"

Write-Host ""
Write-Host "=== Inference Complete! ===" -ForegroundColor Green
Write-Host "Results saved to: $OUTPUT_DIR" -ForegroundColor Cyan
Write-Host ""

if (Test-Path $OUTPUT_DIR) {
    Write-Host "Generated files:" -ForegroundColor Yellow
    Get-ChildItem "$OUTPUT_DIR\*.png" | Sort-Object Name | ForEach-Object {
        Write-Host "  $($_.Name)" -ForegroundColor Gray
    }
}
