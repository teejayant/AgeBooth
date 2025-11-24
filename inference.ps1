Write-Host "=== AgeBooth InfiniteYou Inference ===" -ForegroundColor Green
Write-Host "Continuous Age Transformation with LoRA Interpolation" -ForegroundColor Yellow
Write-Host ""

# Configuration
$METHOD = "direct_linear"  # Interpolation method: direct_linear or svdmix
$YOUNG_LORA = "models/ageLoRA/young_10_20/checkpoint-200"
$OLD_LORA = "models/ageLoRA/old_70_80/checkpoint-200"
$IMAGE_PATH = "example_inputs/test_face.jpg"  # Input face image
$SEX = "m"  # Gender: m or f
$ID_SCALE = 0.8  # Identity preservation strength (0.0-1.0)
$OUTPUT_DIR = "outputs/infiniu_results"

Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Method: $METHOD"
Write-Host "  Young LoRA: $YOUNG_LORA"
Write-Host "  Old LoRA: $OLD_LORA"
Write-Host "  Input Image: $IMAGE_PATH"
Write-Host "  Gender: $SEX"
Write-Host "  ID Scale: $ID_SCALE"
Write-Host "  Output: $OUTPUT_DIR"
Write-Host ""

# Check if LoRAs exist
if (-not (Test-Path "$YOUNG_LORA/pytorch_lora_weights.safetensors")) {
    Write-Host "❌ Young LoRA not found: $YOUNG_LORA" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "$OLD_LORA/pytorch_lora_weights.safetensors")) {
    Write-Host "❌ Old LoRA not found: $OLD_LORA" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $IMAGE_PATH)) {
    Write-Host "❌ Input image not found: $IMAGE_PATH" -ForegroundColor Red
    Write-Host "Please add a test face image to example_inputs/" -ForegroundColor Yellow
    exit 1
}

Write-Host "GPU Verification:" -ForegroundColor Yellow
python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}'); print(f'  CUDA: {torch.cuda.is_available()}')"
Write-Host ""

Write-Host "Starting inference in 3 seconds..." -ForegroundColor Green
Start-Sleep -Seconds 3

# Run inference
python inference_lora_interp_traverse_infU.py `
  --method="$METHOD" `
  --young_lora="$YOUNG_LORA" `
  --old_lora="$OLD_LORA" `
  --image_path="$IMAGE_PATH" `
  --sex="$SEX" `
  --id_scale=$ID_SCALE `
  --output_dir="$OUTPUT_DIR" `
  --num_interpolations=7 `
  --resolution=512 `
  --guidance_scale=7.5 `
  --num_inference_steps=50

Write-Host ""
Write-Host "=== Inference Complete! ===" -ForegroundColor Green
Write-Host "Results saved to: $OUTPUT_DIR" -ForegroundColor Cyan
Write-Host ""
Write-Host "Check the age progression images:" -ForegroundColor Yellow
Write-Host "  Age 15 (Young LoRA)"
Write-Host "  Age 25, 35, 45, 55, 65 (Interpolated)"
Write-Host "  Age 75 (Old LoRA)"