Write-Host "=== AgeBooth SDXL LoRA Interpolation Inference ===" -ForegroundColor Green
Write-Host "Continuous Age Transformation (15 -> 75 years)" -ForegroundColor Yellow
Write-Host ""

# Set HuggingFace cache to D: drive to avoid C: drive space issues
$env:HF_HOME = "D:\huggingface_cache"
$env:TRANSFORMERS_CACHE = "D:\huggingface_cache"
$env:DIFFUSERS_CACHE = "D:\huggingface_cache"

# Configuration
$YOUNG_LORA = "models/ageLoRA/young_10_20/checkpoint-200"
$OLD_LORA = "models/ageLoRA/old_70_80/checkpoint-200"
$INPUT_IMAGE = "example_inputs/hinton.jpeg"  # Change to your test image
$OUTPUT_DIR = "outputs/sdxl_interpolation"
$GENDER = "m"  # m or f
$NUM_STEPS = 7  # Number of age steps (15, 25, 35, 45, 55, 65, 75)
$SEED = 42

Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Young LoRA: $YOUNG_LORA"
Write-Host "  Old LoRA: $OLD_LORA"
Write-Host "  Input Image: $INPUT_IMAGE"
Write-Host "  Gender: $GENDER"
Write-Host "  Output: $OUTPUT_DIR"
Write-Host "  Age Steps: $NUM_STEPS"
Write-Host "  Seed: $SEED"
Write-Host ""

# Validation checks
Write-Host "Validating setup..." -ForegroundColor Yellow

if (-not (Test-Path "$YOUNG_LORA/pytorch_lora_weights.safetensors")) {
    Write-Host "❌ Young LoRA not found!" -ForegroundColor Red
    Write-Host "   Expected: $YOUNG_LORA/pytorch_lora_weights.safetensors" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "$OLD_LORA/pytorch_lora_weights.safetensors")) {
    Write-Host "❌ Old LoRA not found!" -ForegroundColor Red
    Write-Host "   Expected: $OLD_LORA/pytorch_lora_weights.safetensors" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $INPUT_IMAGE)) {
    Write-Host "❌ Input image not found!" -ForegroundColor Red
    Write-Host "   Expected: $INPUT_IMAGE" -ForegroundColor Red
    Write-Host ""
    Write-Host "Available images in example_inputs/:" -ForegroundColor Yellow
    Get-ChildItem "example_inputs/" | ForEach-Object { Write-Host "   - $_" -ForegroundColor Gray }
    Write-Host ""
    Write-Host "Please update the INPUT_IMAGE variable in this script." -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ All files found!" -ForegroundColor Green
Write-Host ""

# GPU check
Write-Host "GPU Status:" -ForegroundColor Yellow
python -c "import torch; print(f'  Device: {torch.cuda.get_device_name(0)}'); print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB'); print(f'  CUDA Available: {torch.cuda.is_available()}')"
Write-Host ""

# Dependency check
Write-Host "Checking dependencies..." -ForegroundColor Yellow
$missing = @()
python -c "import safetensors" 2>$null
if ($LASTEXITCODE -ne 0) { $missing += "safetensors" }

if ($missing.Count -gt 0) {
    Write-Host "❌ Missing dependencies: $($missing -join ', ')" -ForegroundColor Red
    Write-Host "Installing..." -ForegroundColor Yellow
    pip install $missing
}

Write-Host "✅ Dependencies OK" -ForegroundColor Green
Write-Host ""

Write-Host "Starting inference in 3 seconds..." -ForegroundColor Green
Write-Host "Estimated time: ~7-10 minutes (7 age steps × ~1 min each)" -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Run inference
python inference_sdxl_interpolation.py `
  --young_lora="$YOUNG_LORA" `
  --old_lora="$OLD_LORA" `
  --image_path="$INPUT_IMAGE" `
  --output_dir="$OUTPUT_DIR" `
  --num_steps=$NUM_STEPS `
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

# Display generated files
if (Test-Path $OUTPUT_DIR) {
    Write-Host "Generated images:" -ForegroundColor Yellow
    Get-ChildItem "$OUTPUT_DIR\*.png" | Sort-Object Name | ForEach-Object {
        Write-Host "  $($_.Name)" -ForegroundColor Gray
    }
    
    Write-Host ""
    Write-Host "Age progression sequence:" -ForegroundColor Yellow
    Write-Host "  Age 15 to Age 75 with interpolation" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "You can now:" -ForegroundColor Yellow
    Write-Host "  1. View images in outputs folder" -ForegroundColor Gray
    Write-Host "  2. Test with different faces" -ForegroundColor Gray
    Write-Host "  3. Adjust NUM_STEPS for more intervals" -ForegroundColor Gray
}
