Write-Host "=== AgeBooth LoRA Interpolation Inference ===" -ForegroundColor Green
Write-Host "Implementing Paper Methods: Linear + SVDMix" -ForegroundColor Yellow
Write-Host ""

# Set HuggingFace cache to D: drive
$env:HF_HOME = "D:\huggingface_cache"

# Configuration
$YOUNG_LORA = "models/ageLoRA/young_10_20/checkpoint-200"
$OLD_LORA = "models/ageLoRA/old_70_80/checkpoint-200"
$INPUT_IMAGE = "example_inputs/hinton.jpeg"
$GENDER = "m"
$NUM_STEPS = 7
$SEED = 42

Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Young LoRA: $YOUNG_LORA"
Write-Host "  Old LoRA: $OLD_LORA"
Write-Host "  Input Image: $INPUT_IMAGE"
Write-Host "  Gender: $GENDER"
Write-Host "  Interpolation Steps: $NUM_STEPS"
Write-Host ""

# Validation
if (-not (Test-Path "$YOUNG_LORA/pytorch_lora_weights.safetensors")) {
    Write-Host "ERROR: Young LoRA not found!" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "$OLD_LORA/pytorch_lora_weights.safetensors")) {
    Write-Host "ERROR: Old LoRA not found!" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $INPUT_IMAGE)) {
    Write-Host "ERROR: Input image not found!" -ForegroundColor Red
    exit 1
}

Write-Host "All files validated" -ForegroundColor Green
Write-Host ""

# Ask user which method to run
Write-Host "Available interpolation methods:" -ForegroundColor Yellow
Write-Host "  1. LINEAR   - Naive linear interpolation (Eq. 4)" -ForegroundColor Gray
Write-Host "  2. SVDMIX   - SVD-based fusion (Eq. 9-11, RECOMMENDED)" -ForegroundColor Gray
Write-Host "  3. BOTH     - Run both methods for comparison" -ForegroundColor Gray
Write-Host ""
$choice = Read-Host "Select method (1/2/3)"

$methods = @()
switch ($choice) {
    "1" { $methods = @("linear") }
    "2" { $methods = @("svdmix") }
    "3" { $methods = @("linear", "svdmix") }
    default { 
        Write-Host "Invalid choice, defaulting to SVDMix" -ForegroundColor Yellow
        $methods = @("svdmix")
    }
}

Write-Host ""
Write-Host "GPU Status:" -ForegroundColor Yellow
python -c "import torch; print(f'  Device: {torch.cuda.get_device_name(0)}'); print(f'  CUDA: {torch.cuda.is_available()}')"
Write-Host ""

foreach ($method in $methods) {
    $OUTPUT_DIR = "outputs/interpolation_$method"
    
    Write-Host "======================================" -ForegroundColor Cyan
    Write-Host "Running: $($method.ToUpper())" -ForegroundColor Cyan
    Write-Host "======================================" -ForegroundColor Cyan
    Write-Host "Output: $OUTPUT_DIR" -ForegroundColor Gray
    Write-Host "Estimated time: ~7-10 minutes" -ForegroundColor Yellow
    Write-Host ""
    
    python inference_interpolation.py `
      --young_lora="$YOUNG_LORA" `
      --old_lora="$OLD_LORA" `
      --image_path="$INPUT_IMAGE" `
      --output_dir="$OUTPUT_DIR" `
      --method="$method" `
      --num_steps=$NUM_STEPS `
      --resolution=512 `
      --guidance_scale=7.5 `
      --num_inference_steps=50 `
      --strength=0.75 `
      --seed=$SEED `
      --sex="$GENDER"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "SUCCESS: $($method.ToUpper()) completed!" -ForegroundColor Green
        Write-Host ""
    } else {
        Write-Host ""
        Write-Host "ERROR: $($method.ToUpper()) failed!" -ForegroundColor Red
        Write-Host ""
    }
}

Write-Host "======================================" -ForegroundColor Green
Write-Host "All Methods Complete!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "Results:" -ForegroundColor Yellow

foreach ($method in $methods) {
    $OUTPUT_DIR = "outputs/interpolation_$method"
    if (Test-Path $OUTPUT_DIR) {
        Write-Host ""
        Write-Host "  $($method.ToUpper()):" -ForegroundColor Cyan
        $imageCount = (Get-ChildItem "$OUTPUT_DIR\*.png" | Measure-Object).Count
        Write-Host "    Generated: $imageCount images" -ForegroundColor Gray
        Write-Host "    Location: $OUTPUT_DIR" -ForegroundColor Gray
        
        if (Test-Path "$OUTPUT_DIR\comparison_grid_$method.png") {
            Write-Host "    Comparison: comparison_grid_$method.png" -ForegroundColor Gray
        }
    }
}

Write-Host ""
Write-Host "Compare the methods:" -ForegroundColor Yellow
Write-Host "  LINEAR: Simple interpolation, may have quality issues" -ForegroundColor Gray
Write-Host "  SVDMIX: SVD-based fusion, better quality (recommended in paper)" -ForegroundColor Gray
