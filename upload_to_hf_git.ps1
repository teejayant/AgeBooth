# Upload LoRA models to HuggingFace using Git LFS
# More reliable than API for large files

param(
    [string]$Username = "ShubhamBaghel307",
    [string]$RepoName = "agebooth-loras"
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "AgeBooth LoRA Upload (Git LFS Method)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Git LFS is installed
try {
    git lfs version | Out-Null
    Write-Host "✅ Git LFS is installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Git LFS not found. Installing..." -ForegroundColor Red
    Write-Host "Run: git lfs install" -ForegroundColor Yellow
    exit 1
}

# Create temporary directory for repo
$TempDir = "temp_hf_upload"
if (Test-Path $TempDir) {
    Remove-Item -Recurse -Force $TempDir
}

Write-Host "📁 Creating temporary directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path $TempDir | Out-Null

# Clone or create repository
$RepoUrl = "https://huggingface.co/$Username/$RepoName"
Write-Host ""
Write-Host "🔄 Setting up repository: $RepoUrl" -ForegroundColor Yellow
Write-Host "   You'll be prompted for your HuggingFace token as password" -ForegroundColor Gray
Write-Host ""

Push-Location $TempDir

try {
    # Try to clone existing repo
    git clone $RepoUrl . 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "📦 Repository doesn't exist, creating new one..." -ForegroundColor Yellow
        git init
        git lfs install
        git remote add origin $RepoUrl
    } else {
        Write-Host "✅ Repository cloned" -ForegroundColor Green
    }
    
    # Set up Git LFS for safetensors files
    Write-Host ""
    Write-Host "🔧 Configuring Git LFS for .safetensors files..." -ForegroundColor Yellow
    git lfs track "*.safetensors"
    git add .gitattributes
    
    # Copy LoRA files
    Write-Host ""
    Write-Host "📋 Copying LoRA files..." -ForegroundColor Yellow
    
    $YoungLoraPath = "..\models\ageLoRA\young_10_20\checkpoint-200\pytorch_lora_weights.safetensors"
    $OldLoraPath = "..\models\ageLoRA\old_70_80\checkpoint-200\pytorch_lora_weights.safetensors"
    
    if (Test-Path $YoungLoraPath) {
        Copy-Item $YoungLoraPath "young_lora.safetensors"
        Write-Host "  ✅ young_lora.safetensors copied" -ForegroundColor Green
    } else {
        Write-Host "  ❌ Young LoRA not found at: $YoungLoraPath" -ForegroundColor Red
        throw "Young LoRA file missing"
    }
    
    if (Test-Path $OldLoraPath) {
        Copy-Item $OldLoraPath "old_lora.safetensors"
        Write-Host "  ✅ old_lora.safetensors copied" -ForegroundColor Green
    } else {
        Write-Host "  ❌ Old LoRA not found at: $OldLoraPath" -ForegroundColor Red
        throw "Old LoRA file missing"
    }
    
    # Create README.md
    Write-Host ""
    Write-Host "📝 Creating README.md..." -ForegroundColor Yellow
    
    $ReadmeContent = @"
---
license: apache-2.0
tags:
  - stable-diffusion
  - sdxl
  - lora
  - age-transformation
  - dreambooth
base_model: stabilityai/stable-diffusion-xl-base-1.0
---

# AgeBooth LoRA Models

Two LoRA adapters for age transformation with Stable Diffusion XL.

## Files

- ``young_lora.safetensors``: Young age group (10-20 years) - 20MB
- ``old_lora.safetensors``: Old age group (70-80 years) - 20MB

## Training Details

- **Base Model:** SDXL 1.0
- **Method:** DreamBooth LoRA
- **LoRA Rank:** 4
- **Resolution:** 512x512
- **Steps:** 200 per LoRA
- **Precision:** FP16 mixed precision
- **Dataset:** IMDB-Wiki age-filtered subsets

## Usage

``````python
from diffusers import StableDiffusionXLImg2ImgPipeline
import torch

# Load base model
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

# Load young LoRA
pipe.load_lora_weights("$Username/$RepoName", weight_name="young_lora.safetensors")
young_image = pipe(
    prompt="portrait of a young person",
    image=input_face,
    strength=0.5,
    num_inference_steps=50
).images[0]

# Load old LoRA
pipe.load_lora_weights("$Username/$RepoName", weight_name="old_lora.safetensors")
old_image = pipe(
    prompt="portrait of an elderly person",
    image=input_face,
    strength=0.5,
    num_inference_steps=50
).images[0]
``````

## Linear Interpolation

For intermediate ages, blend the LoRAs:

``````python
from safetensors.torch import load_file
import torch

# Load both LoRAs
young_state = load_file("young_lora.safetensors")
old_state = load_file("old_lora.safetensors")

# Interpolate (alpha=0.5 for middle age)
alpha = 0.5  # 0.0=young, 1.0=old
mixed_state = {
    k: (1 - alpha) * young_state[k] + alpha * old_state[k]
    for k in young_state.keys()
}

# Apply to pipeline
pipe.load_lora_weights(mixed_state)
``````

## Performance

- **Inference Time:** ~4-5 sec/step on RTX 4050
- **VRAM Usage:** ~5.5GB with SDXL
- **Quality:** Best results with 50+ inference steps
- **Strength:** 0.45-0.5 for identity preservation

## Dataset

Trained on age-filtered IMDB-Wiki dataset:
- **Young:** 25 images (ages 10-20)
- **Old:** 25 images (ages 70-80)
- **Ethnicity:** South Asian focused

## Try It!

🎨 **Demo:** [AgeBooth Spaces App](https://huggingface.co/spaces/$Username/agebooth-age-transformation)

## Citation

``````bibtex
@misc{agebooth2025,
  title={AgeBooth: Identity-Preserved Age Transformation},
  author={Baghel, Shubham},
  year={2025}
}
``````

## License

Apache 2.0 License
"@
    
    Set-Content -Path "README.md" -Value $ReadmeContent
    Write-Host "  ✅ README.md created" -ForegroundColor Green
    
    # Add all files
    Write-Host ""
    Write-Host "➕ Adding files to Git..." -ForegroundColor Yellow
    git add .gitattributes
    git add young_lora.safetensors
    git add old_lora.safetensors
    git add README.md
    
    # Commit
    Write-Host ""
    Write-Host "💾 Committing changes..." -ForegroundColor Yellow
    git commit -m "Upload AgeBooth LoRA models and documentation"
    
    # Push
    Write-Host ""
    Write-Host "🚀 Pushing to HuggingFace Hub..." -ForegroundColor Yellow
    Write-Host "   Enter your HuggingFace token when prompted for password" -ForegroundColor Gray
    Write-Host ""
    
    git push origin main
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Green
        Write-Host "🎉 Upload successful!" -ForegroundColor Green
        Write-Host "============================================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "View your models at:" -ForegroundColor Cyan
        Write-Host "  $RepoUrl" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Load in code with:" -ForegroundColor Cyan
        Write-Host "  pipe.load_lora_weights('$Username/$RepoName', weight_name='young_lora.safetensors')" -ForegroundColor Yellow
        Write-Host ""
    } else {
        Write-Host ""
        Write-Host "❌ Push failed. Check your token and try again." -ForegroundColor Red
    }
    
} catch {
    Write-Host ""
    Write-Host "❌ Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "1. Check your HuggingFace token has write access" -ForegroundColor Gray
    Write-Host "2. Verify Git LFS is installed: git lfs version" -ForegroundColor Gray
    Write-Host "3. Try again in a few minutes if HuggingFace API is down" -ForegroundColor Gray
    Write-Host "4. Manually create repo at: https://huggingface.co/new" -ForegroundColor Gray
} finally {
    Pop-Location
}

# Cleanup
Write-Host ""
$Cleanup = Read-Host "Delete temporary directory? (y/n)"
if ($Cleanup -eq 'y') {
    Remove-Item -Recurse -Force $TempDir
    Write-Host "🧹 Cleanup complete" -ForegroundColor Green
} else {
    Write-Host "📁 Temporary files kept in: $TempDir" -ForegroundColor Yellow
}
