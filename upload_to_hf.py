"""
Upload trained LoRA models to HuggingFace Hub
"""
from huggingface_hub import HfApi, create_repo
import os
from pathlib import Path

# Configuration
HF_USERNAME = input("Enter your HuggingFace username: ")
REPO_NAME = "agebooth-loras"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

# Local LoRA paths
YOUNG_LORA_PATH = "models/ageLoRA/young_10_20/checkpoint-200/pytorch_lora_weights.safetensors"
OLD_LORA_PATH = "models/ageLoRA/old_70_80/checkpoint-200/pytorch_lora_weights.safetensors"

def upload_loras():
    """Upload LoRA weights to HuggingFace Hub"""
    
    # Initialize HuggingFace API
    api = HfApi()
    
    # Get token from environment or prompt
    token = os.getenv("HF_TOKEN")
    if not token:
        token = input("Enter your HuggingFace token (with write access): ")
    
    print(f"\n📦 Creating repository: {REPO_ID}")
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="model",
            token=token,
            exist_ok=True
        )
        print("✅ Repository created/verified")
    except Exception as e:
        print(f"⚠️ Repository creation: {e}")
    
    # Upload young LoRA
    print(f"\n⬆️ Uploading young LoRA...")
    try:
        api.upload_file(
            path_or_fileobj=YOUNG_LORA_PATH,
            path_in_repo="young_lora.safetensors",
            repo_id=REPO_ID,
            repo_type="model",
            token=token
        )
        print("✅ Young LoRA uploaded")
    except Exception as e:
        print(f"❌ Young LoRA upload failed: {e}")
    
    # Upload old LoRA
    print(f"\n⬆️ Uploading old LoRA...")
    try:
        api.upload_file(
            path_or_fileobj=OLD_LORA_PATH,
            path_in_repo="old_lora.safetensors",
            repo_id=REPO_ID,
            repo_type="model",
            token=token
        )
        print("✅ Old LoRA uploaded")
    except Exception as e:
        print(f"❌ Old LoRA upload failed: {e}")
    
    # Create model card
    model_card = f"""---
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

- `young_lora.safetensors`: Young age group (10-20 years)
- `old_lora.safetensors`: Old age group (70-80 years)

## Training Details

- **Base Model:** SDXL 1.0
- **Method:** DreamBooth LoRA
- **LoRA Rank:** 4
- **Resolution:** 512x512
- **Steps:** 200 per LoRA
- **Precision:** FP16 mixed precision

## Usage

```python
from diffusers import StableDiffusionXLImg2ImgPipeline
import torch

# Load base model
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

# Load young LoRA
pipe.load_lora_weights("{REPO_ID}", weight_name="young_lora.safetensors")
young_image = pipe(prompt="young person", image=input_face).images[0]

# Load old LoRA
pipe.load_lora_weights("{REPO_ID}", weight_name="old_lora.safetensors")
old_image = pipe(prompt="elderly person", image=input_face).images[0]
```

## Linear Interpolation

For intermediate ages, blend the LoRAs:

```python
# Load both LoRAs
young_state = torch.load("young_lora.safetensors")
old_state = torch.load("old_lora.safetensors")

# Interpolate (alpha=0.5 for middle age)
alpha = 0.5
mixed_state = {{
    k: alpha * young_state[k] + (1 - alpha) * old_state[k]
    for k in young_state.keys()
}}
```

## Dataset

Trained on age-filtered subsets of IMDB-Wiki dataset:
- Young: 25 images (ages 10-20)
- Old: 25 images (ages 70-80)

## Performance

- **Inference Time:** ~4-5 sec/step on RTX 4050
- **VRAM Usage:** ~5.5GB
- **Quality:** Best with 50+ inference steps

## Citation

```bibtex
@misc{{agebooth2025,
  title={{AgeBooth: Identity-Preserved Age Transformation}},
  author={{Baghel, Shubham}},
  year={{2025}}
}}
```
"""
    
    print(f"\n📝 Uploading model card...")
    try:
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model",
            token=token
        )
        print("✅ Model card uploaded")
    except Exception as e:
        print(f"❌ Model card upload failed: {e}")
    
    print(f"\n🎉 Upload complete! View at: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    # Verify files exist
    if not Path(YOUNG_LORA_PATH).exists():
        print(f"❌ Young LoRA not found at: {YOUNG_LORA_PATH}")
        exit(1)
    
    if not Path(OLD_LORA_PATH).exists():
        print(f"❌ Old LoRA not found at: {OLD_LORA_PATH}")
        exit(1)
    
    print("=" * 60)
    print("AgeBooth LoRA Upload Script")
    print("=" * 60)
    
    upload_loras()
