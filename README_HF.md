---
title: AgeBooth - AI Age Transformation
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.19.0
app_file: app.py
pinned: false
license: apache-2.0
tags:
  - stable-diffusion
  - sdxl
  - age-transformation
  - lora
  - dreambooth
  - image-to-image
---

# AgeBooth: AI Age Transformation 🎭

Transform faces across ages (15-75 years) using LoRA interpolation and Stable Diffusion XL.

## Features

- 🔄 **Age Transformation:** Generate realistic age progressions and regressions
- 🎨 **Customizable Prompts:** Fine-tune outputs with custom prompts
- 🧬 **Identity Preservation:** Maintains facial features across transformations
- ⚡ **Fast Inference:** Optimized for 6GB+ VRAM GPUs
- 📊 **Comparison Grid:** See all ages side-by-side

## How It Works

1. **LoRA Training:** Two LoRA adapters trained on young (10-20) and old (70-80) age groups
2. **Linear Interpolation:** Smooth blending between young and old LoRAs
3. **SDXL Generation:** High-quality image synthesis with identity preservation

## Model Details

- **Base Model:** Stable Diffusion XL (SDXL)
- **Training Method:** DreamBooth LoRA
- **LoRA Rank:** 4
- **Training Steps:** 200 per age group
- **Dataset:** Age-specific face images from IMDB-Wiki

## Usage

```python
from diffusers import StableDiffusionXLImg2ImgPipeline
import torch

# Load pipeline
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

# Load LoRA
pipe.load_lora_weights("ShubhamBaghel309/agebooth-loras", weight_name="young_lora.safetensors")

# Generate
output = pipe(prompt="portrait of a young person", image=input_face).images[0]
```

## Technical Implementation

Based on the AgeBooth research paper:
- **Paper:** AgeBooth: Identity-Preserved Age Transformation
- **Method:** Training-free LoRA and Prompt Fusion
- **Equation:** ∆θ_fused = α·∆θ_young + (1-α)·∆θ_old

## Performance

- **Inference Time:** ~5-7 minutes for 7 age steps
- **GPU Memory:** ~5.5GB VRAM
- **Quality:** 512x512 resolution, 50 inference steps

## Limitations

- Works best with frontal face images
- Identity preservation depends on transformation strength
- May struggle with extreme poses or lighting
- Optimized for South Asian features (training data)

## Citation

If you use this work, please cite:

```bibtex
@misc{agebooth2025,
  title={AgeBooth: Identity-Preserved Age Transformation},
  author={Baghel, Shubham},
  year={2025}
}
```

## License

Apache 2.0 License

---

**Made with ❤️ using Stable Diffusion XL, Diffusers, and Gradio**
