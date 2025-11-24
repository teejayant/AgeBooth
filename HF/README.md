---
title: AgeBooth - AI Age Transformation
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.19.0"
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

### 1. Train LoRA

Use the script **`train_dreambooth_lora_sdxl.sh` / `train_dreambooth_lora_flux.sh`** to train a LoRA for a specific age range.
You only need to set:

- `MODEL_NAME` → base model path (e.g., `RunDiffusion/Juggernaut-XL-v9`)
- `OUTPUT_DIR` → path where the LoRA will be saved
- `INSTANCE_DIR` → dataset location for the target age group

👉 Train one LoRA for the **young age group** (e.g., 10–20) and another for the **old age group** (e.g., 70–80).

---

### 2. Run Inference

Once both LoRAs are trained, use **`inference_lora_interp_traverse_pulid.sh` / `inference_lora_interp_traverse_infU.sh`** for PuLID/InfiniteYou inference.
You need to specify:

- `METHOD` → interpolation method (`direct_linear` or `svdmix`)
- `YOUNG_LORA` → checkpoint path of the trained young LoRA
- `OLD_LORA` → checkpoint path of the trained old LoRA
- `IMAGE_PATH` → input image for identity preservation
- `SEX` → gender of the subject (`m` or `f`)

To control the strength of identity injection, you can adjust the `ID_SCALE` parameter. You can find some pre-downloaded identity images in the `example_inputs` folder.

The outputs will be saved in the `outputs/` directory, organized by method and parameters.


## 🚩TODO

- [x] training code for SDXL age LoRA
- [x] training code for FLUX age LoRA
- [x] AgeBooth inference for PuLID
- [x] AgeBooth inference for InfiniteYou
- [ ] AgeBooth inference for InstantID
- [ ] Pretrained age LoRAs

## 📄Citation
If you find this AgeBooth useful for your research or applications, please cite using this BibTeX:

```BibTeX
@misc{zhu2025ageboothcontrollablefacialaging,
      title={AgeBooth: Controllable Facial Aging and Rejuvenation via Diffusion Models}, 
      author={Shihao Zhu and Bohan Cao and Ziheng Ouyang and Zhen Li and Peng-Tao Jiang and Qibin Hou},
      year={2025},
      eprint={2510.05715},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.05715}, 
}
```
