
# ChronoFlux: Controllable Facial Aging & Rejuvenation on SDXL

ChronoFlux is a practitioner-friendly re-implementation of the AgeBooth workflow that shows how to:

- curate age-specific datasets from IMDB-Wiki,
- fine-tune SDXL with DreamBooth-style LoRA adapters for young (10–20) and old (70–80) cohorts,
- interpolate those adapters (linear or SVDMix) to synthesize any intermediate age from 15 to 75,
- deploy the pipeline via CLI scripts, Gradio UI, or HuggingFace Spaces with GPU memory optimizations.

> **TL;DR**: Data → DreamBooth LoRA (young & old) → LoRA interpolation + prompt fusion → SDXL Img2Img → interactive UI.

---

## 📁 Repository Map

| Path | Description |
| --- | --- |
| `dataset preparation scripts/` | wiki.mat processing, filters, and automation scripts. |
| `dataset/`, `dataset_small/` | Full and toy datasets (young_10_20, old_70_80 splits). |
| `train_dreambooth_lora_sdxl.py` | Rank-4 SDXL LoRA training (DreamBooth). |
| `train_young_lora.ps1`, `train_old_lora.ps1` | Convenience wrappers with dataset paths. |
| `inference_interpolation.py` | SDXL Img2Img inference with linear/SVDMix LoRA fusion. |
| `inference_lora_interp_traverse_infU.py` | InfiniteYou / PuLID style interpolation. |
| `app.py`, `hf_space/` | Gradio front-end (local + HuggingFace Spaces). |
| `InfiniteYou/` | Identity guidance utilities (InfuNet, PuLID). |
| `outputs/` | Sample generations and comparison grids. |

---

## 🚀 Quick Start

```bash
git clone <this-repo>
cd AgeBooth-masterr
python -m venv venv && venv\Scripts\activate  # or source venv/bin/activate
pip install -r requirements.txt
```

> For HuggingFace Spaces or datasets scripts, see `requirements_spaces.txt` and `requirements_dataset.txt`.

---

## 📦 Data Preparation

1. Download IMDB-Wiki (`wiki_crop/wiki_crop/wiki.mat` + images).  
2. Filter & split by age/quality:
   ```powershell
   cd "dataset preparation scripts"
   python prepare_dataset_from_mat.py
   ```
3. Resulting structure:
   ```
   dataset/
     training/young_10_20, old_70_80
     validation/young_10_20, old_70_80
   dataset_small/  # 25 train / 10 val per class for quick loops
   ```

Quality filters include face-score ≥ 1.0, single-face constraint, min resolution 256², and min face size 100 px (see `DATASET_PREPARATION_GUIDE.md` for details).

---

## 🧠 Training (DreamBooth + LoRA)

Each adapter (young & old) is trained separately on SDXL:

```bash
accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --instance_data_dir dataset/training/young_10_20 \
  --output_dir models/ageLoRA/young_10_20 \
  --instance_prompt "photo of young-face-token person" \
  --rank 4 --resolution 512 --train_batch_size 1 \
  --gradient_accumulation_steps 4 --learning_rate 1e-4 \
  --max_train_steps 400 --mixed_precision fp16 \
  --enable_xformers_memory_efficient_attention
```

Key features:

- LoRA injected into every SDXL attention projection (UNet + optional text encoders).
- Training fits inside 6 GB VRAM thanks to FP16, gradient checkpointing, attention slicing, and 8-bit optimizers.
- Final LoRA weights are stored as `.safetensors` under `models/ageLoRA/<age_bucket>/checkpoint-XXX/`.

Use `train_young_lora.ps1` / `train_old_lora.ps1` to run the above with project defaults.

---

## 🌀 Interpolation & Inference

### CLI (SDXL Img2Img)

```bash
python inference_interpolation.py \
  --young_lora models/ageLoRA/young_10_20/checkpoint-200 \
  --old_lora   models/ageLoRA/old_70_80/checkpoint-200 \
  --image_path example_inputs/rihanna.webp \
  --method svdmix --num_steps 7 --strength 0.55 \
  --guidance_scale 7.5 --num_inference_steps 50 \
  --sex f --output_dir outputs/rihanna_svdmix
```

What happens:

1. Load SDXL Img2Img pipeline (`StableDiffusionXLImg2ImgPipeline`) with FP16 + memory optimizations.
2. For each interpolation weight α:
   - Blend LoRA weights (linear or SVDMix).
   - Load the fused adapter, build an age-aware prompt via `get_age_prompt`.
   - Run Img2Img with identity-preserving negative prompts.
3. Save every age PNG and a comparison grid.

### InfiniteYou / PuLID Mode

`inference_lora_interp_traverse_infU.py` adds identity embeddings via InfiniteYou for tighter face preservation and allows direct traversal across α values with PuLID/Infusenet conditioning.

### Gradio UI

```bash
python app.py
```

Features:

- Upload a portrait, choose number of ages, strength, guidance, seed.
- Optional custom prompts per age bracket.
- Gallery output plus comparison grid.
- Ready for HuggingFace Spaces deployment (`hf_space/app.py`).

---

## ⚙️ Optimization Highlights

- **LoRA Rank-4** adapters (~100 MB) fine-tune in <40 minutes per age bucket.
- **Memory tricks**: `enable_attention_slicing`, `enable_vae_slicing`, `enable_model_cpu_offload`, `xformers`.
- **Prompt fusion**: blend CLIP embeddings for young/old prompts, mirroring weight interpolation.
- **SVDMix**: singular-value interpolation of LoRA A/B matrices for smoother mid-age traits.
- **Caching**: HuggingFace cache redirected to `D:` drive; LoRAs cached in memory for the Gradio session.

---

## 🖼️ Sample Outputs

See `outputs/` for:

- `comparison_grid_linear.png` – 15→75 yrs linear blend.  
- `interpolation_svdmix/` – SVDMix progression.  
- Subject-specific folders (`anshika/`, `my_face_indian/`, …) with original + aged PNGs.

---

## 🚀 Deployment

- **Local**: `python app.py` (Gradio queue enabled).  
- **HuggingFace Space**: copy `hf_space/` assets, run `huggingface-cli repo create ...`, then push via `upload_to_hf.py` or `upload_to_hf_git.ps1`.  
- **LoRA distribution**: `upload_to_hf.py` publishes `.safetensors` to the Hub for reproducibility.

---

## 🗺️ Roadmap

- [x] SDXL LoRA training (DreamBooth)
- [x] FLUX LoRA support (via `train_dreambooth_lora_flux.py`)
- [x] InfiniteYou / PuLID inference
- [ ] InstantID inference
- [ ] Public pretrained adapters for multiple demographics

---

## 📚 Citation

This implementation follows [AgeBooth: Controllable Facial Aging and Rejuvenation via Diffusion Models](https://arxiv.org/abs/2510.05715). Cite the original work if you use the method:

```bibtex
@misc{zhu2025ageboothcontrollablefacialaging,
  title         = {AgeBooth: Controllable Facial Aging and Rejuvenation via Diffusion Models},
  author        = {Shihao Zhu and Bohan Cao and Ziheng Ouyang and Zhen Li and Peng-Tao Jiang and Qibin Hou},
  year          = {2025},
  eprint        = {2510.05715},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

---

## 🙌 Acknowledgements

- Stability AI for SDXL.  
- Diffusers, Accelerate, and xFormers teams.  
- Original AgeBooth authors for the inspiration.  
- InfiniteYou / PuLID contributors for identity guidance tooling.
