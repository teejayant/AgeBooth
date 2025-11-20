# AgeBooth Training Guide for RTX 4050 (6GB)

## 📋 Prerequisites Checklist

- ✅ RTX 4050 GPU with 6GB VRAM
- ✅ Python environment with all dependencies installed
- ✅ Dataset prepared in `dataset_small/` folder
  - ✅ 25 images in `training/young_10_20/`
  - ✅ 25 images in `training/old_70_80/`
  - ✅ Validation images in respective validation folders

## 🚀 Step-by-Step Training Process

### Step 1: Configure Accelerate (First Time Only)

Run this command to configure accelerate for single GPU training:

```powershell
accelerate config
```

**Answer the prompts as follows:**
- `In which compute environment are you running?` → **This machine**
- `Which type of machine are you using?` → **No distributed training**
- `Do you want to run your training on CPU only?` → **NO**
- `Do you wish to optimize your script with torch dynamo?` → **NO**
- `Do you want to use DeepSpeed?` → **NO**
- `What GPU(s) should be used?` → **0** (your RTX 4050)
- `Do you wish to use FP16 or BF16 (mixed precision)?` → **fp16**

### Step 2: Train Young LoRA (Age 10-20)

```powershell
.\train_young_lora.ps1
```

**Expected:**
- Training time: 15-20 minutes
- VRAM usage: ~5.5 GB
- Output: `models/ageLoRA/young_10_20/`
- Checkpoints saved every 100 steps

**What happens:**
- Downloads SDXL base model (~6.9 GB) - only first time
- Trains LoRA adapter on young faces (25 images)
- Generates validation images every 100 steps
- Saves final LoRA weights (~100 MB)

### Step 3: Train Old LoRA (Age 70-80)

```powershell
.\train_old_lora.ps1
```

**Expected:**
- Training time: 15-20 minutes
- VRAM usage: ~5.5 GB
- Output: `models/ageLoRA/old_70_80/`
- Checkpoints saved every 100 steps

### Step 4: Monitor Training Progress

**Option A: TensorBoard (Recommended)**

Open a new terminal and run:
```powershell
tensorboard --logdir=models/ageLoRA
```

Then open your browser to: `http://localhost:6006`

You'll see:
- Training loss curves
- Generated validation images
- Training progress

**Option B: Check Output Folders**

Look at checkpoint folders:
```
models/ageLoRA/young_10_20/checkpoint-100/
models/ageLoRA/young_10_20/checkpoint-200/
models/ageLoRA/young_10_20/checkpoint-300/
models/ageLoRA/young_10_20/checkpoint-400/
```

### Step 5: Test Your Trained LoRAs

After both LoRAs are trained, you need to:

1. **Prepare inference script** (we'll need to adapt the existing one for Windows)
2. **Test interpolation** between young and old LoRAs
3. **Generate age progression sequences**

## ⚙️ Training Parameters Explained

| Parameter | Value | Why? |
|-----------|-------|------|
| `resolution=512` | 512×512 | Fits in 6GB VRAM |
| `train_batch_size=1` | 1 | Memory constraint |
| `gradient_accumulation_steps=4` | 4 | Effective batch = 4 |
| `learning_rate=1e-4` | 0.0001 | Optimal for LoRA |
| `max_train_steps=400` | 400 | 16 epochs × 25 images |
| `rank=16` | 16 | LoRA rank (quality/size) |
| `gradient_checkpointing` | ON | Saves VRAM |
| `use_8bit_adam` | ON | Reduces memory |
| `mixed_precision=fp16` | ON | 2× faster, less VRAM |
| `enable_xformers` | ON | Efficient attention |

## 🔍 Troubleshooting

### Issue: Out of Memory (OOM)

**Solution 1:** Reduce resolution
```powershell
# Change in the .ps1 file
--resolution=448
```

**Solution 2:** Enable more optimizations
```powershell
# Already enabled:
--gradient_checkpointing
--use_8bit_adam
--mixed_precision="fp16"
--enable_xformers_memory_efficient_attention
```

**Solution 3:** Close other applications
- Close Chrome/browsers
- Close Discord, gaming overlays
- Restart GPU driver if needed

### Issue: Training is Very Slow

**Check:**
1. GPU is being used: `nvidia-smi` in terminal
2. Mixed precision is enabled (fp16)
3. xformers is installed correctly

### Issue: Poor Quality Results

**Solutions:**
1. Train longer: Change `max_train_steps` to 600-800
2. Increase LoRA rank: `--rank=32` (but needs more VRAM)
3. Adjust learning rate: Try `5e-5` or `2e-4`
4. Check dataset quality: Ensure faces are clear and centered

### Issue: CUDA Out of Memory During Validation

Add to training script:
```powershell
--validation_batch_size=1
```

## 📊 Expected Training Metrics

**Good training:**
- Loss starts at ~0.15-0.20
- Loss decreases to ~0.05-0.08
- Validation images show clear age features
- No mode collapse or artifacts

**Warning signs:**
- Loss stays high (>0.15) after 200 steps
- Validation images are blurry
- Artifacts or distortions appear

## 🎯 After Training

Once both LoRAs are trained, you'll have:

```
models/
└── ageLoRA/
    ├── young_10_20/
    │   ├── pytorch_lora_weights.safetensors
    │   └── checkpoint-100, 200, 300, 400/
    └── old_70_80/
        ├── pytorch_lora_weights.safetensors
        └── checkpoint-100, 200, 300, 400/
```

**Next:** You'll need to run the inference script to:
1. Load both LoRAs
2. Interpolate between them (α = 0.0 to 1.0)
3. Generate age progression sequences
4. Combine with InfiniteYou for identity preservation

## 💾 Disk Space Required

- SDXL base model: ~6.9 GB
- Each LoRA: ~100 MB
- Checkpoints (4 per LoRA): ~400 MB each = 3.2 GB total
- TensorBoard logs: ~100 MB
- **Total: ~10-12 GB**

## ⏱️ Total Time Estimate

- Young LoRA training: 15-20 minutes
- Old LoRA training: 15-20 minutes
- **Total training time: 30-40 minutes**

## 🎓 What's Happening During Training?

1. **Model loads** SDXL base model (frozen weights)
2. **LoRA adapters** are initialized (trainable)
3. **For each step:**
   - Load batch of face images
   - Add noise (forward diffusion)
   - Predict noise with LoRA-enhanced U-Net
   - Calculate loss (MSE between predicted and actual noise)
   - Update only LoRA weights (not SDXL base)
4. **Save checkpoints** every 100 steps
5. **Generate validation images** to monitor quality

The magic: You're only training 2M parameters (LoRA) instead of 6.9B (full model)!
