# HuggingFace Deployment Guide 🚀

Complete guide to deploy AgeBooth to HuggingFace Spaces.

## Prerequisites

1. **HuggingFace Account:** Create at [huggingface.co](https://huggingface.co/join)
2. **Access Token:** Generate at [Settings > Access Tokens](https://huggingface.co/settings/tokens)
   - Select "Write" permission
3. **Git LFS:** Install for large files: `git lfs install`

## Step 1: Upload LoRA Models 📦

First, upload your trained LoRA weights to HuggingFace Hub:

```powershell
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Run upload script
python upload_to_hf.py
```

**What it does:**
- Creates repository: `{your-username}/agebooth-loras`
- Uploads `young_lora.safetensors` (~20MB)
- Uploads `old_lora.safetensors` (~20MB)
- Generates model card with usage instructions

**Input prompts:**
- HuggingFace username (e.g., `ShubhamBaghel309`)
- HuggingFace token (paste your write-access token)

**Verify upload:**
Visit `https://huggingface.co/{your-username}/agebooth-loras`

## Step 2: Update app.py with Your Username 📝

Edit `app.py` line 24 and 34 to use YOUR username:

```python
repo_id="YOUR_USERNAME/agebooth-loras"  # Replace YOUR_USERNAME
```

Current default: `ShubhamBaghel309/agebooth-loras`

## Step 3: Create HuggingFace Space 🎨

### Option A: Web Interface (Recommended)

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in details:
   - **Space name:** `agebooth-age-transformation`
   - **License:** Apache 2.0
   - **SDK:** Gradio
   - **Hardware:** GPU (T4 small or better)
   - **Visibility:** Public or Private
3. Click "Create Space"

### Option B: Git CLI

```powershell
# Clone the space repository
git clone https://huggingface.co/spaces/{your-username}/agebooth-age-transformation
cd agebooth-age-transformation

# Copy files
cp ../app.py .
cp ../requirements_spaces.txt ./requirements.txt
cp ../README_HF.md ./README.md

# Commit and push
git add .
git commit -m "Initial commit: AgeBooth app"
git push
```

## Step 4: Deploy Files 📤

Upload these files to your Space:

### Required Files:
1. **app.py** - Main Gradio application
2. **requirements.txt** - Python dependencies (use `requirements_spaces.txt`)
3. **README.md** - Space description (use `README_HF.md`)

### Upload via Web UI:
1. Go to your Space page
2. Click "Files" tab
3. Click "Add file" > "Upload files"
4. Drag and drop: `app.py`, `requirements_spaces.txt` (rename to `requirements.txt`), `README_HF.md` (rename to `README.md`)
5. Commit changes

### Upload via Git:
```powershell
cd path/to/space/repo
git add app.py requirements.txt README.md
git commit -m "Add application files"
git push
```

## Step 5: Configure Hardware ⚙️

Your Space needs GPU to run SDXL:

1. Go to Space Settings
2. Under "Hardware", select:
   - **T4 small** (free tier, may have queue)
   - **T4 medium** (faster, paid)
   - **A10G small** (best performance, paid)
3. Save settings
4. Space will restart with GPU

## Step 6: Test Your Space 🧪

1. Wait for Space to build (5-10 minutes first time)
2. Check logs for errors: Click "Logs" in top bar
3. Once running, test the interface:
   - Upload a face image
   - Set parameters (default: 7 steps, strength 0.5)
   - Generate transformations
   - Verify output grid

## Common Issues & Solutions 🔧

### Issue: "Out of memory"
**Solution:** Reduce `num_inference_steps` in UI or upgrade hardware

### Issue: "Model not found"
**Solution:** 
- Verify LoRA upload at `huggingface.co/{username}/agebooth-loras`
- Check `repo_id` in `app.py` matches your username
- Ensure files are named `young_lora.safetensors` and `old_lora.safetensors`

### Issue: "Building" takes forever
**Solution:** 
- Check logs for errors
- Verify `requirements.txt` has correct package versions
- Try rebuilding: Settings > Factory reboot

### Issue: GPU not detected
**Solution:** 
- Verify hardware setting is set to GPU (not CPU)
- May need to upgrade to paid tier
- Check if free GPU quota exceeded

### Issue: Import errors
**Solution:**
- Ensure all packages in `requirements.txt`
- Check PyTorch version compatibility: `torch>=2.1.0`
- Verify xformers version: `xformers>=0.0.23`

## File Structure 📁

Your Space should have:
```
agebooth-age-transformation/
├── app.py                    # Main Gradio app
├── requirements.txt          # Python dependencies
├── README.md                 # Space description
└── .gitignore               # (optional) Ignore temp files
```

## Performance Tips ⚡

1. **Hardware:** T4 small works but slow (~30 sec/image). A10G recommended for production
2. **Caching:** First run loads models (~5 min), subsequent runs faster
3. **Batch Size:** UI generates one age at a time to show progress
4. **Memory:** 50 inference steps fits in 6GB VRAM. Increase for quality

## Cost Estimation 💰

- **Free Tier:** T4 small with queue, limited hours
- **T4 medium:** ~$0.60/hour (unlimited)
- **A10G small:** ~$3.15/hour (best performance)

## Example Space URLs 🌐

After deployment:
- **Space:** `https://huggingface.co/spaces/{username}/agebooth-age-transformation`
- **Direct App:** `https://{username}-agebooth-age-transformation.hf.space`
- **LoRA Model:** `https://huggingface.co/{username}/agebooth-loras`

## Sharing Your Space 🎉

Once deployed, share with:
```markdown
Try AgeBooth: https://huggingface.co/spaces/{username}/agebooth-age-transformation

Transform any face across ages 15-75 using AI!
```

## Next Steps 🚀

1. ✅ **Test locally:** Run `python app.py` to test before deploying
2. ✅ **Upload LoRAs:** Run `python upload_to_hf.py`
3. ✅ **Create Space:** Use web interface or git
4. ✅ **Deploy files:** Upload app.py, requirements.txt, README.md
5. ✅ **Set GPU:** Configure T4 or better
6. ✅ **Test live:** Verify all features work
7. ✅ **Share:** Post on Twitter, LinkedIn, Reddit!

## Support 💬

- **HuggingFace Docs:** [huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)
- **Gradio Docs:** [gradio.app/docs](https://gradio.app/docs)
- **Diffusers Docs:** [huggingface.co/docs/diffusers](https://huggingface.co/docs/diffusers)

---

**Good luck with your deployment! 🎭✨**
