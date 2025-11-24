import gradio as gr
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import os
from safetensors.torch import load_file
import numpy as np

# Global variables for caching
pipe = None
young_weights = None
old_weights = None

def initialize_models():
    """Initialize models and cache them"""
    global pipe, young_weights, old_weights
    
    if pipe is None:
        print("Loading SDXL pipeline...")
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to("cuda")
        
        # Enable memory optimizations
        pipe.enable_attention_slicing(1)
        pipe.enable_vae_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
    
    if young_weights is None:
        print("Loading Young LoRA from HuggingFace Hub...")
        # Try loading from HuggingFace Hub, fallback to local if not available
        try:
            from huggingface_hub import hf_hub_download
            young_path = hf_hub_download(
                repo_id="ShubhamBaghel307/agebooth-loras",
                filename="young_lora.safetensors"
            )
            young_weights = load_file(young_path)
        except:
            print("Fallback to local Young LoRA...")
            young_weights = load_file("models/ageLoRA/young_10_20/checkpoint-200/pytorch_lora_weights.safetensors")
    
    if old_weights is None:
        print("Loading Old LoRA from HuggingFace Hub...")
        # Try loading from HuggingFace Hub, fallback to local if not available
        try:
            from huggingface_hub import hf_hub_download
            old_path = hf_hub_download(
                repo_id="ShubhamBaghel307/agebooth-loras",
                filename="old_lora.safetensors"
            )
            old_weights = load_file(old_path)
        except:
            print("Fallback to local Old LoRA...")
            old_weights = load_file("models/ageLoRA/old_70_80/checkpoint-200/pytorch_lora_weights.safetensors")
    
    return pipe, young_weights, old_weights

def interpolate_lora_weights(young_weights, old_weights, alpha):
    """Linear interpolation between LoRA weights"""
    interpolated = {}
    for key in young_weights.keys():
        if key in old_weights:
            young_tensor = young_weights[key].cpu().float()
            old_tensor = old_weights[key].cpu().float()
            interpolated[key] = (1 - alpha) * young_tensor + alpha * old_tensor
    return interpolated

def get_age_from_alpha(alpha):
    """Map alpha to age: α=1.0 → 15 years, α=0.0 → 75 years"""
    young_age = 15
    old_age = 75
    return int(young_age + (1 - alpha) * (old_age - young_age))

def generate_age_transformations(
    input_image,
    gender,
    num_steps,
    strength,
    guidance_scale,
    num_inference_steps,
    seed,
    custom_prompt_young,
    custom_prompt_middle,
    custom_prompt_old,
    use_custom_prompts,
    progress=gr.Progress()
):
    """Generate age transformation sequence"""
    
    if input_image is None:
        return None, "Please upload an image first!"
    
    # Initialize models
    pipe, young_w, old_w = initialize_models()
    
    # Set seed
    torch.manual_seed(seed)
    
    # Resize input image
    input_image = input_image.resize((512, 512))
    
    # Generate alpha values
    alphas = [i / (num_steps - 1) for i in range(num_steps)]
    
    # Negative prompt
    negative_prompt = "blurry, low quality, distorted, deformed, ugly, bad anatomy, different person, changed face, wrong ethnicity, multiple people, inconsistent features"
    
    generated_images = [input_image]  # Start with original
    
    for idx, alpha in enumerate(alphas):
        age = get_age_from_alpha(alpha)
        progress((idx + 1) / num_steps, desc=f"Generating age {age}...")
        
        # Interpolate LoRA weights
        fused_weights = interpolate_lora_weights(young_w, old_w, alpha)
        
        # Save temporarily
        temp_path = "temp_lora.safetensors"
        from safetensors.torch import save_file
        save_file(fused_weights, temp_path)
        
        # Unload previous LoRA
        if idx > 0:
            pipe.unload_lora_weights()
        
        # Load fused LoRA
        pipe.load_lora_weights(temp_path)
        
        # Generate prompt
        if use_custom_prompts:
            if age < 30:
                prompt = custom_prompt_young.replace("{age}", str(age))
            elif age < 55:
                prompt = custom_prompt_middle.replace("{age}", str(age))
            else:
                prompt = custom_prompt_old.replace("{age}", str(age))
        else:
            # Default prompts
            if age < 25:
                prompt = f"RAW photograph portrait of a handsome young Indian man at age {age}, attractive facial features, strong jawline, clear glowing skin, warm brown complexion, thick dark hair, expressive brown eyes, South Asian ethnicity, youthful smooth skin, professional photography, 8k resolution"
            elif age < 50:
                prompt = f"RAW photograph portrait of a handsome middle-aged Indian man at age {age}, distinguished attractive features, strong jawline, warm brown complexion, dark hair with grays, South Asian ethnicity, professional photography, 8k resolution"
            else:
                prompt = f"RAW photograph portrait of a handsome elderly Indian man at age {age}, dignified features, warm brown complexion, gray hair, South Asian ethnicity, professional photography, 8k resolution"
        
        # Generate image
        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=input_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            ).images[0]
        
        generated_images.append(output)
        
        # Clean up
        os.remove(temp_path)
    
    # Create comparison grid
    grid_width = 512 * len(generated_images)
    grid = Image.new('RGB', (grid_width, 512))
    for idx, img in enumerate(generated_images):
        grid.paste(img, (idx * 512, 0))
    
    status = f"✅ Generated {num_steps} age transformations (ages {get_age_from_alpha(alphas[0])} to {get_age_from_alpha(alphas[-1])})"
    
    return generated_images + [grid], status

# Default prompts
DEFAULT_YOUNG = "RAW photograph portrait of a handsome young Indian person at age {age}, attractive facial features, strong jawline, clear glowing skin, warm brown complexion, thick dark hair, expressive brown eyes, South Asian ethnicity, youthful smooth skin, professional photography, 8k resolution, maintaining same identity"

DEFAULT_MIDDLE = "RAW photograph portrait of a handsome middle-aged Indian person at age {age}, distinguished attractive features, strong jawline, warm brown complexion, dark hair with distinguished grays, expressive brown eyes, South Asian ethnicity, mature refined skin, professional photography, 8k resolution, maintaining same identity"

DEFAULT_OLD = "RAW photograph portrait of a handsome elderly Indian person at age {age}, dignified attractive features, warm brown complexion, distinguished gray/white hair, wise expressive brown eyes, South Asian ethnicity, naturally aged skin with dignity, professional photography, 8k resolution, maintaining same identity"

# Gradio Interface
with gr.Blocks(title="AgeBooth: AI Age Transformation", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎭 AgeBooth: AI Age Transformation
    
    Transform faces across ages (15-75 years) using LoRA interpolation and SDXL.
    
    **How to use:**
    1. Upload a face image (frontal, clear lighting works best)
    2. Adjust settings (or use defaults)
    3. Optionally customize prompts for better results
    4. Click "Generate Age Transformations"
    5. Wait ~5-7 minutes for results
    
    **Trained on:** SDXL with DreamBooth LoRA (rank=4, 200 steps)
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input Face Image", type="pil", height=400)
            
            gr.Markdown("### ⚙️ Basic Settings")
            gender = gr.Radio(["m", "f"], value="m", label="Gender (for prompt generation)")
            num_steps = gr.Slider(3, 11, value=7, step=2, label="Number of Age Steps")
            strength = gr.Slider(0.3, 0.8, value=0.5, step=0.05, label="Transformation Strength (lower = more identity preservation)")
            
            gr.Markdown("### 🎨 Advanced Settings")
            guidance_scale = gr.Slider(5.0, 15.0, value=7.5, step=0.5, label="Guidance Scale")
            num_inference_steps = gr.Slider(25, 75, value=50, step=5, label="Inference Steps (higher = better quality, slower)")
            seed = gr.Slider(0, 999999, value=42, step=1, label="Random Seed")
            
            gr.Markdown("### 📝 Custom Prompts (Optional)")
            use_custom = gr.Checkbox(label="Use Custom Prompts", value=False)
            prompt_young = gr.Textbox(
                label="Young Age Prompt (use {age} placeholder)",
                value=DEFAULT_YOUNG,
                lines=3,
                visible=False
            )
            prompt_middle = gr.Textbox(
                label="Middle Age Prompt (use {age} placeholder)",
                value=DEFAULT_MIDDLE,
                lines=3,
                visible=False
            )
            prompt_old = gr.Textbox(
                label="Old Age Prompt (use {age} placeholder)",
                value=DEFAULT_OLD,
                lines=3,
                visible=False
            )
            
            use_custom.change(
                fn=lambda x: (gr.update(visible=x), gr.update(visible=x), gr.update(visible=x)),
                inputs=[use_custom],
                outputs=[prompt_young, prompt_middle, prompt_old]
            )
            
            generate_btn = gr.Button("🚀 Generate Age Transformations", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            status = gr.Textbox(label="Status", interactive=False)
            gallery = gr.Gallery(
                label="Age Transformation Results",
                show_label=True,
                columns=4,
                rows=2,
                height=600,
                object_fit="contain"
            )
            
            gr.Markdown("""
            ### 📊 Results Guide:
            - **First image:** Original input
            - **Following images:** Age transformations from young to old
            - **Last image:** Full comparison grid
            
            ### 💡 Tips:
            - Lower strength (0.4-0.5) preserves identity better
            - Higher inference steps (50-60) give better quality
            - Custom prompts can help maintain specific features
            - Use frontal face images with good lighting
            """)
    
    generate_btn.click(
        fn=generate_age_transformations,
        inputs=[
            input_image, gender, num_steps, strength, guidance_scale,
            num_inference_steps, seed, prompt_young, prompt_middle, prompt_old, use_custom
        ],
        outputs=[gallery, status]
    )
    
    gr.Markdown("""
    ---
    ### 🔬 Technical Details:
    - **Model:** Stable Diffusion XL (SDXL)
    - **Method:** Linear LoRA Interpolation (AgeBooth paper)
    - **Training:** DreamBooth LoRA on age-specific datasets
    - **GPU:** Optimized for 6GB+ VRAM
    
    ### 📚 Citation:
    If you use this app, please cite the AgeBooth paper.
    
    **Made with ❤️ using Diffusers, Gradio, and SDXL**
    """)

if __name__ == "__main__":
    demo.queue(max_size=5)
    demo.launch(share=True)
