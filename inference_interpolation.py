import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import os
import argparse
from tqdm import tqdm
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="AgeBooth LoRA Interpolation (Linear + SVDMix)")
    parser.add_argument("--young_lora", type=str, required=True, help="Path to young LoRA checkpoint")
    parser.add_argument("--old_lora", type=str, required=True, help="Path to old LoRA checkpoint")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input face image")
    parser.add_argument("--output_dir", type=str, default="outputs/interpolation", help="Output directory")
    parser.add_argument("--method", type=str, default="svdmix", choices=["linear", "svdmix"], 
                        help="Interpolation method: linear or svdmix")
    parser.add_argument("--num_steps", type=int, default=7, help="Number of age interpolation steps")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--strength", type=float, default=0.75, help="Transformation strength")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sex", type=str, default="m", choices=["m", "f"], help="Gender")
    return parser.parse_args()

def load_lora_weights(checkpoint_path):
    """Load LoRA weights from safetensors file."""
    from safetensors.torch import load_file
    lora_path = os.path.join(checkpoint_path, "pytorch_lora_weights.safetensors")
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
    return load_file(lora_path)

def split_lora_weights(lora_weights):
    """
    Split LoRA weights into A and B matrices.
    LoRA format: ∆θ = B·A where B ∈ R^(m×r), A ∈ R^(r×n)
    """
    lora_a = {}
    lora_b = {}
    
    for key, value in lora_weights.items():
        if '.lora_A' in key or '.down' in key:
            base_key = key.replace('.lora_A.weight', '').replace('.down.weight', '')
            lora_a[base_key] = value
        elif '.lora_B' in key or '.up' in key:
            base_key = key.replace('.lora_B.weight', '').replace('.up.weight', '')
            lora_b[base_key] = value
    
    return lora_a, lora_b

def svd_mix(M0, M1, alpha):
    """
    SVD-based matrix fusion from paper Eq. 5-7:
    SVDMix(M0, M1; α) = U_α Σ_α V_α^T
    where X_α = α·X_0 + (1-α)·X_1 for X ∈ {U, Σ, V}
    
    Memory-optimized: Clear intermediate tensors immediately
    """
    # Move to CPU for SVD computation
    M0_cpu = M0.cpu().float()
    M1_cpu = M1.cpu().float()
    
    # Compute SVD for both matrices
    U0, S0, V0 = torch.svd(M0_cpu)
    U1, S1, V1 = torch.svd(M1_cpu)
    
    # Clear CPU matrices immediately
    del M0_cpu, M1_cpu
    
    # Interpolate in SVD space
    U_alpha = alpha * U0 + (1 - alpha) * U1
    S_alpha = alpha * S0 + (1 - alpha) * S1
    V_alpha = alpha * V0 + (1 - alpha) * V1
    
    # Clear intermediate SVD components
    del U0, S0, V0, U1, S1, V1
    
    # Reconstruct matrix
    M_fused = U_alpha @ torch.diag(S_alpha) @ V_alpha.T
    
    # Clear interpolation components
    del U_alpha, S_alpha, V_alpha
    
    # Force garbage collection
    import gc
    gc.collect()
    
    return M_fused.to(M0.device).to(M0.dtype)

def fuse_lora_linear(young_weights, old_weights, alpha):
    """
    Naive linear LoRA fusion from paper Eq. 4:
    ∆θ_fused = α·∆θ_young + (1-α)·∆θ_old
    """
    fused = {}
    for key in young_weights.keys():
        if key in old_weights:
            young_tensor = young_weights[key].cpu().float()
            old_tensor = old_weights[key].cpu().float()
            fused[key] = alpha * young_tensor + (1 - alpha) * old_tensor
    return fused

def fuse_lora_svdmix(young_a, young_b, old_a, old_b, alpha):
    """
    SVD-based LoRA fusion from paper Eq. 9-11:
    B_fused = SVDMix(B_young, B_old; α)
    A_fused = SVDMix(A_young, A_old; α)
    ∆θ_fused = B_fused · A_fused
    
    Complexity: O((m+n)r²) instead of O(min(m,n)²·max(m,n))
    """
    fused_weights = {}
    
    # Get all unique base keys
    all_keys = set(young_a.keys()) | set(young_b.keys())
    
    for base_key in all_keys:
        if base_key in young_a and base_key in old_a and base_key in young_b and base_key in old_b:
            # Apply SVDMix to B matrices (up/lora_B)
            B_fused = svd_mix(young_b[base_key], old_b[base_key], alpha)
            
            # Apply SVDMix to A matrices (down/lora_A)
            A_fused = svd_mix(young_a[base_key], old_a[base_key], alpha)
            
            # Reconstruct full LoRA weight: ∆θ = B·A
            # Store as separate A and B for compatibility
            if '.lora_A' in list(young_a.keys())[0]:
                fused_weights[f"{base_key}.lora_A.weight"] = A_fused
                fused_weights[f"{base_key}.lora_B.weight"] = B_fused
            else:
                fused_weights[f"{base_key}.down.weight"] = A_fused
                fused_weights[f"{base_key}.up.weight"] = B_fused
    
    return fused_weights

def interpolate_prompts(young_prompt, old_prompt, alpha, text_encoder, tokenizer):
    """
    Prompt fusion from paper Eq. 12:
    c_fused = α·c_young + (1-α)·c_old
    where c = Γ(p) is the text embedding
    """
    # Tokenize prompts
    young_tokens = tokenizer(young_prompt, padding="max_length", max_length=77, return_tensors="pt")
    old_tokens = tokenizer(old_prompt, padding="max_length", max_length=77, return_tensors="pt")
    
    # Get embeddings
    with torch.no_grad():
        young_embed = text_encoder(young_tokens.input_ids.to(text_encoder.device))[0]
        old_embed = text_encoder(old_tokens.input_ids.to(text_encoder.device))[0]
    
    # Interpolate in embedding space
    fused_embed = alpha * young_embed + (1 - alpha) * old_embed
    
    return fused_embed

def get_age_from_alpha(alpha):
    """Map alpha to age: α=1.0 → 15 years, α=0.0 → 75 years"""
    young_age = 15
    old_age = 75
    return int(young_age + (1 - alpha) * (old_age - young_age))

def get_age_prompt(age, gender):
    """Generate age-specific prompt"""
    if gender == "m":
        if age < 25:
            return f"portrait, realistic, a young man at the age of {age}, youthful features, high quality"
        elif age < 50:
            return f"portrait, realistic, a man at the age of {age}, mature features, high quality"
        else:
            return f"portrait, realistic, an elderly man at the age of {age}, aged features, high quality"
    else:
        if age < 25:
            return f"portrait, realistic, a young woman at the age of {age}, youthful features, high quality"
        elif age < 50:
            return f"portrait, realistic, a woman at the age of {age}, mature features, high quality"
        else:
            return f"portrait, realistic, an elderly woman at the age of {age}, aged features, high quality"

def main():
    args = parse_args()
    
    print("=" * 70)
    print("AgeBooth: Training-free LoRA and Prompt Fusion")
    print("=" * 70)
    print(f"Method: {args.method.upper()}")
    print(f"Young LoRA: {args.young_lora}")
    print(f"Old LoRA: {args.old_lora}")
    print(f"Input Image: {args.image_path}")
    print(f"Interpolation Steps: {args.num_steps}")
    print(f"Gender: {args.sex}")
    print("=" * 70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    
    # Load input image
    print("\n[1/6] Loading input image...")
    input_image = Image.open(args.image_path).convert("RGB")
    input_image = input_image.resize((args.resolution, args.resolution))
    input_image.save(os.path.join(args.output_dir, "00_original.png"))
    
    # Load pipeline with LOW_CPU_MEM_USAGE to reduce RAM consumption
    print("[2/6] Loading SDXL pipeline (memory-optimized)...")
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    
    # Move to GPU and enable memory optimizations
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing(1)
    pipe.enable_vae_slicing()
    pipe.unet.to(memory_format=torch.channels_last)  # Optimize memory layout
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("  ✓ XFormers memory-efficient attention enabled")
    except:
        print("  ⚠ XFormers not available, using attention slicing")
    
    # Enable model CPU offload to reduce GPU VRAM usage
    try:
        pipe.enable_model_cpu_offload()
        print("  ✓ CPU offload enabled (reduces VRAM usage)")
    except:
        pass
    
    # Load LoRA weights
    print("[3/6] Loading LoRA weights...")
    young_weights = load_lora_weights(args.young_lora)
    old_weights = load_lora_weights(args.old_lora)
    print(f"  Young LoRA: {len(young_weights)} parameters")
    print(f"  Old LoRA: {len(old_weights)} parameters")
    
    # Generate alpha values for interpolation
    print(f"[4/6] Generating {args.num_steps} interpolation steps...")
    alphas = [i / (args.num_steps - 1) for i in range(args.num_steps)]
    print(f"  Alpha range: {alphas[0]:.2f} → {alphas[-1]:.2f}")
    print(f"  Age range: {get_age_from_alpha(alphas[0])} → {get_age_from_alpha(alphas[-1])}")
    
    # Prepare for SVDMix if needed
    if args.method == "svdmix":
        print("[5/6] Preparing SVD-based fusion...")
        young_a, young_b = split_lora_weights(young_weights)
        old_a, old_b = split_lora_weights(old_weights)
        print(f"  Split into A matrices: {len(young_a)} layers")
        print(f"  Split into B matrices: {len(young_b)} layers")
    
    # Generate images
    print(f"[6/6] Generating age-transformed images ({args.method})...")
    negative_prompt = "blurry, low quality, distorted, deformed, ugly, bad anatomy"
    
    for idx, alpha in enumerate(tqdm(alphas, desc="Progress")):
        age = get_age_from_alpha(alpha)
        
        # Fuse LoRA weights
        if args.method == "linear":
            # Naive linear fusion (Eq. 4)
            fused_weights = fuse_lora_linear(young_weights, old_weights, alpha)
        else:  # svdmix
            # SVD-based fusion (Eq. 9-11)
            fused_weights = fuse_lora_svdmix(young_a, young_b, old_a, old_b, alpha)
        
        # Save fused LoRA temporarily
        temp_path = os.path.join(args.output_dir, f"temp_lora_{alpha:.2f}.safetensors")
        from safetensors.torch import save_file
        save_file(fused_weights, temp_path)
        
        # Unload previous LoRA
        if idx > 0:
            pipe.unload_lora_weights()
        
        # Load fused LoRA
        pipe.load_lora_weights(temp_path)
        
        # Generate age-specific prompt
        prompt = get_age_prompt(age, args.sex)
        
        # Generate image
        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=input_image,
                strength=args.strength,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps
            ).images[0]
        
        # Save result
        output_name = f"age_{age:02d}_alpha_{alpha:.2f}_{args.method}.png"
        output.save(os.path.join(args.output_dir, output_name))
        
        # Clean up temp file
        os.remove(temp_path)
    
    # Create comparison grid
    print("\nCreating comparison grid...")
    images = [input_image]
    for alpha in alphas:
        age = get_age_from_alpha(alpha)
        img_path = os.path.join(args.output_dir, f"age_{age:02d}_alpha_{alpha:.2f}_{args.method}.png")
        images.append(Image.open(img_path))
    
    # Create horizontal grid
    grid_width = args.resolution * len(images)
    grid = Image.new('RGB', (grid_width, args.resolution))
    for idx, img in enumerate(images):
        grid.paste(img, (idx * args.resolution, 0))
    grid.save(os.path.join(args.output_dir, f"comparison_grid_{args.method}.png"))
    
    print("\n" + "=" * 70)
    print("✅ Inference Complete!")
    print(f"Method: {args.method.upper()}")
    print(f"Results: {args.output_dir}")
    print("=" * 70)
    print("Generated images:")
    print("  00_original.png (input)")
    for alpha in alphas:
        age = get_age_from_alpha(alpha)
        print(f"  age_{age:02d}_alpha_{alpha:.2f}_{args.method}.png")
    print(f"  comparison_grid_{args.method}.png (all ages side-by-side)")

if __name__ == "__main__":
    main()
