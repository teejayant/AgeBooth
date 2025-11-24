import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import os
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="SDXL LoRA Interpolation Inference")
    parser.add_argument("--young_lora", type=str, required=True, help="Path to young LoRA checkpoint")
    parser.add_argument("--old_lora", type=str, required=True, help="Path to old LoRA checkpoint")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input face image")
    parser.add_argument("--output_dir", type=str, default="outputs/sdxl_interpolation", help="Output directory")
    parser.add_argument("--num_steps", type=int, default=7, help="Number of interpolation steps (including endpoints)")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--strength", type=float, default=0.75, help="Transformation strength (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sex", type=str, default="m", choices=["m", "f"], help="Gender for prompt")
    return parser.parse_args()

def interpolate_lora_weights(young_weights, old_weights, alpha):
    """
    Interpolate between young and old LoRA weights.
    alpha=0.0: young (age 15)
    alpha=1.0: old (age 75)
    """
    interpolated = {}
    for key in young_weights.keys():
        if key in old_weights:
            # Ensure tensors are on CPU for interpolation
            young_tensor = young_weights[key].cpu() if hasattr(young_weights[key], 'cpu') else young_weights[key]
            old_tensor = old_weights[key].cpu() if hasattr(old_weights[key], 'cpu') else old_weights[key]
            interpolated[key] = (1 - alpha) * young_tensor + alpha * old_tensor
    return interpolated

def load_lora_weights(checkpoint_path):
    """Load LoRA weights from safetensors file."""
    from safetensors.torch import load_file
    lora_path = os.path.join(checkpoint_path, "pytorch_lora_weights.safetensors")
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
    return load_file(lora_path)

def get_age_from_alpha(alpha):
    """Map alpha to approximate age."""
    young_age = 15
    old_age = 75
    return int(young_age + alpha * (old_age - young_age))

def get_prompt(age, gender):
    """Generate age-appropriate prompt."""
    if gender == "m":
        if age < 25:
            return f"portrait, realistic, a young man at the age of {age}, youthful features, high quality"
        elif age < 50:
            return f"portrait, realistic, a man at the age of {age}, mature features, high quality"
        else:
            return f"portrait, realistic, an elderly man at the age of {age}, aged features, high quality"
    else:  # female
        if age < 25:
            return f"portrait, realistic, a young woman at the age of {age}, youthful features, high quality"
        elif age < 50:
            return f"portrait, realistic, a woman at the age of {age}, mature features, high quality"
        else:
            return f"portrait, realistic, an elderly woman at the age of {age}, aged features, high quality"

def main():
    args = parse_args()
    
    print("=" * 60)
    print("AgeBooth SDXL LoRA Interpolation Inference")
    print("=" * 60)
    print(f"Young LoRA: {args.young_lora}")
    print(f"Old LoRA: {args.old_lora}")
    print(f"Input Image: {args.image_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Interpolation Steps: {args.num_steps}")
    print(f"Gender: {args.sex}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Load base pipeline
    print("\n[1/5] Loading SDXL pipeline...")
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")
    
    # Load LoRA weights
    print("[2/5] Loading LoRA weights...")
    young_weights = load_lora_weights(args.young_lora)
    old_weights = load_lora_weights(args.old_lora)
    print(f"  Young LoRA keys: {len(young_weights)}")
    print(f"  Old LoRA keys: {len(old_weights)}")
    
    # Load input image
    print("[3/5] Loading input image...")
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Input image not found: {args.image_path}")
    
    input_image = Image.open(args.image_path).convert("RGB")
    input_image = input_image.resize((args.resolution, args.resolution))
    input_image.save(os.path.join(args.output_dir, "00_original.png"))
    
    # Generate alpha values (linear interpolation)
    print("[4/5] Generating interpolation steps...")
    alphas = [i / (args.num_steps - 1) for i in range(args.num_steps)]
    print(f"  Alpha values: {[f'{a:.2f}' for a in alphas]}")
    
    # Negative prompt
    negative_prompt = "blurry, low quality, distorted, deformed, ugly, bad anatomy"
    
    # Generate images for each interpolation step
    print(f"[5/5] Generating {args.num_steps} age transformations...")
    
    # Disable bitsandbytes to avoid Windows compatibility issues
    import os as os_module
    os_module.environ['BITSANDBYTES_NOWELCOME'] = '1'
    
    for idx, alpha in enumerate(tqdm(alphas, desc="Progress")):
        age = get_age_from_alpha(alpha)
        
        # Interpolate LoRA weights
        interpolated_weights = interpolate_lora_weights(young_weights, old_weights, alpha)
        
        # Save interpolated weights temporarily
        temp_lora_path = os.path.join(args.output_dir, f"temp_lora_alpha_{alpha:.2f}.safetensors")
        from safetensors.torch import save_file
        save_file(interpolated_weights, temp_lora_path)
        
        # Unload previous LoRA
        if idx > 0:
            pipe.unload_lora_weights()
        
        # Load interpolated LoRA with low_cpu_mem_usage to avoid bitsandbytes
        pipe.load_lora_weights(temp_lora_path, low_cpu_mem_usage=False)
        
        # Generate prompt
        prompt = get_prompt(age, args.sex)
        
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
        output_filename = f"age_{age:02d}_alpha_{alpha:.2f}.png"
        output.save(os.path.join(args.output_dir, output_filename))
        
        # Clean up temp LoRA file
        os.remove(temp_lora_path)
        
        print(f"  ✓ Age {age} (α={alpha:.2f}) -> {output_filename}")
    
    print("\n" + "=" * 60)
    print("✅ Inference Complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)
    print("Generated images:")
    print("  00_original.png (input)")
    for idx, alpha in enumerate(alphas):
        age = get_age_from_alpha(alpha)
        print(f"  age_{age:02d}_alpha_{alpha:.2f}.png")

if __name__ == "__main__":
    main()
