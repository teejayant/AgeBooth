import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import os
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="SDXL LoRA Interpolation Inference (Simple)")
    parser.add_argument("--young_lora", type=str, required=True, help="Path to young LoRA checkpoint")
    parser.add_argument("--old_lora", type=str, required=True, help="Path to old LoRA checkpoint")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input face image")
    parser.add_argument("--output_dir", type=str, default="outputs/sdxl_simple", help="Output directory")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--strength", type=float, default=0.75, help="Transformation strength (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sex", type=str, default="m", choices=["m", "f"], help="Gender for prompt")
    return parser.parse_args()

def load_lora_weights(checkpoint_path):
    """Load LoRA weights from safetensors file."""
    from safetensors.torch import load_file
    lora_path = os.path.join(checkpoint_path, "pytorch_lora_weights.safetensors")
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
    return load_file(lora_path)

def get_age_prompt(lora_type, gender):
    """Generate prompts for young or old LoRA."""
    if lora_type == "young":
        if gender == "m":
            return "portrait, realistic, a young man at the age of 15, youthful features, high quality", 15
        else:
            return "portrait, realistic, a young woman at the age of 15, youthful features, high quality", 15
    else:  # old
        if gender == "m":
            return "portrait, realistic, an elderly man at the age of 75, aged features, high quality", 75
        else:
            return "portrait, realistic, an elderly woman at the age of 75, aged features, high quality", 75

def main():
    args = parse_args()
    
    print("=" * 60)
    print("AgeBooth SDXL Simple LoRA Inference")
    print("=" * 60)
    print(f"Young LoRA: {args.young_lora}")
    print(f"Old LoRA: {args.old_lora}")
    print(f"Input Image: {args.image_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Gender: {args.sex}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Load input image
    print("\n[1/4] Loading input image...")
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Input image not found: {args.image_path}")
    
    input_image = Image.open(args.image_path).convert("RGB")
    input_image = input_image.resize((args.resolution, args.resolution))
    input_image.save(os.path.join(args.output_dir, "00_original.png"))
    
    # Negative prompt
    negative_prompt = "blurry, low quality, distorted, deformed, ugly, bad anatomy"
    
    # Process Young LoRA
    print("[2/4] Generating YOUNG transformation (age 15)...")
    print("  Loading SDXL pipeline...")
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")
    
    print("  Loading Young LoRA...")
    pipe.load_lora_weights(args.young_lora)
    
    young_prompt, young_age = get_age_prompt("young", args.sex)
    
    print("  Generating image...")
    with torch.no_grad():
        young_output = pipe(
            prompt=young_prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps
        ).images[0]
    
    young_output.save(os.path.join(args.output_dir, f"age_{young_age:02d}_young.png"))
    print(f"  ✓ Saved: age_{young_age:02d}_young.png")
    
    # Unload Young LoRA
    print("  Unloading Young LoRA...")
    pipe.unload_lora_weights()
    
    # Process Old LoRA
    print("\n[3/4] Generating OLD transformation (age 75)...")
    print("  Loading Old LoRA...")
    pipe.load_lora_weights(args.old_lora)
    
    old_prompt, old_age = get_age_prompt("old", args.sex)
    
    print("  Generating image...")
    with torch.no_grad():
        old_output = pipe(
            prompt=old_prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps
        ).images[0]
    
    old_output.save(os.path.join(args.output_dir, f"age_{old_age:02d}_old.png"))
    print(f"  ✓ Saved: age_{old_age:02d}_old.png")
    
    # Create comparison grid
    print("\n[4/4] Creating comparison grid...")
    width, height = input_image.size
    comparison = Image.new('RGB', (width * 3, height))
    comparison.paste(input_image, (0, 0))
    comparison.paste(young_output, (width, 0))
    comparison.paste(old_output, (width * 2, 0))
    comparison.save(os.path.join(args.output_dir, "comparison_original_young_old.png"))
    print("  ✓ Saved: comparison_original_young_old.png")
    
    print("\n" + "=" * 60)
    print("✅ Inference Complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)
    print("Generated images:")
    print("  00_original.png (input)")
    print(f"  age_{young_age:02d}_young.png (de-aged)")
    print(f"  age_{old_age:02d}_old.png (aged)")
    print("  comparison_original_young_old.png (side-by-side)")

if __name__ == "__main__":
    main()
