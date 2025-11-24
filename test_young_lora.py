import torch 
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import os

print("Loading SDXL Img2Img pipeline...")
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

print("Loading Young LoRA...")
pipe.load_lora_weights("models/ageLoRA/young_10_20/checkpoint-200")

# Create output folder
os.makedirs("Lora_testing", exist_ok=True)

# Load input face image
print("Loading input image...")
input_image_path = "input_face.jpg"  # Change this to your image path
input_image = Image.open(input_image_path).convert("RGB")
input_image = input_image.resize((512, 512))

# Save original for comparison
input_image.save("Lora_testing/original.png")

print("Transforming to young age...")

# Transform to young age
prompt = "A person at young age, portrait, 15 years old, high quality, same person, same identity, youthful features"
negative_prompt = "old, elderly, wrinkles, aged, different person, blurry, low quality"

output_image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=input_image,
    strength=0.75,  # How much to transform (0.5 = subtle, 0.9 = dramatic)
    guidance_scale=7.5,
    num_inference_steps=50
).images[0]

output_image.save("Lora_testing/test_young_lora_output.png")
print("✅ Transformation complete!")

# Create side-by-side comparison
print("Creating comparison...")
width, height = input_image.size
comparison = Image.new('RGB', (width * 2, height))
comparison.paste(input_image, (0, 0))
comparison.paste(output_image, (width, 0))
comparison.save("Lora_testing/comparison_before_after.png")

print("✅ All images saved in Lora_testing/ folder!")
print("  - original.png (input)")
print("  - test_young_lora_output.png (de-aged)")
print("  - comparison_before_after.png (side-by-side)")