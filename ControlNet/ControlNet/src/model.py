import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from PIL import Image
import numpy as np
import cv2

base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
controlnet_model_id = "diffusers/controlnet-canny-sdxl-1.0"
vae_model_id = "madebyollin/sdxl-vae-fp16-fix"

def load_models():
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_id,
        torch_dtype=torch.float16
    )

    vae = AutoencoderKL.from_pretrained(
        vae_model_id,
        torch_dtype=torch.float16
    )

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base_model_id,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
    )

    pipe.to("cuda")
    return pipe

def process_image(input_image_path):
    original_image = Image.open(input_image_path).convert("RGB")
    original_image = original_image.resize((1024, 1024))
    image_np = np.array(original_image)

    low_threshold = 100
    high_threshold = 200
    canny_image_np = cv2.Canny(image_np, low_threshold, high_threshold)
    canny_image_np = canny_image_np[:, :, None]
    canny_image_np = np.concatenate([canny_image_np, canny_image_np, canny_image_np], axis=2)

    control_image = Image.fromarray(canny_image_np)
    return control_image

def generate_image(pipe, prompt, negative_prompt, control_image):
    generator = torch.manual_seed(42)
    output_image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        controlnet_conditioning_scale=0.5,
        generator=generator,
        num_inference_steps=30,
    ).images[0].convert("RGB")
    return output_image