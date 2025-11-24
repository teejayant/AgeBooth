import gradio as gr
import torch
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
import numpy as np
import cv2

# Load models
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
controlnet_model_id = "diffusers/controlnet-canny-sdxl-1.0"
vae_model_id = "madebyollin/sdxl-vae-fp16-fix"

controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=torch.float16)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_id,
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
)

pipe.to("cuda")

def process_image(input_image, prompt, negative_prompt):
    original_image = input_image.convert("RGB")
    original_image = original_image.resize((1024, 1024))

    image_np = np.array(original_image)
    low_threshold = 100
    high_threshold = 200
    canny_image_np = cv2.Canny(image_np, low_threshold, high_threshold)

    canny_image_np = canny_image_np[:, :, None]
    canny_image_np = np.concatenate([canny_image_np, canny_image_np, canny_image_np], axis=2)

    control_image = Image.fromarray(canny_image_np)

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

# Gradio interface
iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Textbox(label="Prompt", placeholder="Enter your prompt here..."),
        gr.Textbox(label="Negative Prompt", placeholder="Enter negative prompt here...")
    ],
    outputs=gr.Image(label="Output Image"),
    title="ControlNet Age Transformation",
    description="Upload an image and provide prompts to generate an aged portrait using ControlNet."
)

if __name__ == "__main__":
    iface.launch()