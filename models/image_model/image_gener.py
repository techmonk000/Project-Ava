import numpy as np 
from diffusers import StableDiffusionPipeline 
from transformers import pipeline, set_seed 
import matplotlib.pyplot as plt 
import cv2 
from PIL import Image 
import torch

class CFG:
    device = "cuda"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12
    save_image_path = "generated_image.png"  

image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.model_id, torch_dtype=torch.float16,
    revision="fp16", use_auth_token="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",
    guidance_scale=9
)
image_gen_model = image_gen_model.to(CFG.device)

def generate_image(prompt, model):

    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]
    
    image = image.resize(CFG.image_gen_size)
    
    image.save(CFG.save_image_path)
    
    return image

prompt = "A futuristic cityscape at sunset"
image = generate_image(prompt, image_gen_model)


