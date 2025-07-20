import yaml
from diffusers import DiffusionPipeline
import torch
import os
import random
import tomesd
import time
from make_dark import darken_and_blur_image

# Load config from yml file
config_file = "config_new.yml"
with open(config_file, "r") as file:
    config = yaml.safe_load(file)
categories = config["categories"]

# Load the base model
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
# base.to("cuda")
# base.enable_sequential_cpu_offload()
base.enable_model_cpu_offload()
base.enable_xformers_memory_efficient_attention()


# Define inference steps
n_steps = 50
darkness_factor=0.5
blur_radius=5
base_dir = "/Video/SDXL_Output"
os.makedirs(base_dir, exist_ok=True)
# Generate images for each category
for category, prompts in categories.items():
    # Select a random prompt from the category
    
    category_dir = os.path.join(base_dir, category)
    os.makedirs(category_dir, exist_ok=True)
    
    print(f"Processing category '{category}' with prompt: {prompt}")
    
    for i in range(20):
        prompt = random.choice(prompts)
        tic = time.time()
        image = base(
            prompt=prompt,
            num_inference_steps=n_steps,
            height=1024,
            width=1024
        ).images[0]
        toc = time.time()
        print(f'Time required for category "{category}", image {i+1}: {toc - tic:.2f} seconds')
        dark_image = darken_and_blur_image(image, darkness_factor, blur_radius)
        # Save image in the category folder
        image.save(os.path.join(category_dir, f'{category}_image_{i+1}.png'))
        dark_image.save(os.path.join(category_dir, f'dark_{category}_image_{i+1}.png'))
