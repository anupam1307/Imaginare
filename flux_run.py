import yaml
from diffusers import DiffusionPipeline
import torch
import os
import random
import tomesd
import time
from make_dark import darken_and_blur_image
from optimum.quanto import freeze, qfloat8, quantize
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL, DiffusionPipeline
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast


# Load config from yml file
config_file = "config_jsr_hmd.yml"
with open(config_file, "r") as file:
    config = yaml.safe_load(file)
categories = config["categories"]

# # Load the base model
# base = DiffusionPipeline.from_pretrained(
#     "black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
# )
# # base.to("cuda")
# # base.enable_sequential_cpu_offload()
# base.enable_model_cpu_offload()
# base.enable_xformers_memory_efficient_attention()

# Define parameters and load models
dtype = torch.bfloat16
bfl_repo = "black-forest-labs/FLUX.1-dev"
revision = "refs/heads/main"

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", revision=revision)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision)
tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype, revision=revision)
vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision)
transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype, revision=revision)

# Quantize and freeze models
quantize(transformer, weights=qfloat8)
freeze(transformer)
quantize(text_encoder_2, weights=qfloat8)
freeze(text_encoder_2)

base_model = "black-forest-labs/FLUX.1-dev"
pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)

pipe.text_encoder_2 = text_encoder_2
pipe.transformer = transformer
pipe.enable_model_cpu_offload()


# Define inference steps
n_steps = 50
darkness_factor=0.5
blur_radius=5
base_dir = "/Video/Output"
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
        image = pipe(
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
