
# Image Generation & Post-Processing using Diffusion Models

This project allows you to generate high-quality images using **Stable Diffusion XL** and **FLUX** models. It then applies custom post-processing (darkening and blurring) to these images.

Designed with simplicity in mind, this is a great starting point for beginners in AI image generation.

---

##  Project Structure

```
├── flux_run.py                # Generate images using FLUX diffusion model
├── sdxl_run.py                # Generate images using Stable Diffusion XL
├── make_dark.py               # Contains function to darken and blur images
├── config_flux.yml            # Prompt categories for FLUX model
├── config_new.yml             # Prompt categories for SDXL model
```

---

##  Features

-  **Automated image generation** for different categories using prompts.
-  **Post-processing** using `PIL` to darken and blur images.
-  Uses state-of-the-art models like:
  - `black-forest-labs/FLUX.1-dev`
  - `stabilityai/stable-diffusion-xl-base-1.0`
-  Uses HuggingFace `diffusers`, `transformers`, and other modern ML tools.

---

##  Requirements

Install all the required libraries:

```bash
pip install torch diffusers transformers tomesd optimum pytorch pillow pyyaml
```

---

##  Input Files

- `config_flux.yml`: YAML file with prompts grouped by categories for `flux_run.py`.
- `config_new.yml`: YAML file with prompts grouped by categories for `sdxl_run.py`.

Here’s a sample format:

```yaml
categories:
  nature:
    - "A beautiful landscape with mountains and rivers"
    - "Sunset over a calm ocean"
  futuristic:
    - "A city skyline in 3023 with flying cars"
    - "AI-powered futuristic village"
```

---

##  How to Run

###  1. Using the Stable Diffusion XL model

This script uses **Stability AI's** SDXL to generate and save images.

```bash
python sdxl_run.py
```

- Loads prompts from `config_new.yml`
- Generates 20 images per category
- Applies darkening and blurring
- Saves output to:

```
/Video/SDXL_Output/
```

---

###  2. Using the FLUX model

This script uses **FLUX.1-dev**, a new image generation model.

```bash
python flux_run.py
```

- Loads prompts from `config_flux.yml`
- Quantizes and optimizes certain models for better performance
- Generates and saves processed images in:

```
/Video/Flux_Output/
```

---

##  Post-Processing Logic (`make_dark.py`)

Images are post-processed using two steps:

1. **Darkening**: Reduces brightness by a factor (e.g., 0.5).
2. **Blurring**: Applies Gaussian blur to smoothen the image.

Usage:

```python
from make_dark import darken_and_blur_image

processed = darken_and_blur_image(image, darkness_factor=0.5, blur_radius=5)
```

---

##  For Beginners

No deep ML knowledge needed! Just:

1. Set up your environment.
2. Update the config YAMLs with your desired prompts.
3. Run the script and explore generated images!

---

##  Notes

- The script assumes GPU support (for best performance).
- Use `torch_dtype=torch.float16` or `bfloat16` depending on your hardware.
- You can change the number of images, image size, or prompts in the scripts or config files.

---


