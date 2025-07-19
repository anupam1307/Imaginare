from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import warnings

# Ignore DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def darken_image(image, factor):
    """
    Darken the given PIL Image.

    Args:
    image (PIL.Image): Input PIL Image.
    factor (float): Darkness factor. Values between 0 and 1.
                    0 means no change, 1 means completely black.

    Returns:
    PIL.Image: Darkened PIL Image.
    """
    # Ensure the factor is within the valid range
    factor = max(0, min(factor, 1))

    # Create a copy of the image to work with
    darkened_image = image.copy()

    # Load pixel data
    pixels = darkened_image.load()

    # Iterate over each pixel and darken it
    for y in range(darkened_image.size[1]):
        for x in range(darkened_image.size[0]):
            # Get the pixel value at this location
            r, g, b = pixels[x, y]

            # Darken each channel
            r = int(r * factor)
            g = int(g * factor)
            b = int(b * factor)

            # Set the new pixel value
            pixels[x, y] = (r, g, b)

    return darkened_image

def darken_and_blur_image(image, darkness_factor, blur_radius):
    """
    Darken and blur the given PIL Image.

    Args:
    image (PIL.Image): Input PIL Image.
    darkness_factor (float): Darkness factor. Values between 0 and 1.
                             0 means no change, 1 means completely black.
    blur_radius (float): Radius of blur effect.

    Returns:
    PIL.Image: Darkened and blurred PIL Image.
    """
    # Ensure the factors are within the valid range
    darkness_factor = max(0, min(darkness_factor, 1))

    # Darken the image
    darkened_image = darken_image(image, darkness_factor)

    # Apply blur effect
    blurred_image = darkened_image.filter(ImageFilter.GaussianBlur(blur_radius))

    return blurred_image