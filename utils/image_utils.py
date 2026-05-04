#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

from PIL import Image, ImageDraw
import cv2
import numpy as np

def circleMask(input):
    # Open input image
    image = Image.open(input).convert("RGBA")
    width, height = image.size

    # Create a new image
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # Draw a circle on center (Based on width)
    center = width // 2
    radius = width // 2
    draw.ellipse((center - radius, center - radius, 
                  center + radius, center + radius), fill=255)

    # Apply mask into image
    result = Image.new("RGBA", (width, height))
    result.paste(image, (0, 0), mask=mask)

    # Return result
    return result

def overlay_black_background_image(background, overlay):
    """Overlay non-black pixels from overlay onto background.
    
    Uses NumPy boolean indexing instead of multiple cv2 operations
    to minimize temporary array allocations (6 → 1 copy).
    """
    mask = np.any(overlay != 0, axis=2)
    result = background.copy()
    result[mask] = overlay[mask]
    return result