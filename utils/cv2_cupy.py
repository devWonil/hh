#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import cupy as cp
import numpy as np
import cv2

def cvtColor(src, code):
    """
    CuPy implementation of cv2.cvtColor function
    """
    if isinstance(src, np.ndarray):
        src = cp.asarray(src)
    
    if code == cv2.COLOR_BGR2GRAY:
        # BGR to Grayscale conversion
        weights = cp.array([0.114, 0.587, 0.299], dtype=cp.float32)
        return cp.dot(src, weights)
    
    elif code == cv2.COLOR_BGR2RGB:
        # BGR to RGB conversion
        return src[:, :, [2, 1, 0]]
    
    elif code == cv2.COLOR_RGB2BGR:
        # RGB to BGR conversion
        return src[:, :, [2, 1, 0]]
    
    elif code == cv2.COLOR_BGR2HSV:
        # BGR to HSV conversion
        src_float = src.astype(cp.float32) / 255.0
        b, g, r = src_float[:, :, 0], src_float[:, :, 1], src_float[:, :, 2]
        
        max_val = cp.maximum(cp.maximum(r, g), b)
        min_val = cp.minimum(cp.minimum(r, g), b)
        diff = max_val - min_val
        
        # Value
        v = max_val
        
        # Saturation
        s = cp.where(max_val == 0, 0, diff / max_val)
        
        # Hue
        h = cp.zeros_like(max_val)
        mask_r = (max_val == r) & (diff != 0)
        mask_g = (max_val == g) & (diff != 0)
        mask_b = (max_val == b) & (diff != 0)
        
        h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360
        h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360
        h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360
        
        h = h / 2  # OpenCV uses 0-179 for H
        s = s * 255
        v = v * 255
        
        return cp.stack([h, s, v], axis=2).astype(cp.uint8)
    elif code == cv2.COLOR_BGR2HLS:
        # Optimized BGR to HLS conversion
        src_float = src.astype(cp.float32) / 255.0
        b, g, r = src_float[:, :, 0], src_float[:, :, 1], src_float[:, :, 2]
        
        max_val = cp.maximum(cp.maximum(r, g), b)
        min_val = cp.minimum(cp.minimum(r, g), b)
        diff = max_val - min_val
        
        l = (max_val + min_val) * 0.5
        
        # Saturation: vectorized with cp.where
        s = cp.where(
            diff == 0,
            0,
            cp.where(
                l <= 0.5,
                diff / (max_val + min_val + 1e-8),
                diff / (2.0 - max_val - min_val + 1e-8)
            )
        )
        
        # Hue: vectorized with cp.where
        h = cp.zeros_like(max_val)
        mask = diff != 0
        h = cp.where(
            (max_val == r) & mask,
            ((g - b) / (diff + 1e-8)) % 6,
            h
        )
        h = cp.where(
            (max_val == g) & mask,
            ((b - r) / (diff + 1e-8)) + 2,
            h
        )
        h = cp.where(
            (max_val == b) & mask,
            ((r - g) / (diff + 1e-8)) + 4,
            h
        )
        h = (h * 60) % 360
        h = h / 2  # OpenCV uses 0-179 for H
        
        # Pre-allocate output and assign channels
        out = cp.empty(src.shape, dtype=cp.uint8)
        out[:, :, 0] = h.astype(cp.uint8)
        out[:, :, 1] = (l * 255).astype(cp.uint8)
        out[:, :, 2] = (s * 255).astype(cp.uint8)
        
        return out

    elif code == cv2.COLOR_HLS2BGR:
        # Optimized HLS to BGR conversion
        src_float = src.astype(cp.float32)
        h, l, s = src_float[:, :, 0], src_float[:, :, 1], src_float[:, :, 2]
        h = h * 2  # Convert back from 0-179 to 0-360
        l = l / 255.0
        s = s / 255.0
        # Calculate C, X, m
        c = cp.where(l <= 0.5, s * (2 * l), s * (2 - 2 * l))
        x = c * (1 - cp.abs((h / 60) % 2 - 1))
        m = l - c / 2
        # Pre-allocate output channels
        r_prime = cp.zeros_like(h)
        g_prime = cp.zeros_like(h)
        b_prime = cp.zeros_like(h)
        # Vectorized assignment for each hue sector
        mask1 = (h >= 0) & (h < 60)
        mask2 = (h >= 60) & (h < 120)
        mask3 = (h >= 120) & (h < 180)
        mask4 = (h >= 180) & (h < 240)
        mask5 = (h >= 240) & (h < 300)
        mask6 = (h >= 300) & (h < 360)
        r_prime = cp.where(mask1, c, r_prime)
        g_prime = cp.where(mask1, x, g_prime)
        g_prime = cp.where(mask2, c, g_prime)
        r_prime = cp.where(mask2, x, r_prime)
        g_prime = cp.where(mask3, c, g_prime)
        b_prime = cp.where(mask3, x, b_prime)
        b_prime = cp.where(mask4, c, b_prime)
        g_prime = cp.where(mask4, x, g_prime)
        b_prime = cp.where(mask5, c, b_prime)
        r_prime = cp.where(mask5, x, r_prime)
        r_prime = cp.where(mask6, c, r_prime)
        b_prime = cp.where(mask6, x, b_prime)
        # Final RGB values
        r = (r_prime + m) * 255
        g = (g_prime + m) * 255
        b = (b_prime + m) * 255
        # Pre-allocate output and assign channels
        out = cp.empty(src.shape, dtype=cp.uint8)
        out[:, :, 0] = b.astype(cp.uint8)
        out[:, :, 1] = g.astype(cp.uint8)
        out[:, :, 2] = r.astype(cp.uint8)
        return out
    
    else:
        raise ValueError(f"Unsupported color conversion code: {code}")
    
def warpPerspective(src, M, dsize, borderValue=0):
    """
    CuPy implementation of cv2.warpPerspective
    
    Args:
        src: Input image (CuPy array)
        M: 3x3 transformation matrix (CuPy array)
        dsize: Output image size (width, height)
        flags: Interpolation method
        borderMode: Border extrapolation method
        borderValue: Border value for constant border
    
    Returns:
        Warped image (CuPy array)
    """
    if isinstance(src, np.ndarray):
        src = cp.asarray(src)
    
    if isinstance(M, np.ndarray):
        M = cp.asarray(M)
    
    height, width = src.shape[:2]
    out_width, out_height = dsize
    
    # Create coordinate meshgrid for output image
    x_out, y_out = cp.meshgrid(cp.arange(out_width), cp.arange(out_height))
    ones = cp.ones_like(x_out)
    
    # Stack coordinates as homogeneous coordinates
    coords_out = cp.stack([x_out.flatten(), y_out.flatten(), ones.flatten()])
    
    # Apply inverse transformation
    M_inv = cp.linalg.inv(M)
    coords_in = M_inv @ coords_out
    
    # Convert from homogeneous to cartesian coordinates
    x_in = coords_in[0] / coords_in[2]
    y_in = coords_in[1] / coords_in[2]
    
    # Reshape back to image dimensions
    x_in = x_in.reshape(out_height, out_width)
    y_in = y_in.reshape(out_height, out_width)
    
    # Bilinear interpolation
    x_floor = cp.floor(x_in).astype(cp.int32)
    y_floor = cp.floor(y_in).astype(cp.int32)
    x_ceil = x_floor + 1
    y_ceil = y_floor + 1
    
    # Calculate interpolation weights
    dx = x_in - x_floor
    dy = y_in - y_floor
    
    # Create mask for valid coordinates
    valid_mask = (x_floor >= 0) & (x_ceil < width) & (y_floor >= 0) & (y_ceil < height)
    
    # Initialize output image
    if len(src.shape) == 3:
        output = cp.full((out_height, out_width, src.shape[2]), borderValue, dtype=src.dtype)
    else:
        output = cp.full((out_height, out_width), borderValue, dtype=src.dtype)
    
    # Apply bilinear interpolation only for valid coordinates
    if len(src.shape) == 3:
        for c in range(src.shape[2]):
            # Get pixel values at four corners
            top_left = src[y_floor, x_floor, c]
            top_right = src[y_floor, x_ceil, c]
            bottom_left = src[y_ceil, x_floor, c]
            bottom_right = src[y_ceil, x_ceil, c]
            
            # Bilinear interpolation
            top = top_left * (1 - dx) + top_right * dx
            bottom = bottom_left * (1 - dx) + bottom_right * dx
            interpolated = top * (1 - dy) + bottom * dy
            
            output[:, :, c] = cp.where(valid_mask, interpolated, borderValue)
    else:
        # Get pixel values at four corners
        top_left = src[y_floor, x_floor]
        top_right = src[y_floor, x_ceil]
        bottom_left = src[y_ceil, x_floor]
        bottom_right = src[y_ceil, x_ceil]
        
        # Bilinear interpolation
        top = top_left * (1 - dx) + top_right * dx
        bottom = bottom_left * (1 - dx) + bottom_right * dx
        interpolated = top * (1 - dy) + bottom * dy
        
        output = cp.where(valid_mask, interpolated, borderValue)
    
    return output