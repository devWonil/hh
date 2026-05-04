#!/usr/bin/env python3
#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import os
import sys
import argparse
import numpy as np
import cv2
import cupy as cp
import cupyx.scipy.ndimage as ndi
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to system path to import SULIVAN modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import coord_utils as coord
import config
from utils.sulivan_logger import add_callback

# Constants
COLOR_MAX = 255
CAMERA_W = 1920
CAMERA_H = 1080
PIXEL_W = config.PIXELS_WIDTH
PIXEL_H = config.PIXELS_HEIGHT

def cupy_gaussian_blur(image, ksize, sigmaX, sigmaY=None):    
    # Define sigmaY if sigmaY is None
    if sigmaY is None:
        sigmaY = sigmaX
    
    # Check kernel size
    if isinstance(ksize, int):
        ksize = (ksize, ksize)
    
    # Apply Gaussian filter
    blurred = ndi.gaussian_filter(image, sigma=(sigmaY, sigmaX), mode='reflect')
    
    return blurred

class MaskTester:
    def __init__(self, img_blank_path, img_src_path, camera_image_path, mask_table_dir, M_cp=None):
        """
        Initialize the MaskTester with necessary images and mask tables
        
        Args:
            img_blank_path (str): Path to image with nothing projected (blank background)
            img_src_path (str): Path to source image that is projected
            camera_image_path (str): Path to camera image with projection
            mask_table_dir (str): Directory containing mask table .npy files
            M_cp (numpy.ndarray, optional): Camera to projector transformation matrix. If None, uses config
        """
        print(f"Initializing MaskTester...")
        
        # Load images
        self.img_blank = cv2.imread(img_blank_path)
        if self.img_blank is None:
            raise FileNotFoundError(f"Could not read blank image: {img_blank_path}")
            
        self.img_src = cv2.imread(img_src_path)
        if self.img_src is None:
            raise FileNotFoundError(f"Could not read source image: {img_src_path}")
            
        self.camera_image = cv2.imread(camera_image_path)  
        if self.camera_image is None:
            raise FileNotFoundError(f"Could not read camera image: {camera_image_path}")
        
        # Set transformation matrix
        if M_cp is not None:
            coord.M_cp = M_cp
            coord.M_pc = np.linalg.inv(M_cp)
        
        # Load mask tables
        try:
            self.mask_table_b = cp.load(os.path.join(mask_table_dir, 'mask_table_b.npy'))
            self.mask_table_g = cp.load(os.path.join(mask_table_dir, 'mask_table_g.npy'))
            self.mask_table_r = cp.load(os.path.join(mask_table_dir, 'mask_table_r.npy'))
            print(f"Loaded mask tables from {mask_table_dir}")
        except Exception as e:
            print(f"Error loading mask tables: {e}")
            raise
        
        # Copy blank image to GPU
        self.img_blank_hls = cp.asarray(cv2.cvtColor(self.img_blank, cv2.COLOR_BGR2HLS))
        
        print("MaskTester initialized successfully")

    def apply_lut_b(self, x):
        return self.mask_table_b[x]

    def apply_lut_g(self, x):
        return self.mask_table_g[x]
        
    def apply_lut_r(self, x):
        return self.mask_table_r[x]
        
    def get_rgb_masked_image(self):
        """Apply RGB masking only"""
        # Process screen source image
        screen_transformed = cv2.warpPerspective(self.img_src, coord.M_pc, (CAMERA_W, CAMERA_H))
        screen_transformed_bgr = cp.asarray(cv2.split(screen_transformed))

        # Generate masks
        mask_b = cp.zeros((CAMERA_H, CAMERA_W, 3), dtype=cp.float32)
        mask_g = cp.zeros((CAMERA_H, CAMERA_W, 3), dtype=cp.float32)
        mask_r = cp.zeros((CAMERA_H, CAMERA_W, 3), dtype=cp.float32)
        
        mask_b[:, :, :] = self.apply_lut_b(screen_transformed_bgr[0])
        mask_g[:, :, :] = self.apply_lut_g(screen_transformed_bgr[1])
        mask_r[:, :, :] = self.apply_lut_r(screen_transformed_bgr[2])

        # Apply masks
        img_projected_gpu = cp.asarray(self.camera_image).astype(cp.float32)
        img_projected_gpu = img_projected_gpu - mask_b
        img_projected_gpu = img_projected_gpu - mask_g
        img_projected_gpu = img_projected_gpu - mask_r

        # Filter overflow
        img_projected_gpu[:,:,:] = cp.where(img_projected_gpu > 255, 255, img_projected_gpu)
        img_projected_gpu[:,:,:] = cp.where(img_projected_gpu < 0, 0, img_projected_gpu)
            
        # Return RGB masked image
        return img_projected_gpu.astype(cp.uint8).get()
        
    def get_hls_compensated_image(self):
        """Apply HLS compensation only"""
        # Process screen source image
        screen_transformed = cv2.warpPerspective(self.img_src, coord.M_pc, (CAMERA_W, CAMERA_H))
        
        # Generate masks for visualization only - we need the transformed image for HLS
        img_projected_hls = cp.asarray(cv2.cvtColor(self.camera_image, cv2.COLOR_BGR2HLS))
        mask_hls = cp.asarray(cv2.cvtColor(screen_transformed, cv2.COLOR_BGR2GRAY))
        mask_hls[:,:] = cp.where(mask_hls == 0, 0, 255)  # binarize

        # Generate a prediction of projected image for error calculation
        screen_transformed_bgr = cp.asarray(cv2.split(screen_transformed))
        
        mask_b = cp.zeros((CAMERA_H, CAMERA_W, 3), dtype=cp.float32)
        mask_g = cp.zeros((CAMERA_H, CAMERA_W, 3), dtype=cp.float32)
        mask_r = cp.zeros((CAMERA_H, CAMERA_W, 3), dtype=cp.float32)
        
        mask_b[:, :, :] = self.apply_lut_b(screen_transformed_bgr[0])
        mask_g[:, :, :] = self.apply_lut_g(screen_transformed_bgr[1])
        mask_r[:, :, :] = self.apply_lut_r(screen_transformed_bgr[2])
        
        # Generate a prediction of projected image
        img_generated = cp.asarray(self.img_blank)
        img_generated = img_generated + mask_b
        img_generated = img_generated + mask_g
        img_generated = img_generated + mask_r
        img_generated = cp.clip(img_generated, 0, 255).astype(cp.uint8)

        # Predicted image
        img_generated_gray = cv2.cvtColor(img_generated.get(), cv2.COLOR_BGR2GRAY)
        # Original image
        img_projected_gray = cv2.cvtColor(self.camera_image, cv2.COLOR_BGR2GRAY)

        # Calculate the difference between the predicted image and the original image
        error_gen_proj = cp.asarray(cv2.absdiff(img_generated_gray, img_projected_gray))

        # Don't apply mask when error_gen_proj is significant
        threshold = 50
        mask_hls[:,:] = cp.where(error_gen_proj > threshold, 0, mask_hls[:,:])

        # Post-processing
        mask_hls = cupy_gaussian_blur(mask_hls, (25, 25), 3)
        
        # Get L value from img_blank and apply
        img_projected_hls[:,:,1] = cp.where(mask_hls > 127, self.img_blank_hls[:,:,1], img_projected_hls[:,:,1])

        # Convert back to BGR
        return cv2.cvtColor(img_projected_hls.get(), cv2.COLOR_HLS2BGR)
        
    def get_combined_masked_image(self):
        """Apply both RGB masking and HLS compensation"""
        # Process screen source image
        screen_transformed = cv2.warpPerspective(self.img_src, coord.M_pc, (CAMERA_W, CAMERA_H))
        screen_transformed_bgr = cp.asarray(cv2.split(screen_transformed))

        # Generate masks
        mask_b = cp.zeros((CAMERA_H, CAMERA_W, 3), dtype=cp.float32)
        mask_g = cp.zeros((CAMERA_H, CAMERA_W, 3), dtype=cp.float32)
        mask_r = cp.zeros((CAMERA_H, CAMERA_W, 3), dtype=cp.float32)
        
        mask_b[:, :, :] = self.apply_lut_b(screen_transformed_bgr[0])
        mask_g[:, :, :] = self.apply_lut_g(screen_transformed_bgr[1])
        mask_r[:, :, :] = self.apply_lut_r(screen_transformed_bgr[2])

        # Apply RGB masks
        img_projected_gpu = cp.asarray(self.camera_image).astype(cp.float32)
        img_projected_gpu = img_projected_gpu - mask_b
        img_projected_gpu = img_projected_gpu - mask_g
        img_projected_gpu = img_projected_gpu - mask_r

        # Filter overflow
        img_projected_gpu[:,:,:] = cp.where(img_projected_gpu > 255, 255, img_projected_gpu)
        img_projected_gpu[:,:,:] = cp.where(img_projected_gpu < 0, 0, img_projected_gpu)
            
        # Get RGB masked image
        img_masked = img_projected_gpu.astype(cp.uint8).get()
        
        # Apply HLS compensation
        # Generate a prediction of projected image
        img_generated = cp.asarray(self.img_blank)
        img_generated = img_generated + mask_b
        img_generated = img_generated + mask_g
        img_generated = img_generated + mask_r
        img_generated = cp.clip(img_generated, 0, 255).astype(cp.uint8)

        # Predicted image
        img_generated_gray = cv2.cvtColor(img_generated.get(), cv2.COLOR_BGR2GRAY)
        # RGB masked image
        img_projected_gray = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)

        # Calculate the difference between the predicted image and the original image
        error_gen_proj = cp.asarray(cv2.absdiff(img_generated_gray, img_projected_gray))

        # HLS compensation
        img_projected_hls = cp.asarray(cv2.cvtColor(img_masked, cv2.COLOR_BGR2HLS))
        mask_hls = cp.asarray(cv2.cvtColor(screen_transformed, cv2.COLOR_BGR2GRAY))
        mask_hls[:,:] = cp.where(mask_hls == 0, 0, 255) # binarize

        # Don't apply mask when error_gen_proj is significant
        threshold = 50
        mask_hls[:,:] = cp.where(error_gen_proj > threshold, 0, mask_hls[:,:])

        # Post-processing
        mask_hls = cupy_gaussian_blur(mask_hls, (25, 25), 3)
        
        # Get L value from img_blank and apply
        img_projected_hls[:,:,1] = cp.where(mask_hls > 127, self.img_blank_hls[:,:,1], img_projected_hls[:,:,1])

        # Convert back to BGR
        return cv2.cvtColor(img_projected_hls.get(), cv2.COLOR_HLS2BGR)

    def test_masking(self, output_dir=None):
        """
        Test different masking approaches and save the results
        
        Args:
            output_dir (str, optional): Directory to save output images. If None, uses timestamp.
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"mask_test_{timestamp}"
            
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Testing masking approaches, results will be saved to {output_dir}...")
        
        # Apply different masking approaches
        rgb_masked = self.get_rgb_masked_image()
        hls_compensated = self.get_hls_compensated_image()
        combined_masked = self.get_combined_masked_image()
        
        # Save original images for reference
        cv2.imwrite(os.path.join(output_dir, "01_blank_image.png"), self.img_blank)
        cv2.imwrite(os.path.join(output_dir, "02_source_image.png"), self.img_src)
        cv2.imwrite(os.path.join(output_dir, "03_camera_image.png"), self.camera_image)
        
        # Save masked images
        cv2.imwrite(os.path.join(output_dir, "04_rgb_masked.png"), rgb_masked)
        cv2.imwrite(os.path.join(output_dir, "05_hls_compensated.png"), hls_compensated)
        cv2.imwrite(os.path.join(output_dir, "06_combined_masked.png"), combined_masked)
        
        # Also create a visualization image with all results side by side
        vis_height = CAMERA_H
        vis_width = CAMERA_W * 2
        visualization = np.zeros((vis_height * 2, vis_width, 3), dtype=np.uint8)
        
        # Original images
        visualization[0:vis_height, 0:CAMERA_W] = self.camera_image
        visualization[0:vis_height, CAMERA_W:vis_width] = self.img_blank
        
        # Masked images
        visualization[vis_height:vis_height*2, 0:CAMERA_W] = rgb_masked
        visualization[vis_height:vis_height*2, CAMERA_W:vis_width] = combined_masked
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(visualization, "Camera Image (With Projection)", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(visualization, "Blank Image (No Projection)", (CAMERA_W + 10, 30), font, 1, (255, 255, 255), 2) 
        cv2.putText(visualization, "RGB Masked Only", (10, vis_height + 30), font, 1, (255, 255, 255), 2)
        cv2.putText(visualization, "RGB + HLS Combined", (CAMERA_W + 10, vis_height + 30), font, 1, (255, 255, 255), 2)
        
        cv2.imwrite(os.path.join(output_dir, "07_comparison.png"), visualization)
        
        print(f"Masking tests completed. Results saved to {output_dir}")
        return {
            "blank_image": self.img_blank,
            "source_image": self.img_src,
            "camera_image": self.camera_image,
            "rgb_masked": rgb_masked,
            "hls_compensated": hls_compensated,
            "combined_masked": combined_masked,
            "comparison": visualization
        }

def main():
    parser = argparse.ArgumentParser(description="Test Masking Approaches for SULIVAN")
    parser.add_argument("--blank", required=True, help="Path to blank image (no projection)")
    parser.add_argument("--src", required=True, help="Path to source image (to be projected)")
    parser.add_argument("--camera", required=True, help="Path to camera image (with projection)")
    parser.add_argument("--masks", required=True, help="Directory containing mask table .npy files")
    parser.add_argument("--output", help="Directory to save output images (default: timestamped dir)")
    parser.add_argument("--display", action="store_true", help="Display results using matplotlib")
    
    args = parser.parse_args()
    
    try:
        tester = MaskTester(
            args.blank, 
            args.src, 
            args.camera,
            args.masks
        )
        
        results = tester.test_masking(args.output)
        
        if args.display:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(231)
            plt.title("Camera Image (With Projection)")
            plt.imshow(cv2.cvtColor(results["camera_image"], cv2.COLOR_BGR2RGB))
            
            plt.subplot(232)
            plt.title("Blank Image (No Projection)")
            plt.imshow(cv2.cvtColor(results["blank_image"], cv2.COLOR_BGR2RGB))
            
            plt.subplot(233)
            plt.title("Source Image (Projected)")
            plt.imshow(cv2.cvtColor(results["source_image"], cv2.COLOR_BGR2RGB))
            
            plt.subplot(234)
            plt.title("RGB Masked Only")
            plt.imshow(cv2.cvtColor(results["rgb_masked"], cv2.COLOR_BGR2RGB))
            
            plt.subplot(235)
            plt.title("HLS Compensated Only")
            plt.imshow(cv2.cvtColor(results["hls_compensated"], cv2.COLOR_BGR2RGB))
            
            plt.subplot(236)
            plt.title("RGB + HLS Combined")
            plt.imshow(cv2.cvtColor(results["combined_masked"], cv2.COLOR_BGR2RGB))
            
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
