#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndi
import cv2
import config

import utils.cv2_cupy as cv2_cupy
from utils import coord_utils as coord
from utils.sulivan_logger import get_log_dir

import datetime as dt
import logging

logger = logging.getLogger(__name__)

# Separate file logger for mask_projector raw data (handler added lazily)
mask_logger = logging.getLogger('mask_projector')
mask_logger.propagate = False
mask_logger.setLevel(logging.DEBUG)
_mask_logger_initialized = False


def _ensure_mask_logger():
    """Lazily attach a FileHandler once logging has been set up."""
    global _mask_logger_initialized
    if _mask_logger_initialized:
        return
    handler = logging.FileHandler(f'{get_log_dir()}/mask_projector.log', encoding='utf-8')
    handler.setFormatter(logging.Formatter('', datefmt='%Y-%m-%d %H:%M:%S'))
    mask_logger.addHandler(handler)
    _mask_logger_initialized = True

# Define the inverse of the camera projection matrix

COLOR_MAX = 255
COLOR_INTENSITY_STEPS = 16
REPETITIONS_PER_COLOR = 10
MAX_PATTERN_IMAGES = 30


CAMERA_W = 1920
CAMERA_H = 1080
PIXEL_W = config.PIXELS_WIDTH
PIXEL_H = config.PIXELS_HEIGHT

r = 10
STD_THRESHOLD = 8.0 # Standard deviation threshold for pattern validation

import time
import random

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

class MaskProjector():
    def __init__(self):
        _ensure_mask_logger()
        self.is_initialized = False
        self.img_blank = None
        self.img_blank_hls = None
        self.mask_table_b = None
        self.mask_table_g = None
        self.mask_table_r = None
        self.safe_zone = None
        self.safe_zone_gpu = None

        # Load pre-defined table if exists
        try:
            self.mask_table_b = cp.load(f'{get_log_dir()}/mask_table_b.npy')
            self.mask_table_g = cp.load(f'{get_log_dir()}/mask_table_g.npy')
            self.mask_table_r = cp.load(f'{get_log_dir()}/mask_table_r.npy')
            
            try:
                self.safe_zone = np.load(f'{get_log_dir()}/safe_zone.npy')
                # Prepare GPU version for getMaskedImage
                safe_zone_camera = cv2.warpPerspective(self.safe_zone, coord.M_pc, (CAMERA_W, CAMERA_H))
                self.safe_zone_gpu = cp.asarray(safe_zone_camera, dtype=cp.float32) / 255.0
                self.safe_zone_gpu = cp.where(self.safe_zone_gpu > 0.5, 1.0, 0.0)
                self.safe_zone_gpu = cp.expand_dims(self.safe_zone_gpu, axis=2)
                logger.info(f"Loaded safe_zone from disk")
            except Exception as e:
                logger.warning(f"Failed to load safe_zone: {e}")

            # Also need to load img_blank if we want to use it without recalibration
            # But for now, let's assume we need calibration or at least img_blank
            # If tables exist, we might be partially ready, but we need img_blank for HLS compensation
            # So we still need calibration or loading img_blank.
            # For safety, let's set initialized to False unless we load everything.
            # But the user wants explicit init button.
            logger.info(f"Loaded mask tables from disk")
        except:
            logger.info(f"No mask tables found on disk.")

    def calibrate(self, display_func, capture_func, set_show_default_bg_func):
        logger.info("Starting MaskProjector calibration...")
        self.is_initialized = False
        set_show_default_bg_func(False)

        # 1. Capture blank image
        logger.info("Capturing blank image...")
        display_func(None)
        time.sleep(0.4)
        img_blank = capture_func()
        if img_blank is None:
            logger.error("Failed to capture blank image")
            return False
            
        # Save base images
        cv2.imwrite(f"{get_log_dir()}/img_blank.png", img_blank)
        
        # Copy images to GPU
        self.img_blank = cp.asarray(img_blank, dtype=cp.float32)
        self.img_blank_hls = cv2_cupy.cvtColor(self.img_blank, cv2.COLOR_BGR2HLS)

        # 2. Find projection area
        # Warp blank image to projector space to find empty spots
        img_blank_pixel = cv2.warpPerspective(img_blank, coord.M_cp, (PIXEL_W, PIXEL_H))
        cv2.imwrite(f"{get_log_dir()}/find_proj_01_warped.png", img_blank_pixel)
        
        # Find dominant color area (assuming white/bright background)
        # Use DISPLAY_POINTS from config to limit the search area if needed, 
        # but M_cp warping already maps the screen area to PIXEL_W x PIXEL_H.
        # So we just need to find the "screen" area in the warped image which should be the whole image
        # if the calibration is perfect. However, there might be objects.
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_blank_pixel, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"{get_log_dir()}/find_proj_02_gray.png", gray)
        
        # Threshold to find bright areas (assuming screen is white)
        # Otsu's method automatically determines optimal threshold from the bimodal histogram
        otsu_thresh_val, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        logger.info(f"Otsu's threshold value: {otsu_thresh_val}")
        cv2.imwrite(f"{get_log_dir()}/find_proj_03_thresh.png", thresh)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Debug: Draw all contours
        debug_contours = img_blank_pixel.copy()
        cv2.drawContours(debug_contours, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(f"{get_log_dir()}/find_proj_04_contours.png", debug_contours)

        # Keep only the largest contour
        mask_largest = np.zeros_like(thresh)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask_largest, [largest_contour], -1, 255, -1)
        else:
            mask_largest = thresh.copy()

        # Create safe zone by eroding the thresholded area
        # This ensures we are inside the white area and have a margin from edges/objects
        margin = 30
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (margin, margin))
        safe_zone = cv2.erode(mask_largest, kernel, iterations=1)
        cv2.imwrite(f"{get_log_dir()}/find_proj_06_safe_zone.png", safe_zone)

        # Find bounding box of the safe zone to limit the grid search
        points = cv2.findNonZero(safe_zone)
        if points is None:
            logger.error("No valid projection area found after erosion")
            # Fallback to center if detection fails
            target_rect = (100, 100, PIXEL_W-200, PIXEL_H-200)
            # Make a fake safe zone for fallback
            safe_zone = np.zeros_like(thresh)
            cv2.rectangle(safe_zone, (100, 100), (PIXEL_W-100, PIXEL_H-100), 255, -1)
        else:
            x, y, w, h = cv2.boundingRect(points)
            target_rect = (x, y, w, h)
            
        tx, ty, tw, th = target_rect
        logger.info(f"Selected projection area bounds: {target_rect}")

        # Save safe_zone for masking
        self.safe_zone = safe_zone
        # np.save(f'{get_log_dir()}/safe_zone.npy', self.safe_zone)
        
        # Prepare GPU version for getMaskedImage
        safe_zone_camera = cv2.warpPerspective(self.safe_zone, coord.M_pc, (CAMERA_W, CAMERA_H))
        self.safe_zone_gpu = cp.asarray(safe_zone_camera, dtype=cp.float32) / 255.0
        self.safe_zone_gpu = cp.where(self.safe_zone_gpu > 0.5, 1.0, 0.0)
        self.safe_zone_gpu = cp.expand_dims(self.safe_zone_gpu, axis=2)

        # Debug: Draw selected rect
        debug_selected = img_blank_pixel.copy()
        cv2.rectangle(debug_selected, (tx, ty), (tx+tw, ty+th), (0, 0, 255), 3)
        cv2.imwrite(f"{get_log_dir()}/find_proj_05_selected.png", debug_selected)
        
        # 3. Generate and project patterns
        # COLOR_INTENSITY_STEPS+1 patterns for R, G, B
        # Pattern size
        pat_size = 100 # 100x100 pixels
        
        collected_data = {
            'r': [], 'g': [], 'b': []
        }
        
        # Define the 33 colors we need * 4 repetitions
        # Format: (channel_code, bgr_color, intensity_index)
        target_patterns = []
        for _ in range(REPETITIONS_PER_COLOR):
            for i in range(COLOR_INTENSITY_STEPS+1):
                val = int(255 * (i / COLOR_INTENSITY_STEPS))
                target_patterns.append(('r', (0, 0, val), i)) # BGR
                target_patterns.append(('g', (0, val, 0), i))
                target_patterns.append(('b', (val, 0, 0), i))
            
        # Shuffle patterns to be random
        random.shuffle(target_patterns)
            
        # Try to fit them in frames
        max_retries = MAX_PATTERN_IMAGES # Increased retries since we have more patterns
        current_patterns = target_patterns[:]
        
        for attempt in range(max_retries):
            if not current_patterns:
                break
            
            logger.info(f"Projection attempt {attempt+1}/{max_retries}")
                
            # Create a frame
            frame = np.zeros((PIXEL_H, PIXEL_W, 3), dtype=np.uint8)
            
            # Find all valid spots in the grid
            valid_spots = []
            
            # Grid inside target_rect
            cols = tw // (pat_size + 20)
            rows = th // (pat_size + 20)
            
            if cols <= 0 or rows <= 0:
                logger.error("Projection area too small")
                # Try to continue with whatever we have or fail?
                pass 
            else:
                for r_idx in range(rows):
                    for c_idx in range(cols):
                        px = tx + c_idx * (pat_size + 20) + 10
                        py = ty + r_idx * (pat_size + 20) + 10
                        
                        # Check if this spot is fully inside safe_zone
                        # We check the ROI in safe_zone
                        roi = safe_zone[py:py+pat_size, px:px+pat_size]
                        if roi.shape[0] != pat_size or roi.shape[1] != pat_size:
                            continue
                            
                        # If all pixels are 255 (white), it's valid.
                        if cv2.countNonZero(roi) == (pat_size * pat_size):
                            valid_spots.append((px, py))
            
            if not valid_spots:
                logger.warning("No valid spots found in this attempt")
                continue

            # Shuffle valid spots to ensure random placement
            random.shuffle(valid_spots)

            # Fill valid spots with patterns
            capacity = len(valid_spots)
            batch = current_patterns[:capacity]
            current_patterns = current_patterns[capacity:]
            
            placed_items = []
            
            for idx, (code, color, intensity_idx) in enumerate(batch):
                px, py = valid_spots[idx]
                
                cv2.rectangle(frame, (px, py), (px+pat_size, py+pat_size), color, -1)
                placed_items.append({
                    'rect': (px, py, px+pat_size, py+pat_size),
                    'code': code,
                    'color': color,
                    'intensity_idx': intensity_idx
                })
            
            # Project
            display_func(frame)
            time.sleep(0.4)
            img_captured = capture_func()
            
            if img_captured is None:
                logger.error("Failed to capture pattern image")
                # If capture failed, we should probably retry this batch?
                # For now, let's just add them back to current_patterns
                for item in placed_items:
                    current_patterns.append((item['code'], item['color'], item['intensity_idx']))
                continue
                
            # Warp captured image
            img_captured_pixel = cv2.warpPerspective(img_captured, coord.M_cp, (PIXEL_W, PIXEL_H))
            
            # Extract data
            failed_items = []
            
            for item in placed_items:
                x1, y1, x2, y2 = item['rect']
                
                # ROI extraction
                roi_captured = img_captured_pixel[y1:y2, x1:x2]
                roi_blank = img_blank_pixel[y1:y2, x1:x2]
                
                if roi_captured.size == 0 or roi_blank.size == 0:
                    failed_items.append((item['code'], item['color'], item['intensity_idx']))
                    continue

                # Calculate stats for captured image
                mean_captured, std_captured = cv2.meanStdDev(roi_captured)
                mean_captured = mean_captured.flatten()
                std_captured = std_captured.flatten()
                
                # Check standard deviation
                if np.max(std_captured) > STD_THRESHOLD:
                    logger.debug(f"Pattern rejected due to high STD: {std_captured}, Color: {item['code']}")
                    failed_items.append((item['code'], item['color'], item['intensity_idx']))
                    continue

                # Calculate stats for blank image (background)
                mean_blank, _ = cv2.meanStdDev(roi_blank)
                mean_blank = mean_blank.flatten()
                
                # Store
                collected_data[item['code']].append({
                    'src': item['color'],
                    'captured': mean_captured,
                    'blank': mean_blank,
                    'idx': item['intensity_idx']
                })
            
            # Re-queue failed items
            if failed_items:
                logger.debug(f"Re-queueing {len(failed_items)} failed patterns")
                current_patterns.extend(failed_items)
                random.shuffle(current_patterns)
                
        if current_patterns:
            logger.error("Could not project all patterns in attempts")
            return False
            
        # 4. Calculate mask constants
        logger.info("Calculating mask constants...")
        
        # Helper to average data for a channel
        def process_channel_data(channel_data):
            # Group by index
            grouped = {}
            for item in channel_data:
                idx = item['idx']
                if idx not in grouped:
                    grouped[idx] = {'captured': [], 'blank': [], 'src': item['src']}
                
                grouped[idx]['captured'].append(item['captured'])
                grouped[idx]['blank'].append(item['blank'])
            
            # Average and sort
            result = []
            for idx in sorted(grouped.keys()):
                avg_captured = np.mean(grouped[idx]['captured'], axis=0)
                avg_blank = np.mean(grouped[idx]['blank'], axis=0)
                result.append({
                    'idx': idx,
                    'src': grouped[idx]['src'],
                    'captured': avg_captured,
                    'blank': avg_blank
                })
            return result

        # Process each channel
        final_data_r = process_channel_data(collected_data['r'])
        final_data_g = process_channel_data(collected_data['g'])
        final_data_b = process_channel_data(collected_data['b'])
        
        # We need to ensure we have all points for each channel
        if len(final_data_r) != COLOR_INTENSITY_STEPS+1 \
            or len(final_data_g) != COLOR_INTENSITY_STEPS+1 \
            or len(final_data_b) != COLOR_INTENSITY_STEPS+1:
            logger.error(f"Insufficient data collected: R={len(final_data_r)}, G={len(final_data_g)}, B={len(final_data_b)}")
            return False

        # Prepare arrays for calculation
        masks_const_r = []
        masks_const_g = []
        masks_const_b = []
        
        colors_r_src = []
        colors_g_src = []
        colors_b_src = []
        
        for i in range(COLOR_INTENSITY_STEPS+1):
            # R channel
            data_r = final_data_r[i]
            masks_const_r.append(data_r['captured'] - data_r['blank'])
            colors_r_src.append(data_r['src'])
            
            # G channel
            data_g = final_data_g[i]
            masks_const_g.append(data_g['captured'] - data_g['blank'])
            colors_g_src.append(data_g['src'])
            
            # B channel
            data_b = final_data_b[i]
            masks_const_b.append(data_b['captured'] - data_b['blank'])
            colors_b_src.append(data_b['src'])

        masks_const_r = np.array(masks_const_r).astype(np.float32)
        masks_const_g = np.array(masks_const_g).astype(np.float32)
        masks_const_b = np.array(masks_const_b).astype(np.float32)

        # Offset masks
        masks_const_r[:] -= masks_const_r[0]
        masks_const_g[:] -= masks_const_g[0]
        masks_const_b[:] -= masks_const_b[0]
    
        # Step 4 : Create lookup table for each color
        mask_table_b = []
        mask_table_g = []
        mask_table_r = []

        screen_color_idx = 0
        for i in range(COLOR_MAX+1):
            if screen_color_idx < COLOR_INTENSITY_STEPS and i > colors_r_src[screen_color_idx+1][2]:
                screen_color_idx += 1
            
            # Interpolation
            # Note: colors_b_src[i][0] is Blue channel value. 
            # colors_r_src[i][2] is Red channel value.
            
            # Blue
            denom_b = (colors_b_src[screen_color_idx+1][0] - colors_b_src[screen_color_idx][0])
            if denom_b == 0: denom_b = 1
            color_b = (masks_const_b[screen_color_idx+1] - masks_const_b[screen_color_idx]) / denom_b * (i - colors_b_src[screen_color_idx][0]) + masks_const_b[screen_color_idx]
            
            # Green
            denom_g = (colors_g_src[screen_color_idx+1][1] - colors_g_src[screen_color_idx][1])
            if denom_g == 0: denom_g = 1
            color_g = (masks_const_g[screen_color_idx+1] - masks_const_g[screen_color_idx]) / denom_g * (i - colors_g_src[screen_color_idx][1]) + masks_const_g[screen_color_idx]
            
            # Red
            denom_r = (colors_r_src[screen_color_idx+1][2] - colors_r_src[screen_color_idx][2])
            if denom_r == 0: denom_r = 1
            color_r = (masks_const_r[screen_color_idx+1] - masks_const_r[screen_color_idx]) / denom_r * (i - colors_r_src[screen_color_idx][2]) + masks_const_r[screen_color_idx]
            
            mask_table_b.append(color_b)
            mask_table_g.append(color_g)
            mask_table_r.append(color_r)

        # Save LUTs
        self.mask_table_b = cp.asarray(mask_table_b)
        self.mask_table_g = cp.asarray(mask_table_g)
        self.mask_table_r = cp.asarray(mask_table_r)
        cp.save(f'{get_log_dir()}/mask_table_b.npy', self.mask_table_b)
        cp.save(f'{get_log_dir()}/mask_table_g.npy', self.mask_table_g)
        cp.save(f'{get_log_dir()}/mask_table_r.npy', self.mask_table_r)

        logger.info("MaskProjector calibration completed successfully.")
        self.is_initialized = True
        
        # Clear screen
        set_show_default_bg_func(True)
        return True
    
    def apply_lut_b(self, x):
        return self.mask_table_b[x, :]

    def apply_lut_g(self, x):
        return self.mask_table_g[x, :]

    def apply_lut_r(self, x):
        return self.mask_table_r[x, :]
    
    def getMaskedImage(self, camera_image, projector_image):
        if not self.is_initialized:
            return camera_image
            
        # Process screen source image

        screen_transformed = cv2_cupy.warpPerspective(projector_image, coord.M_pc, (CAMERA_W, CAMERA_H))
        # cv2.imwrite(f"{get_log_dir()}/img_src_transformed.png", screen_transformed)
        screen_transformed_bgr = cp.dsplit(screen_transformed, 3)
        screen_transformed_bgr = [cp.squeeze(c, axis=2) for c in screen_transformed_bgr]

        # Generate masks
        mask_b = cp.zeros((PIXEL_H, PIXEL_W, 3), dtype=cp.float32)
        mask_g = cp.zeros((PIXEL_H, PIXEL_W, 3), dtype=cp.float32)
        mask_r = cp.zeros((PIXEL_H, PIXEL_W, 3), dtype=cp.float32)
        mask_b[:, :, :] = self.apply_lut_b(screen_transformed_bgr[0])
        mask_g[:, :, :] = self.apply_lut_g(screen_transformed_bgr[1])
        mask_r[:, :, :] = self.apply_lut_r(screen_transformed_bgr[2])

        # Apply safe zone mask
        if self.safe_zone_gpu is not None:
            mask_b *= self.safe_zone_gpu
            mask_g *= self.safe_zone_gpu
            mask_r *= self.safe_zone_gpu

        # cv2.imwrite(f"{get_log_dir()}/img_mask.png", (mask_b + mask_g + mask_r).get())

        # Apply masks
        img_projected_gpu = cp.asarray(camera_image)
        img_projected_gpu = img_projected_gpu - mask_b
        img_projected_gpu = img_projected_gpu - mask_g
        img_projected_gpu = img_projected_gpu - mask_r

        # Filter overflow
        cp.clip(img_projected_gpu, 0, 255, out=img_projected_gpu)
            
        # Save masked image
        if config.use_rgb_masking:
            img_masked = img_projected_gpu.astype(cp.uint8)
        else:
            img_masked = camera_image

        if config.use_compensate_hls:
            # Generate a prediction of projected image
            img_generated = cp.asarray(self.img_blank)
            img_generated = img_generated + mask_b
            img_generated = img_generated + mask_g
            img_generated = img_generated + mask_r
            img_generated = cp.clip(img_generated, 0, 255)

            # Calculate error between generated and masked image
            error_gen_proj = img_generated - img_projected_gpu
            error_gen_proj_b, error_gen_proj_g, error_gen_proj_r = cp.split(error_gen_proj, 3, axis=2)
            error_gen_proj_b = error_gen_proj_b.squeeze()
            error_gen_proj_g = error_gen_proj_g.squeeze()
            error_gen_proj_r = error_gen_proj_r.squeeze()

            # HLS compensation
            img_projected_hls = cv2_cupy.cvtColor(img_masked, cv2.COLOR_BGR2HLS)
            mask_hls = cv2_cupy.cvtColor(screen_transformed, cv2.COLOR_BGR2GRAY)
            mask_hls[:,:] = cp.where(mask_hls == 0, 0, 255) # binarize

            # Don't apply mask when error_gen_proj is significant
            threshold = 50
            mask_hls[:,:] = cp.where(cp.abs(error_gen_proj_b) > threshold, 0, mask_hls[:,:])
            mask_hls[:,:] = cp.where(cp.abs(error_gen_proj_g) > threshold, 0, mask_hls[:,:])
            mask_hls[:,:] = cp.where(cp.abs(error_gen_proj_r) > threshold, 0, mask_hls[:,:])

            # Post-processing
            mask_hls = cupy_gaussian_blur(mask_hls, (25, 25), 3)
            # cv2.imwrite(f"{get_log_dir()}/img_mask_hsl.png", mask_hls.get())
            
            # Get V value from img_blank and apply
            img_projected_hls[:,:,1] = cp.where(mask_hls > 50,
                                                (self.img_blank_hls[:,:,1] * mask_hls + img_projected_hls[:,:,1] * (255-mask_hls)) / 255,
                                                img_projected_hls[:,:,1])
            img_projected_hls = cv2_cupy.cvtColor(img_projected_hls, cv2.COLOR_HLS2BGR)
            img_result = img_projected_hls.get()
        else:
            if not isinstance(img_masked, np.ndarray):
                img_result = img_masked.get()
            else:
                img_result = img_masked

        return img_result
