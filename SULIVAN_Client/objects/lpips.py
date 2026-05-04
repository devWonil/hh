#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import cv2
import torch
import lpips
import numpy as np
import os
import pygame
from datetime import datetime
import logging

from utils.sulivan_logger import get_log_dir
import utils.coord_utils as coord

logger = logging.getLogger(__name__)
import config

import time
import threading
import concurrent.futures

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from client import SULIVAN_Client

class LPIPSModelLoader:
    _instance = None
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            logger.info("Loading LPIPS model...")
            # Using alexnet as it is faster and commonly used
            try:
                cls._model = lpips.LPIPS(net='alex')
                if torch.cuda.is_available():
                    cls._model.cuda()
                logger.info("LPIPS model loaded.")
            except Exception as e:
                logger.error(f"Failed to load LPIPS model: {e}")
                return None
        return cls._model

class LPIPSObject:
    def __init__(self, client: "SULIVAN_Client", name, bbox_world, reference_path, threshold, complete_callback, callback_arg,
                 autocheck_interval=None, autocheck_hand_free_interval=None, use_mask=True, mask_margin=None,
                 use_alignment=True, alignment_margin=None, alignment_method="multi_offset",
                 alignment_search_step=4, alignment_blur_sigma=0.5):
        self._client = client
        self.name = name
        self.step = client.scenario.current_step
        self.bbox_world = bbox_world
        self.bbox_pixel = coord.fromWorldToPixelBbox(self.bbox_world)
        self.reference_path = reference_path
        self.threshold = threshold
        self.complete_callback = complete_callback
        self.callback_arg = callback_arg
        self.autocheck_interval = autocheck_interval
        self.autocheck_hand_free_interval = autocheck_hand_free_interval
        self.use_mask = use_mask
        self.mask_margin = mask_margin if mask_margin is not None else [3, 3]
        self.last_check_time = time.time()
        self.is_processing = False
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"LPIPS-{name}")
        self._hand_free_start = None
        
        # Alignment parameters for projector-camera misalignment compensation
        self.use_alignment = use_alignment
        self.alignment_margin = alignment_margin if alignment_margin is not None else [1, 1]  # world coords (cm)
        self.alignment_method = alignment_method  # "ecc", "template", or "multi_offset"
        self.alignment_search_step = max(1, alignment_search_step)  # pixel step for multi_offset grid
        self.alignment_blur_sigma = alignment_blur_sigma  # Gaussian blur sigma (0 to disable)
        
        # Store images for VLM context
        self._reference_image_bgr = None  # BGR format for VLM
        self._last_comparison_image_bgr = None  # BGR format for VLM
        self._last_score = None  # Latest LPIPS score
        
        # Get model instance
        self.loss_fn = LPIPSModelLoader.get_model()
        
        # Blurred reference tensor (computed in _load_reference_image if blur_sigma > 0)
        self._reference_blurred_tensor = None
        
        # Store reference as RGB numpy for alignment
        self._reference_rgb_np = None  # Set in _load_reference_image
        
        # Load reference image
        self.reference_image = self._load_reference_image(reference_path)

    def _load_reference_image(self, path):
        if self.loss_fn is None:
            return None
            
        full_path = self._client.scenario.path_util.getPath(path)
        # lpips.load_image reads as RGB and returns numpy array
        try:
            img = lpips.load_image(full_path)
            
            # Store BGR version for VLM context
            self._reference_image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Store RGB numpy for alignment
            self._reference_rgb_np = img.copy()
            
            # Compute blurred reference tensor if blur is enabled
            if self.alignment_blur_sigma > 0:
                ks = max(3, int(np.ceil(self.alignment_blur_sigma * 3)) * 2 + 1)
                blurred_ref = cv2.GaussianBlur(img, (ks, ks), self.alignment_blur_sigma)
                self._reference_blurred_tensor = self._preprocess_image(blurred_ref)
            else:
                self._reference_blurred_tensor = None
            
            # Copy reference image to lpips log directory
            if config.record_video:
                self._save_reference_to_log_dir(full_path)
            
            return self._preprocess_image(img)
        except Exception as e:
            logger.error(f"Failed to load reference image {full_path}: {e}")
            return None

    def _save_reference_to_log_dir(self, full_path):
        '''Copy reference image to lpips log directory.'''
        try:
            log_dir = os.path.join(get_log_dir(), "lpips")
            os.makedirs(log_dir, exist_ok=True)
            
            # Save with step and name prefix for identification
            ref_filename = f"step{self.step}_{self.name}_reference.png"
            ref_save_path = os.path.join(log_dir, ref_filename)
            
            # Copy the reference image
            import shutil
            shutil.copy2(full_path, ref_save_path)
            logger.info(f"Saved reference image to {ref_save_path}")
        except Exception as e:
            logger.error(f"Failed to save reference image to log dir: {e}")

    def _preprocess_image(self, img):
        # img is numpy array (H, W, C) RGB
        tensor = lpips.im2tensor(img)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        return tensor

    def checkAutoTrigger(self):
        current_time = time.time()

        # check autocheck_interval
        if self.autocheck_interval is not None and self.autocheck_interval > 0:
            if (current_time - self.last_check_time) * 1000 >= self.autocheck_interval:
                self.last_check_time = current_time
                self.trigger(save_debug_image=True)
                return

        # check autocheck_hand_free_interval
        if self._should_trigger_with_hand_free(current_time):
            self.last_check_time = current_time
            self.trigger(save_debug_image=True)

    def _should_trigger_with_hand_free(self, current_time: float) -> bool:
        if self.autocheck_hand_free_interval is None or self.autocheck_hand_free_interval <= 0:
            return False

        hands = self._client.mp_thread.getHands()

        overlap_detected = False
        for hand in hands.copy():
            circle_info = hand.get('hand_circle')
            if circle_info is None:
                continue

            if self._is_circle_over_rect(circle_info['center'], circle_info['radius'], self.bbox_pixel):
                overlap_detected = True
                break

        if overlap_detected:
            self._hand_free_start = None
            return False

        if self._hand_free_start is None:
            self._hand_free_start = current_time
            return False

        elapsed_ms = (current_time - self._hand_free_start) * 1000
        if elapsed_ms >= self.autocheck_hand_free_interval:
            self._hand_free_start = current_time
            return True

        return False

    def get_vlm_context(self):
        """Get LPIPS image information for VLM context.
        
        Returns:
            dict: Contains reference_image (BGR), comparison_image (BGR), 
                  last_score, threshold, and name
        """
        return {
            "name": self.name,
            "reference_image": self._reference_image_bgr,
            "comparison_image": self._last_comparison_image_bgr,
            "last_score": self._last_score,
            "threshold": self.threshold,
        }

    def _align_to_reference(self, padded_crop_bgr, ref_rgb):
        """Align a padded camera crop to the reference image to compensate for
        projector-camera misalignment.
        
        Args:
            padded_crop_bgr: Camera crop (BGR) larger than reference, containing search margin.
            ref_rgb: Reference image (RGB numpy, H x W x C).
            
        Returns:
            Aligned RGB image with the same dimensions as ref_rgb.
        """
        ref_h, ref_w = ref_rgb.shape[:2]
        crop_h, crop_w = padded_crop_bgr.shape[:2]
        
        # If the padded crop is smaller than or equal to reference, no alignment possible
        if crop_h <= ref_h or crop_w <= ref_w:
            logger.debug(f"[{self.name}] Padded crop ({crop_w}x{crop_h}) not larger than reference "
                         f"({ref_w}x{ref_h}), skipping alignment.")
            crop_rgb = cv2.cvtColor(padded_crop_bgr, cv2.COLOR_BGR2RGB)
            return cv2.resize(crop_rgb, (ref_w, ref_h))
        
        padded_crop_rgb = cv2.cvtColor(padded_crop_bgr, cv2.COLOR_BGR2RGB)
        
        aligned = None
        method_used = "none"
        
        # Try ECC alignment
        if self.alignment_method == "ecc":
            aligned, method_used = self._align_ecc(padded_crop_rgb, ref_rgb)
            # Fallback to template matching if ECC fails
            if aligned is None:
                logger.debug(f"[{self.name}] ECC failed, falling back to template matching.")
                aligned, method_used = self._align_template(padded_crop_rgb, ref_rgb)
        elif self.alignment_method == "template":
            aligned, method_used = self._align_template(padded_crop_rgb, ref_rgb)
        else:
            logger.warning(f"[{self.name}] Unknown alignment method '{self.alignment_method}', using template.")
            aligned, method_used = self._align_template(padded_crop_rgb, ref_rgb)
        
        # Final fallback: center crop
        if aligned is None:
            logger.warning(f"[{self.name}] All alignment methods failed, using center crop.")
            aligned = self._center_crop(padded_crop_rgb, ref_w, ref_h)
            method_used = "center_crop"
        
        logger.debug(f"[{self.name}] Alignment done using '{method_used}'.")
        return aligned
    
    def _align_ecc(self, padded_crop_rgb, ref_rgb):
        """Align using Enhanced Correlation Coefficient (ECC) maximization.
        Handles translation + rotation (MOTION_EUCLIDEAN).
        
        Returns:
            Tuple of (aligned_rgb_image, method_name) or (None, None) on failure.
        """
        ref_h, ref_w = ref_rgb.shape[:2]
        
        try:
            # Convert to grayscale
            ref_gray = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY)
            crop_gray = cv2.cvtColor(padded_crop_rgb, cv2.COLOR_RGB2GRAY)
            
            # Resize crop_gray to ref size for ECC (ECC requires same-size inputs)
            # Instead, we use template matching to get initial offset, then refine with ECC
            # on a same-sized region.
            
            # Step 1: Get rough offset via template matching
            result = cv2.matchTemplate(crop_gray, ref_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            ox, oy = max_loc  # top-left of best match
            
            # Extract the matching region from the padded crop
            matched_region = padded_crop_rgb[oy:oy+ref_h, ox:ox+ref_w]
            matched_gray = crop_gray[oy:oy+ref_h, ox:ox+ref_w]
            
            if matched_region.shape[0] != ref_h or matched_region.shape[1] != ref_w:
                return None, None
            
            # Step 2: Refine with ECC on the matched region
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
            
            _, warp_matrix = cv2.findTransformECC(
                ref_gray, matched_gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria
            )
            
            # Apply warp to the matched region
            aligned = cv2.warpAffine(
                matched_region, warp_matrix, (ref_w, ref_h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT_101
            )
            
            tx, ty = warp_matrix[0, 2], warp_matrix[1, 2]
            angle = np.degrees(np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0]))
            logger.debug(f"[{self.name}] ECC alignment: template_offset=({ox},{oy}), "
                         f"ecc_refine=(tx={tx:.2f}, ty={ty:.2f}, angle={angle:.3f}°), "
                         f"match_score={max_val:.4f}")
            
            return aligned, "ecc"
        except cv2.error as e:
            logger.debug(f"[{self.name}] ECC alignment failed: {e}")
            return None, None
        except Exception as e:
            logger.debug(f"[{self.name}] ECC alignment unexpected error: {e}")
            return None, None
    
    def _align_template(self, padded_crop_rgb, ref_rgb):
        """Align using template matching (translation only).
        
        Returns:
            Tuple of (aligned_rgb_image, method_name) or (None, None) on failure.
        """
        ref_h, ref_w = ref_rgb.shape[:2]
        
        try:
            ref_gray = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY)
            crop_gray = cv2.cvtColor(padded_crop_rgb, cv2.COLOR_RGB2GRAY)
            
            result = cv2.matchTemplate(crop_gray, ref_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            ox, oy = max_loc
            
            aligned = padded_crop_rgb[oy:oy+ref_h, ox:ox+ref_w]
            
            if aligned.shape[0] != ref_h or aligned.shape[1] != ref_w:
                logger.debug(f"[{self.name}] Template match crop size mismatch.")
                return None, None
            
            logger.debug(f"[{self.name}] Template alignment: offset=({ox},{oy}), "
                         f"match_score={max_val:.4f}")
            
            return aligned, "template"
        except Exception as e:
            logger.debug(f"[{self.name}] Template alignment failed: {e}")
            return None, None
    
    def _compute_multi_offset_lpips(self, padded_crop_bgr, inner_offset, inner_size, ref_h, ref_w):
        """Compute LPIPS at multiple spatial offsets within a padded crop
        and return the minimum score with the corresponding aligned image.
        
        Instead of relying on feature/template matching to find the correct
        alignment, this method exhaustively searches over a grid of spatial
        offsets and directly optimizes for the minimum LPIPS distance.
        
        Args:
            padded_crop_bgr: Padded camera crop (BGR), larger than inner region.
            inner_offset: (x, y) position of the original bbox crop within padded crop.
            inner_size: (w, h) size of the original (unpadded) crop region.
            ref_h, ref_w: Reference tensor spatial dimensions.
            
        Returns:
            (min_dist_val, best_crop_rgb) or (None, None) on failure.
        """
        pad_h, pad_w = padded_crop_bgr.shape[:2]
        inner_w, inner_h = inner_size
        
        max_x = pad_w - inner_w
        max_y = pad_h - inner_h
        
        if max_x < 0 or max_y < 0:
            logger.debug(f"[{self.name}] Multi-offset: padded crop too small for search")
            return None, None
        
        step = self.alignment_search_step
        
        # Build offset grid
        x_positions = list(range(0, max_x + 1, step))
        if max_x > 0 and max_x not in x_positions:
            x_positions.append(max_x)
        y_positions = list(range(0, max_y + 1, step))
        if max_y > 0 and max_y not in y_positions:
            y_positions.append(max_y)
        
        # Always include the original center position
        cx, cy = inner_offset
        cx = max(0, min(cx, max_x))
        cy = max(0, min(cy, max_y))
        if cx not in x_positions:
            x_positions.append(cx)
            x_positions.sort()
        if cy not in y_positions:
            y_positions.append(cy)
            y_positions.sort()
        
        apply_blur = self.alignment_blur_sigma > 0
        blur_ks = max(3, int(np.ceil(self.alignment_blur_sigma * 3)) * 2 + 1) if apply_blur else 0
        
        # Generate all candidate crops and convert to tensors
        candidates_rgb = []
        positions = []
        tensors = []
        center_idx = -1
        
        for oy in y_positions:
            for ox in x_positions:
                cand = padded_crop_bgr[oy:oy+inner_h, ox:ox+inner_w]
                if cand.shape[0] != inner_h or cand.shape[1] != inner_w:
                    continue
                
                cand_rgb = cv2.cvtColor(cand, cv2.COLOR_BGR2RGB)
                cand_resized = cv2.resize(cand_rgb, (ref_w, ref_h))
                
                if apply_blur:
                    cand_for_lpips = cv2.GaussianBlur(cand_resized, (blur_ks, blur_ks), self.alignment_blur_sigma)
                else:
                    cand_for_lpips = cand_resized
                
                t = lpips.im2tensor(cand_for_lpips)
                tensors.append(t)
                candidates_rgb.append(cand_resized)  # Store un-blurred for display/VLM
                
                if ox == cx and oy == cy:
                    center_idx = len(positions)
                positions.append((ox, oy))
        
        if not tensors:
            return None, None
        
        n = len(tensors)
        logger.debug(f"[{self.name}] Multi-offset: {n} candidates "
                     f"(grid {len(x_positions)}x{len(y_positions)}, step={step}px)")
        
        # Batch LPIPS computation
        batch = torch.cat(tensors, dim=0)
        if torch.cuda.is_available():
            batch = batch.cuda()
        
        # Use blurred reference if blur is enabled
        ref_t = (self._reference_blurred_tensor
                 if (apply_blur and self._reference_blurred_tensor is not None)
                 else self.reference_image)
        
        # Process in chunks to avoid GPU OOM
        MAX_BATCH = 64
        all_dists = []
        for i in range(0, n, MAX_BATCH):
            chunk = batch[i:i+MAX_BATCH]
            chunk_size = chunk.shape[0]
            ref_chunk = ref_t.expand(chunk_size, -1, -1, -1)
            with torch.no_grad():
                d = self.loss_fn(ref_chunk, chunk)
            all_dists.append(d.cpu().numpy().flatten())
        
        dists_np = np.concatenate(all_dists)
        min_idx = int(np.argmin(dists_np))
        min_dist = float(dists_np[min_idx])
        best_pos = positions[min_idx]
        best_crop = candidates_rgb[min_idx]
        
        center_dist = float(dists_np[center_idx]) if center_idx >= 0 else -1.0
        improvement = center_dist - min_dist if center_dist >= 0 else 0.0
        
        logger.debug(f"[{self.name}] Multi-offset result: best_offset=({best_pos[0]},{best_pos[1]}), "
                     f"min_dist={min_dist:.4f}, center_dist={center_dist:.4f}, "
                     f"improvement={improvement:.4f}")
        
        return min_dist, best_crop

    @staticmethod
    def _center_crop(image_rgb, target_w, target_h):
        """Center-crop an image to the target size."""
        h, w = image_rgb.shape[:2]
        x_start = max(0, (w - target_w) // 2)
        y_start = max(0, (h - target_h) // 2)
        cropped = image_rgb[y_start:y_start+target_h, x_start:x_start+target_w]
        # In case of rounding issues, resize to exact target
        if cropped.shape[0] != target_h or cropped.shape[1] != target_w:
            cropped = cv2.resize(cropped, (target_w, target_h))
        return cropped

    @staticmethod
    def _is_circle_over_rect(center, radius, rect_bbox) -> bool:
        cx, cy = center
        rx, ry, rw, rh = rect_bbox
        nearest_x = min(max(cx, rx), rx + rw)
        nearest_y = min(max(cy, ry), ry + rh)
        dx = cx - nearest_x
        dy = cy - nearest_y
        return dx * dx + dy * dy <= radius * radius

    def trigger(self, save_debug_image):
        if self.is_processing:
            return

        if self.loss_fn is None or self.reference_image is None:
            logger.error(f"LPIPS object {self.name} is not properly initialized.")
            return
        
        self.is_processing = True
        self._executor.submit(self._trigger_process, save_debug_image)

    def _trigger_process(self, save_debug_image=True):
        try:
            # Mask the area and capture frame
            
            # Expand mask by margin
            margin_w = round(coord.fromWorldToPixelWidth(self.mask_margin[0]))
            margin_h = round(coord.fromWorldToPixelHeight(self.mask_margin[1]))
            
            mask_bbox = [
                self.bbox_pixel[0] - margin_w,
                self.bbox_pixel[1] - margin_h,
                self.bbox_pixel[2] + margin_w * 2,
                self.bbox_pixel[3] + margin_h * 2
            ]
            
            if self.use_mask:
                self._client.addBlackMask(mask_bbox)
                # self._client.updateScreen() # Handled by main loop
                time.sleep(0.5)
            
            # Try to get doubled resolution image first for better LPIPS comparison
            use_doubled = self._client.camera_thread.isDoubledResolutionEnabled()
            if use_doubled:
                frame = self._client.camera_thread.getDoubledImage()
                if frame is None:
                    # Fallback to normal image
                    frame = self._client.camera_thread.getImage()
                    use_doubled = False
            else:
                frame = self._client.camera_thread.getImage()
            
            if self.use_mask:
                self._client.removeBlackMask(mask_bbox)
            
            if frame is None:
                logger.warning("Camera frame is None, skipping LPIPS check.")
                return

            # Crop frame based on bbox
            # bbox is in World coords. Convert to Camera coords.
            # World -> Pixel -> Camera
            # bbox_pixel is already calculated above
            bbox_camera = coord.fromPixelToCameraBbox(self.bbox_pixel)
            
            # If using doubled resolution, scale bbox coordinates by 2
            if use_doubled:
                bbox_camera = [int(c * 2) for c in bbox_camera]
            
            x, y, w, h = bbox_camera
            
            # Ensure bbox is within frame boundaries
            h_img, w_img = frame.shape[:2]
            
            # Handle negative coordinates or out of bounds
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w_img, x + w)
            y2 = min(h_img, y + h)
            
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid crop area for LPIPS {self.name}: {bbox_camera} -> [{x1}:{x2}, {y1}:{y2}]")
                return

            crop = frame[y1:y2, x1:x2]
            
            ref_h, ref_w = self.reference_image.shape[2], self.reference_image.shape[3]
            
            dist_val = None  # May be set directly by multi_offset alignment
            
            # --- Image alignment for projector-camera misalignment compensation ---
            if self.use_alignment and self._reference_rgb_np is not None:
                # Expand bbox_camera by alignment_margin to create a search region
                align_margin_w_world = self.alignment_margin[0]
                align_margin_h_world = self.alignment_margin[1]
                align_margin_w_px = round(coord.fromWorldToPixelWidth(align_margin_w_world))
                align_margin_h_px = round(coord.fromWorldToPixelHeight(align_margin_h_world))
                
                # Convert pixel margin to camera coords
                align_cam_bbox = coord.fromPixelToCameraBbox([
                    self.bbox_pixel[0] - align_margin_w_px,
                    self.bbox_pixel[1] - align_margin_h_px,
                    self.bbox_pixel[2] + align_margin_w_px * 2,
                    self.bbox_pixel[3] + align_margin_h_px * 2
                ])
                
                if use_doubled:
                    align_cam_bbox = [int(c * 2) for c in align_cam_bbox]
                
                ax, ay, aw, ah = align_cam_bbox
                ax1 = max(0, ax)
                ay1 = max(0, ay)
                ax2 = min(w_img, ax + aw)
                ay2 = min(h_img, ay + ah)
                
                padded_crop = frame[ay1:ay2, ax1:ax2]
                
                if self.alignment_method == "multi_offset":
                    # Multi-offset: compute LPIPS at multiple spatial offsets,
                    # directly optimizing for minimum perceptual distance
                    inner_offset = (x1 - ax1, y1 - ay1)
                    inner_size = (x2 - x1, y2 - y1)
                    result = self._compute_multi_offset_lpips(
                        padded_crop, inner_offset, inner_size, ref_h, ref_w)
                    if result is not None:
                        dist_val, crop_resized = result
                    else:
                        # Fallback: use center crop without alignment
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        crop_resized = cv2.resize(crop_rgb, (ref_w, ref_h))
                else:
                    # ECC or template alignment
                    expected_inner_w = x2 - x1
                    expected_inner_h = y2 - y1
                    ref_for_align = cv2.resize(self._reference_rgb_np, (expected_inner_w, expected_inner_h))
                    
                    crop_resized = self._align_to_reference(padded_crop, ref_for_align)
                    crop_resized = cv2.resize(crop_resized, (ref_w, ref_h))
            else:
                # Original behavior: convert BGR→RGB, resize to reference size
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_resized = cv2.resize(crop_rgb, (ref_w, ref_h))
            
            # Store comparison image (BGR) for VLM context
            self._last_comparison_image_bgr = cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR)
            
            # Compute LPIPS distance if not already done by multi_offset
            try:
                if dist_val is None:
                    # Apply Gaussian blur if enabled (reduces sensitivity to small shifts)
                    if self.alignment_blur_sigma > 0:
                        ks = max(3, int(np.ceil(self.alignment_blur_sigma * 3)) * 2 + 1)
                        crop_for_lpips = cv2.GaussianBlur(crop_resized, (ks, ks), self.alignment_blur_sigma)
                        ref_tensor = (self._reference_blurred_tensor
                                      if self._reference_blurred_tensor is not None
                                      else self.reference_image)
                    else:
                        crop_for_lpips = crop_resized
                        ref_tensor = self.reference_image
                    
                    crop_tensor = self._preprocess_image(crop_for_lpips)
                    
                    with torch.no_grad():
                        dist = self.loss_fn(ref_tensor, crop_tensor)
                    dist_val = dist.item()
                
                self._last_score = dist_val  # Store score for VLM context
                logger.debug(f"LPIPS distance for {self.name}: {dist_val:.4f} (Threshold: {self.threshold})")
                
                # Save image for debugging
                if save_debug_image and config.record_video:
                    try:
                        log_dir = os.path.join(get_log_dir(), "lpips")
                        os.makedirs(log_dir, exist_ok=True)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"step{self.step}_{self.name}_{timestamp}_{dist_val:.4f}.png"
                        save_path = os.path.join(log_dir, filename)
                        
                        # crop_resized is RGB, convert back to BGR for cv2.imwrite
                        cv2.imwrite(save_path, cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR))
                        
                        # Record to train recorder
                        if getattr(self._client.scenario, 'current_step', -1) == self.step:
                            self._client.scenario.recordAction(f"LPIPS comparison image saved: {filename}, score: ({dist_val:.4f}/{self.threshold})")
                    except Exception as e:
                        logger.error(f"Failed to save LPIPS image: {e}")

                if dist_val < self.threshold:
                    logger.info(f"LPIPS check passed for {self.name}")
                    if getattr(self._client.scenario, 'current_step', -1) == self.step:
                        self._client.scenario.recordAction(f"LPIPS check passed for {self.name} (dist: {dist_val:.4f})")
                        self.complete_callback(self.callback_arg)
            except Exception as e:
                logger.error(f"Error during LPIPS comparison for {self.name}: {e}")
        except Exception as e:
            logger.error(f"Error in LPIPS trigger process: {e}")
        finally:
            self.is_processing = False

    def capture_reference(self):
        # Mask the area and capture frame
        
        # Expand mask by margin
        margin_w = round(coord.fromWorldToPixelWidth(self.mask_margin[0]))
        margin_h = round(coord.fromWorldToPixelHeight(self.mask_margin[1]))
        
        mask_bbox = [
            self.bbox_pixel[0] - margin_w,
            self.bbox_pixel[1] - margin_h,
            self.bbox_pixel[2] + margin_w * 2,
            self.bbox_pixel[3] + margin_h * 2
        ]
        
        if self.use_mask:
            self._client.addBlackMask(mask_bbox)
            # self._client.updateScreen() # Handled by main loop
            time.sleep(0.5)
        
        # Try to get doubled resolution image first for better quality reference
        use_doubled = self._client.camera_thread.isDoubledResolutionEnabled()
        if use_doubled:
            frame = self._client.camera_thread.getDoubledImage()
            if frame is None:
                # Fallback to normal image
                frame = self._client.camera_thread.getImage()
                use_doubled = False
        else:
            frame = self._client.camera_thread.getImage()
        
        if self.use_mask:
            self._client.removeBlackMask(mask_bbox)
        
        if frame is None:
            logger.warning("Camera frame is None, cannot capture reference.")
            return False

        # Crop frame based on bbox
        # bbox_pixel is already calculated above
        bbox_camera = coord.fromPixelToCameraBbox(self.bbox_pixel)
        
        # If using doubled resolution, scale bbox coordinates by 2
        if use_doubled:
            bbox_camera = [int(c * 2) for c in bbox_camera]
        
        x, y, w, h = bbox_camera
        
        # Ensure bbox is within frame boundaries
        h_img, w_img = frame.shape[:2]
        
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Invalid crop area for LPIPS {self.name}: {bbox_camera}")
            return False

        crop = frame[y1:y2, x1:x2]
        
        # Save to reference_path
        full_path = self._client.scenario.path_util.getPath(self.reference_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        try:
            cv2.imwrite(full_path, crop)
            logger.info(f"Saved reference image for {self.name} to {full_path}")
            
            # Reload reference image
            self.reference_image = self._load_reference_image(self.reference_path)
            return True
        except Exception as e:
            logger.error(f"Failed to save reference image: {e}")
            return False
