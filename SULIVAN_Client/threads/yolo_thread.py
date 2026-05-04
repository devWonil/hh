#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import cv2
import math
import numpy as np
import threading

from ultralytics import YOLO

from ultralytics.utils.plotting import Annotator, colors

from collections import defaultdict

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from client import SULIVAN_Client

import config
import utils.state as STATE
import utils.coord_utils as coord
import logging

logger = logging.getLogger(__name__)


class YOLOThread(threading.Thread):
    def __init__(self, client: "SULIVAN_Client", get_image_func=None, filter=False):
        self._client = client
        self.get_image_func = get_image_func
        self.filter_yolo = filter
        self.ready = False
        self._ready_event = threading.Event()
        self.mode = STATE.STOPPED

        self._stop_event = threading.Event()
        super().__init__()
        self.name = "YOLO"
        self.daemon = True

        self.scale_factor = 0.5

        self.yolo_threshold = 0.1

        # Choose model
        self.use_obb = getattr(config, 'use_obb', False)
        model_path = config.filename_yolo_obb if self.use_obb else config.filename_yolo
        self.model = YOLO(model_path)
        if self.filter_yolo:
            self.model_screen = YOLO(model_path)

        self.track_history = defaultdict(lambda: [])

        # Result values
        self.names = self.model.model.names
        self.indexes = {v:k for k,v in self.names.items()} # https://blog.naver.com/wideeyed/222007663089
        
        # logger.info("YOLO classes:", self.names)

        self.yolo_returns = []
        self.yolo_overlay = np.zeros((config.PIXELS_HEIGHT, config.PIXELS_WIDTH, 3), dtype=np.uint8)
        self._data_lock = threading.Lock()


    def stop(self):
        self._stop_event.set()

    def getOverlayImage(self):
        '''Get the image with detection result overlays'''
        with self._data_lock:
            return self.yolo_overlay.copy()

    # Return Bounding boxes
    def getDetection(self):
        '''Get the detection results'''
        with self._data_lock:
            return self.yolo_returns
    
    def getClassName(self, id):
        return self.names[id]
        
    def getClassId(self, name):
        return self.indexes[name]
    
    def setMode(self, mode):
        if mode == STATE.PAUSED or mode == STATE.RUNNING:
            self.mode = mode

    def cleanTrackHistory(self):
        self.track_history = defaultdict(lambda: [])

    def run(self):
        '''Thread main loop'''
        # Matrices for perspective xform and downscale
        downscale_matrix = np.diag([self.scale_factor, self.scale_factor, 1.0])
        M_cp_yolo = downscale_matrix @ config.M_cp # Downscale the image

        while True:
            if self._stop_event.is_set():
                break

            # Get a raw image (for YOLO input)
            if self.get_image_func is not None:
                camera_image = self.get_image_func()
            else:
                camera_image = self._client.camera_thread.getImage()
            if camera_image is None:
                self._stop_event.wait(0.005)
                continue
            
            # Get a mediapipe-annotated image (for annotation)
            overlay_image = np.zeros_like(camera_image, dtype=np.uint8)
            
            if self.mode != STATE.PAUSED:
                # Preprocess camera image
                resized_size = (round(config.PIXELS_WIDTH * self.scale_factor), round(config.PIXELS_HEIGHT * self.scale_factor))
                yolo_process_image = cv2.warpPerspective(camera_image, M_cp_yolo, resized_size)

                if self.filter_yolo:
                    # Preprocess screen image
                    screen_image = self._client.getDisplayImage()
                    screen_image = cv2.resize(screen_image, dsize=resized_size, interpolation=cv2.INTER_LINEAR)

                # Get results from YOLO
                yolo_results = self.model.track(yolo_process_image, persist=True, verbose=False, conf=self.yolo_threshold)

                if self.filter_yolo:
                    yolo_screen_results = self.model_screen.track(screen_image, persist=True, verbose=False, conf=self.yolo_threshold)

                yolo_returns = []

                # ===== Extract results into unified OBB format =====
                # Both OBB and AABB are normalized to:
                #   xywhr_list: [[cx, cy, w, h, angle_rad], ...]
                #   corners_list: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ...]
                has_detections = False
                # OBB mode
                if self.use_obb:
                    det_res = yolo_results[0].obb
                    if det_res is not None and det_res.id is not None:
                        has_detections = True
                        clss = det_res.cls.cpu().tolist()
                        track_ids = det_res.id.int().cpu().tolist()
                        confs = det_res.conf.float().cpu().tolist()
                        xywhr_list = det_res.xywhr.cpu().tolist()
                        corners_list = det_res.xyxyxyxy.cpu().tolist()
                # AABB mode
                else:
                    det_res = yolo_results[0].boxes
                    if det_res.id is not None:
                        has_detections = True
                        clss = det_res.cls.cpu().tolist()
                        track_ids = det_res.id.int().cpu().tolist()
                        confs = det_res.conf.float().cpu().tolist()
                        xywhr_list = []
                        corners_list = []
                        for box in det_res.xyxy.cpu().tolist():
                            x1, y1, x2, y2 = box
                            xywhr_list.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, 0.0])
                            corners_list.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

                # Extract screen detections for filtering (same unified format)
                if self.filter_yolo:
                    clss_screen = []
                    screen_xywhr = []
                    if self.use_obb:
                        scr_res = yolo_screen_results[0].obb
                        if scr_res is not None and scr_res.cls is not None and len(scr_res.cls) > 0:
                            clss_screen = scr_res.cls.cpu().tolist()
                            screen_xywhr = scr_res.xywhr.cpu().tolist()
                    else:
                        scr_res = yolo_screen_results[0].boxes
                        if scr_res.id is not None:
                            clss_screen = scr_res.cls.cpu().tolist()
                            for box in scr_res.xyxy.cpu().tolist():
                                x1, y1, x2, y2 = box
                                screen_xywhr.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, 0.0])

                # ===== detection processing =====
                if has_detections:
                    for xywhr, det_corners, cls, track_id, conf in zip(xywhr_list, corners_list, clss, track_ids, confs):
                        cx, cy, w, h, angle_rad = xywhr

                        # Check overlap with screen detections
                        bbox_overlapped = False
                        if self.filter_yolo:
                            for cls_screen, scr in zip(clss_screen, screen_xywhr):
                                if cls_screen != cls:
                                    continue

                                rect1 = ((cx, cy), (w, h), math.degrees(angle_rad))
                                rect2 = ((scr[0], scr[1]), (scr[2], scr[3]), math.degrees(scr[4]))
                                ret, pts = cv2.rotatedRectangleIntersection(rect1, rect2)
                                if ret != cv2.INTERSECT_NONE and pts is not None:
                                    intersection_area = cv2.contourArea(pts)
                                    screen_area = scr[2] * scr[3]
                                    if screen_area > 0 and intersection_area / screen_area >= 0.9:
                                        bbox_overlapped = True
                                        screen_xywhr.remove(scr)
                                        clss_screen.remove(cls_screen)
                                        break

                        # Scale up corners and dimensions
                        sf = self.scale_factor
                        scaled_corners = [[round(pt[0] / sf), round(pt[1] / sf)] for pt in det_corners]
                        cx_s, cy_s = cx / sf, cy / sf
                        w_s, h_s = w / sf, h / sf

                        # Transform corners to camera coords for overlay
                        corners_camera = coord.fromPixelToCameraCorners(scaled_corners)
                        center_camera = coord.fromPixelToCameraPoint([cx_s, cy_s], rounded=True)

                        color = colors(int(cls), True)
                        pts_draw = np.array(corners_camera, dtype=np.int32)

                        if bbox_overlapped:
                            cv2.drawContours(overlay_image, [pts_draw], 0, color, 2)
                            cv2.putText(overlay_image, f"{self.names[int(cls)]} filtered",
                                        (pts_draw[0][0], pts_draw[0][1] - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            continue

                        # Store detection (always in unified OBB format)
                        yolo_returns.append({
                            "class": int(cls),
                            "track_id": track_id,
                            "bbox": [round(cx_s), round(cy_s), round(w_s), round(h_s)],
                            "angle": math.degrees(angle_rad),
                            "corners": scaled_corners,
                            "score": round(conf, 5)
                        })

                        cv2.drawContours(overlay_image, [pts_draw], 0, color, 2)
                        cv2.putText(overlay_image, f"{self.names[int(cls)]} {round(conf, 3)}",
                                    (pts_draw[0][0], pts_draw[0][1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # Store tracking history
                        track = self.track_history[track_id]
                        track.append((center_camera[0], center_camera[1]))
                        if len(track) > 30:
                            track.pop(0)

                        # Plot tracks
                        points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.circle(overlay_image, (track[-1]), 7, color, -1)
                        cv2.polylines(overlay_image, [points], isClosed=False, color=color, thickness=2)
                
                # Save YOLO results
                with self._data_lock:
                    self.yolo_returns = yolo_returns
            else:
                # Save dummy results when paused
                with self._data_lock:
                    self.yolo_returns = []
            
            # Mark as ready
            if self.ready == False:
                self.ready = True
                self._ready_event.set()
                self.mode = STATE.PAUSED

            # Save an overlay image
            with self._data_lock:
                self.yolo_overlay = overlay_image