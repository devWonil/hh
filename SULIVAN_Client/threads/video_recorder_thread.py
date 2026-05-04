#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import os
import datetime
import time
import json
from hashlib import md5

import numpy as np
import ffmpegcv
import threading

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from client import SULIVAN_Client

import config
import logging

logger = logging.getLogger(__name__)
from SULIVAN_Client.objects.hand_button import HandButton

FPS = 30

class VideoRecorderThread(threading.Thread):
    def __init__(self, client: "SULIVAN_Client", scenario_name, start_time=None, user_name=None):
        self._client = client
        self.scenario_name = scenario_name
        self.user_name = user_name
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time

        self.date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        
        if self.user_name is None:
            filename = f"{self.date}_{self.scenario_name}.mp4"
        else:
            filename = f"{self.date}_{self.scenario_name}_{self.user_name}.mp4"
        
        # Make hashed filename for saved recording
        self.filename = f"{self.date}_{md5(filename.encode('utf-8')).hexdigest()}.mp4"

        self.camera_thread = self._client.camera_thread
        self.mp_thread = self._client.mp_thread
        self.yolo_thread = self._client.yolo_thread
        
        self._stop_event = threading.Event()
        super().__init__()
        self.name = "VideoRecorder"

        # Create record folder if not exists
        if not config.record_dir in os.listdir('./'):
            logger.warning("records folder not exists. creating...")
            os.mkdir(config.record_dir)

    def stop(self):
        self._stop_event.set()
        return self.filename

    def run(self):
        '''Record video and mediapipe detects'''
        # Wait for camera thread is ready
        self.camera_thread._ready_event.wait()
                
        output_path = f"{config.record_dir}/{self.filename}"
        output_video = ffmpegcv.VideoWriterNV(output_path, 'h264', FPS)
        
        # Record video before exit flag is triggered
        logger.info(f"Recording started: {self.filename}")

        # Define Camera to Pixel coord matrix
        if config.use_camera_projection:
            M_cp = config.M_cp
        else:
            M_cp = np.eye(3)

        json_metadata = {'date': self.date, 'M_cp': M_cp.tolist()}
        all_detects = []

        frame_interval = 1.0 / FPS
        frame_count = 0  # Number of frames already written

        # Use monotonic clock for strict frame scheduling while preserving scenario start_time reference
        clock_offset = time.perf_counter() - time.time()
        start_time_mono = self.start_time + clock_offset

        with output_video:
            # Compensate for elapsed time since scenario start by filling with first frame
            first_frame = None
            while first_frame is None:
                try:
                    first_frame = self.mp_thread.getRawImage()
                except Exception:
                    time.sleep(0.01)
            
            current_time_mono = time.perf_counter()
            elapsed_time = max(0.0, current_time_mono - start_time_mono)
            frames_to_fill = int(elapsed_time / frame_interval)
            
            if frames_to_fill > 0:
                logger.debug(f"Compensating {frames_to_fill} frames ({elapsed_time:.3f}s) for initialization delay")
                # Get initial detect data for compensation frames
                btn_name, btn_mp, btn_state = HandButton.getActiveButtonSnapshot()
                first_detect_data = {
                    'step': self._client.scenario.getCurrentStep(),
                    'mediapipe': self.mp_thread.getHands(),
                    'yolo': self.yolo_thread.getDetection(),
                    'clicking_button_info': btn_mp,
                    'clicking_button_name': btn_name,
                    'clicking_button_state': btn_state
                }
                for _ in range(frames_to_fill):
                    output_video.write(first_frame)
                    detect_data = {'time': round(frame_count * frame_interval, 5)}
                    detect_data.update(first_detect_data)
                    all_detects.append(detect_data)
                    frame_count += 1

            while True:
                if self._stop_event.is_set():
                    break

                current_time_mono = time.perf_counter()
                elapsed_time = max(0.0, current_time_mono - start_time_mono)
                
                # Calculate expected written frame count based on elapsed time
                expected_frame = int(elapsed_time / frame_interval)

                # Record frames if there are dropped frames
                if expected_frame > frame_count:
                    try:
                        frame = self.mp_thread.getRawImage()

                        btn_name, btn_mp, btn_state = HandButton.getActiveButtonSnapshot()
                        detect_data = {
                            'step': self._client.scenario.getCurrentStep(),
                            'mediapipe': self.mp_thread.getHands(),
                            'yolo': self.yolo_thread.getDetection(),
                            'clicking_button_info': btn_mp,
                            'clicking_button_name': btn_name,
                            'clicking_button_state': btn_state,
                        }

                        # Write the same frame for the number of dropped frames
                        frames_to_write = expected_frame - frame_count
                        for _ in range(frames_to_write):
                            output_video.write(frame)
                            detect_entry = {'time': round(frame_count * frame_interval, 5)}
                            detect_entry.update(detect_data)
                            all_detects.append(detect_entry)
                            frame_count += 1
                    except Exception as e:
                        logger.debug(f"Frame recording error: {e}")
                
                # Sleep until the next frame time
                next_frame_time = start_time_mono + (frame_count + 1) * frame_interval
                sleep_time = next_frame_time - time.perf_counter()
                
                if sleep_time > 0.002:
                    time.sleep(sleep_time - 0.001)  # Sleep most of the time
                elif sleep_time > 0:
                    time.sleep(0.0005)  # 0.5 ms
        
        output_video.release()

        # Write JSON using json.dump (proper serialization, no string replace hacks)
        output_json = {'metadata': json_metadata, 'detects': all_detects}
        with open(f"{config.record_dir}/{self.filename[0:-4]}.json", 'w') as f:
            json.dump(output_json, f)

        # return name of recorded file
        return self.filename