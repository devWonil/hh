#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from SULIVAN_Client.threads.camera_thread import CameraThread
import config
config.use_project_masking = False # Disable mask projector for video recording

import datetime
import time

import cv2
import ffmpegcv

FPS = 30

class VideoRecorder:
    def __init__(self, camera_device):
        self.is_exit = False
        if config.undistort_camera:
            self.camera_thread = CameraThread(self, config.camera_device, is_rotate=config.use_rotate_camera, use_flip_camera=config.use_flip_camera,
                                               use_camera_projection=config.use_camera_projection,
                                               mtx=config.filename_calibration_matrix, dist=config.filename_distortion_coefficients)
        else:
            self.camera_thread = CameraThread(self, config.camera_device, is_rotate=config.use_rotate_camera, use_flip_camera=config.use_flip_camera,
                                               use_camera_projection=config.use_camera_projection)

    def stop(self):
        self.camera_thread.stop()
        self.is_exit = True
        exit()

    def run(self):
        print("Starting camera thread")
        self.camera_thread.start()

        # Wait for camera thread is ready
        while self.camera_thread.ready != True:
            time.sleep(1)
        
        # Get video size
        shape = self.camera_thread.getImage().shape
        
        filename = f"{config.record_dir}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.mp4"
        print("Recording filename:", filename)
        
        output_video = ffmpegcv.VideoWriterNV(filename, 'h264', FPS)
        
        # Record video before exit flag is triggered
        print("Recording started")
        frame_interval = 1.0 / FPS
        start_time = time.perf_counter()
        frame_count = 0
        while not self.is_exit:
            now = time.perf_counter()
            elapsed_time = now - start_time
            expected_frame = int(elapsed_time / frame_interval)

            if expected_frame > frame_count:
                frame = self.camera_thread.getImage()
                frames_to_write = expected_frame - frame_count
                for _ in range(frames_to_write):
                    output_video.write(frame)
                    frame_count += 1
            
            next_frame_time = start_time + (frame_count + 1) * frame_interval
            sleep_time = next_frame_time - time.perf_counter()
            if sleep_time > 0.002:
                time.sleep(sleep_time - 0.001)
            elif sleep_time > 0:
                time.sleep(0.0005)
        
        output_video.release()
        print("Recording ended")

if __name__ == '__main__':
    video_recorder = VideoRecorder(0)
    video_recorder.run()