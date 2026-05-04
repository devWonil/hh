#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import os
# Fix long camera loading time
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import sys

# ── HiDPI support (must be set before ANY Qt/OpenCV imports) ─────
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_ENABLE_HIGHDPI_SCALING"]   = "1"

import time
import cv2
from PySide6.QtWidgets import *
from PySide6.QtCore import QTimer, Qt, QEvent, Signal
from PySide6.QtGui import QImage, QPixmap, QFont, QAction

from SULIVAN_Client.client import SULIVAN_Client
import config
import numpy as np
import logging

from utils.sulivan_logger import setup_logging, add_callback, get_log_dir
import utils.state as STATE

# Initialize logging once at application startup
setup_logging(config.logs_dir, debug_mode=getattr(config, 'DEBUG_MODE', False))
logger = logging.getLogger(__name__)

from utils.image_utils import overlay_black_background_image
from ui.instructor_station_ui import Ui_InstructorStation
from ui.styles import FONT_LPIPS_TITLE, FONT_LPIPS_VALUE


class InstructorStation(QMainWindow):
    _log_signal = Signal(str)

    def __init__(self, client: "SULIVAN_Client"):
        super().__init__()

        # Build UI from pure-Python definition
        self.ui = Ui_InstructorStation()
        self.ui.setupUi(self)
        self._promote_ui_attributes()

        # Connect log signal (thread-safe) and register callback
        self._log_signal.connect(self._addLogSlot)
        add_callback(self._emitLog)

        self._client = client
        self.current_button_mode = 0
        self.step_list_click_time = time.time()

        # ── Connect signals ──────────────────────────────────────
        self._connect_list_signals()
        self._connect_checkbox_signals()
        self._connect_button_signals()

        # Load scenario list
        self._load_scenario_list()

        # Run SULIVAN client
        self._client.run()

        # Polling timer (33 ms ≈ 30 fps)
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateState)
        self.timer.start(33)
        self._last_cam_update = 0

        # Window behaviour
        self.activateWindow()
        self._init_window_behavior()

    # ── Helpers called once during init ──────────────────────────

    def _promote_ui_attributes(self):
        """Copy every public attribute from Ui_InstructorStation onto *self*
        so that the rest of the code can use  self.label_camera  instead of
        self.ui.label_camera."""
        for name in dir(self.ui):
            if name.startswith('_'):
                continue
            setattr(self, name, getattr(self.ui, name))

    def _connect_list_signals(self):
        self.list_scenarios.currentItemChanged.connect(self.changeCurrentScenarioEvent)
        self.list_steps.itemDoubleClicked.connect(self.changeCurrentStepEvent)
        self.list_steps.currentItemChanged.connect(self.stepListClickedEvent)
        self.list_steps.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_steps.customContextMenuRequested.connect(self.showStepContextMenu)
        self.list_scenarios.installEventFilter(self)
        self.list_steps.installEventFilter(self)

    def _connect_checkbox_signals(self):
        self.checkbox_record_video.setCheckState(
            Qt.CheckState.Checked if config.record_video else Qt.CheckState.Unchecked)
        self.checkbox_record_video.stateChanged.connect(self.changeRecordVideoEvent)

        if config.use_project_masking:
            self.checkbox_rgb_masking.setCheckState(
                Qt.CheckState.Checked if config.use_rgb_masking else Qt.CheckState.Unchecked)
            self.checkbox_rgb_masking.stateChanged.connect(self.changeMaskProjectorEvent)
            self.checkbox_compensate_hls.setCheckState(
                Qt.CheckState.Checked if config.use_compensate_hls else Qt.CheckState.Unchecked)
            self.checkbox_compensate_hls.stateChanged.connect(self.changeCompensateHLSEvent)
        else:
            self.checkbox_rgb_masking.setVisible(False)
            self.checkbox_compensate_hls.setVisible(False)

    def _connect_button_signals(self):
        self.button_start.clicked.connect(self.startButtonFunction)
        if config.VLM_ENABLED:
            self.button_vlm_request.clicked.connect(self.vlmRequestButtonFunction)
        else:
            self.button_vlm_request.setVisible(False)
        self.button_pause.clicked.connect(self.pauseButtonFunction)
        self.button_stop.clicked.connect(self.stopButtonFunction)
        self.button_exit.clicked.connect(self.close)
        self.button_screenshot.clicked.connect(self.takeScreenshot)
        if config.use_project_masking:
            self.button_calibrate.clicked.connect(self.calibrateProjectorFunction)
        else:
            self.button_calibrate.setVisible(False)

    def _load_scenario_list(self):
        self.selected_scenario = None
        self.selected_scenario_name = None
        scenarios_path = 'scenarios/'
        for scenario in os.listdir(scenarios_path):
            if scenario.startswith('.'):
                continue
            if os.path.isdir(scenarios_path + scenario):
                self.list_scenarios.addItem(os.path.splitext(scenario)[0])
        self.list_scenarios.setCurrentRow(0)

    # ── Runtime display helpers ───────────────────────────────────

    def _format_runtime(self, seconds):
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _set_runtime_display(self, scenario_seconds, step_seconds):
        self.runtime_total_value.setText(self._format_runtime(scenario_seconds))
        self.runtime_step_value.setText(self._format_runtime(step_seconds))

    def _reset_runtime_display(self):
        placeholder = "--:--"
        self.runtime_total_value.setText(placeholder)
        self.runtime_step_value.setText(placeholder)

    def _refresh_runtime_display(self, is_scenario_loaded, state):
        if is_scenario_loaded and state == STATE.RUNNING:
            try:
                scenario_runtime, step_runtime = self._client.scenario.getRunningTimes()
            except AttributeError:
                self._reset_runtime_display()
                return
            self._set_runtime_display(scenario_runtime, step_runtime)
        else:
            self._reset_runtime_display()

    def _init_window_behavior(self):
        self._custom_maximized = False
        self._normal_geometry = self.geometry()
        # For a key press event
        QApplication.instance().installEventFilter(self)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange:
            if self.windowState() & Qt.WindowMaximized and not self._custom_maximized:
                self._enter_custom_maximized_mode()
            elif self._custom_maximized and not self.isFullScreen():
                self._exit_custom_maximized_mode()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            key = event.key()
            if key in [Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down]:
                if self.isActiveWindow() and self._client.isScenarioLoaded():
                    self.keyPressEvent(event)
                    return True

        if event.type() == QEvent.MouseButtonDblClick:
            if self._is_blank_area(obj, event):
                self.toggleMaximizedState()
                event.accept()
                return True
        return super().eventFilter(obj, event)

    def _is_blank_area(self, obj, event):
        if obj is self.centralwidget:
            return obj.childAt(event.pos()) is None
        if obj is getattr(self, "status_bar", None):
            return True
        return False

    def toggleMaximizedState(self):
        if self._custom_maximized:
            self._exit_custom_maximized_mode()
        else:
            self._enter_custom_maximized_mode()

    def _enter_custom_maximized_mode(self):
        if self._custom_maximized:
            return
        if not self.isMaximized() and not self.isFullScreen():
            self._normal_geometry = self.geometry()
        self._custom_maximized = True
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.showFullScreen()
        self.raise_()
        self.activateWindow()

    def _exit_custom_maximized_mode(self):
        if not self._custom_maximized:
            return
        self._custom_maximized = False
        self.setWindowFlag(Qt.WindowStaysOnTopHint, False)
        self.showNormal()
        if self._normal_geometry is not None:
            self.setGeometry(self._normal_geometry)
        self.raise_()
        self.activateWindow()
        self.centralwidget.updateGeometry()
        self.label_camera.updateGeometry()

    def resizeEvent(self, event):
        if (not self._custom_maximized and not self.isMaximized() 
                and not self.isFullScreen()):
            self._normal_geometry = self.geometry()
        super().resizeEvent(event)

    # Handle event which changing scenario from scenarios list
    def changeCurrentScenarioEvent(self):
        is_scenario_loaded = self._client.isScenarioLoaded()
        if not is_scenario_loaded:
            self.selected_scenario      = self.list_scenarios.currentRow()
            self.selected_scenario_name = self.list_scenarios.currentItem().text()

    # Handle event which changing step from steps list
    def changeCurrentStepEvent(self, item):
        is_scenario_loaded = self._client.isScenarioLoaded()
        if is_scenario_loaded:
            move_row = self.list_steps.indexFromItem(item).row()
            self._client.scenario.changeStep(move_row)

    def stepListClickedEvent(self):
        self.step_list_click_time = time.time()

    def showStepContextMenu(self, pos):
        item = self.list_steps.itemAt(pos)
        if item is None:
            return
            
        step_index = self.list_steps.row(item)
        
        # Check if scenario is loaded
        if not self._client.isScenarioLoaded():
            return

        # Can only capture for current step because objects are only instantiated for current step
        if step_index != self._client.scenario.current_step:
            return 
            
        lpips_objects = self._client.scenario.lpips_objects
        if not lpips_objects:
            return
            
        menu = QMenu(self)
        
        for name, obj in lpips_objects.items():
            action = QAction(f"Capture Reference for '{name}'", self)
            # Use default argument to capture loop variable
            action.triggered.connect(lambda checked, o=obj: self.captureLPIPSReference(o))
            menu.addAction(action)
            
        menu.exec_(self.list_steps.mapToGlobal(pos))

    def captureLPIPSReference(self, lpips_object):
        success = lpips_object.capture_reference()
        if success:
            QMessageBox.information(self, "Success", f"Captured reference for {lpips_object.name}")
        else:
            QMessageBox.warning(self, "Error", f"Failed to capture reference for {lpips_object.name}")

    def _create_lpips_item_widget(self, name):
        """Create a widget for displaying one LPIPS comparison."""
        item_widget = QWidget()
        item_layout = QHBoxLayout(item_widget)
        item_layout.setContentsMargins(5, 5, 5, 5)
        item_layout.setSpacing(8)

        def _img_label():
            lbl = QLabel()
            lbl.setFixedSize(100, 80)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("border: 1px solid #888; background-color: #333;")
            return lbl

        # Reference image section
        ref_section = QVBoxLayout()
        ref_title = QLabel("Reference")
        ref_title.setFont(FONT_LPIPS_TITLE)
        ref_title.setAlignment(Qt.AlignCenter)
        ref_image_label = _img_label()
        ref_section.addWidget(ref_title)
        ref_section.addWidget(ref_image_label)
        item_layout.addLayout(ref_section)

        # Comparison image section
        cmp_section = QVBoxLayout()
        cmp_title = QLabel("Current")
        cmp_title.setFont(FONT_LPIPS_TITLE)
        cmp_title.setAlignment(Qt.AlignCenter)
        cmp_image_label = _img_label()
        cmp_section.addWidget(cmp_title)
        cmp_section.addWidget(cmp_image_label)
        item_layout.addLayout(cmp_section)

        # Score section
        score_section = QVBoxLayout()
        name_label = QLabel(name)
        name_label.setFont(FONT_LPIPS_TITLE)
        name_label.setAlignment(Qt.AlignCenter)
        score_label = QLabel("--")
        score_label.setFont(FONT_LPIPS_VALUE)
        score_label.setAlignment(Qt.AlignCenter)
        score_label.setMinimumWidth(80)
        threshold_label = QLabel("")
        threshold_label.setFont(FONT_LPIPS_TITLE)
        threshold_label.setAlignment(Qt.AlignCenter)
        score_section.addWidget(name_label)
        score_section.addWidget(score_label)
        score_section.addWidget(threshold_label)
        item_layout.addLayout(score_section)

        return {
            'widget': item_widget,
            'ref_image': ref_image_label,
            'cmp_image': cmp_image_label,
            'score_label': score_label,
            'threshold_label': threshold_label,
            'name_label': name_label
        }

    def _update_lpips_display(self):
        """Update LPIPS comparison display based on current scenario state."""
        if not self._client.isScenarioLoaded():
            self.lpips_panel.setVisible(False)
            return
        
        lpips_objects = self._client.scenario.lpips_objects
        if not lpips_objects:
            self.lpips_panel.setVisible(False)
            return
        
        # Show panel
        self.lpips_panel.setVisible(True)
        
        # Remove widgets for LPIPS objects that no longer exist
        for name in list(self.lpips_display_widgets.keys()):
            if name not in lpips_objects:
                widget_info = self.lpips_display_widgets.pop(name)
                widget_info['widget'].setParent(None)
                widget_info['widget'].deleteLater()
        
        # Update or create widgets for each LPIPS object
        for name, lpips_obj in lpips_objects.items():
            # Create widget if not exists
            if name not in self.lpips_display_widgets:
                widget_info = self._create_lpips_item_widget(name)
                # Insert before the stretch
                self.lpips_container_layout.insertWidget(
                    self.lpips_container_layout.count() - 1,
                    widget_info['widget']
                )
                self.lpips_display_widgets[name] = widget_info
            
            widget_info = self.lpips_display_widgets[name]
            
            # Update reference image
            ref_img = getattr(lpips_obj, '_reference_image_bgr', None)
            if ref_img is not None:
                self._set_image_to_label(widget_info['ref_image'], ref_img)
            
            # Update comparison image
            cmp_img = getattr(lpips_obj, '_last_comparison_image_bgr', None)
            if cmp_img is not None:
                self._set_image_to_label(widget_info['cmp_image'], cmp_img)
            
            # Update score
            score = getattr(lpips_obj, '_last_score', None)
            threshold = lpips_obj.threshold
            widget_info['threshold_label'].setText(f"Threshold: {threshold:.3f}")
            
            if score is not None:
                widget_info['score_label'].setText(f"{score:.4f}")
                # Color code based on threshold
                if score < threshold:
                    widget_info['score_label'].setStyleSheet("color: #00ff00; font-weight: bold;")  # Green - pass
                else:
                    widget_info['score_label'].setStyleSheet("color: #ff6666; font-weight: bold;")  # Red - not pass
            else:
                widget_info['score_label'].setText("--")
                widget_info['score_label'].setStyleSheet("")

    def _set_image_to_label(self, label, cv_image):
        """Convert OpenCV image to QPixmap and set to QLabel."""
        if cv_image is None:
            return
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale to fit label
        scaled_pixmap = pixmap.scaled(
            label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)

    def displayStepList(self):
        self.list_steps.clear()
        steps = self._client.scenario.getSteps(is_numbered=True)
        for step in steps:
            self.list_steps.addItem(step)

    def changeRecordVideoEvent(self):
        if self.checkbox_record_video.isChecked():
            config.record_video = True
            logger.info("Recording video: ON")
        else:
            config.record_video = False
            logger.info("Recording video: OFF")
            
    def changeMaskProjectorEvent(self):
        if self.checkbox_rgb_masking.isChecked():
            config.use_rgb_masking = True
            logger.info("RGB Masking: ON")
        else:
            config.use_rgb_masking = False
            logger.info("RGB Masking: OFF")
                        
    def changeCompensateHLSEvent(self):
        if self.checkbox_compensate_hls.isChecked():
            config.use_compensate_hls = True
            logger.info("Compensate HLS: ON")
        else:
            config.use_compensate_hls = False
            logger.info("Compensate HLS: OFF")
    
    def action_restart_step(self):
        if self._client.isScenarioLoaded():
            self._client.scenario.recordAction("Step is restarted by operator")
            self._client.scenario.changeStep(self._client.scenario.current_step)

    def action_complete(self):
        if self._client.isScenarioLoaded():
            self._client.scenario.achieveCompletionEvent("popup:ok")
            self._client.scenario.recordAction("Process is completed by operator")
    
    # Update camera image and scenario loaded state
    def updateState(self):
        # Handle client state
        is_client_exit = self._client.is_exit
        if is_client_exit:
            self.close()

        self.button_exit.setEnabled(True)

        # Update scenario loaded state
        is_scenario_loaded = self._client.isScenarioLoaded()
        state = STATE.STOPPED
        if is_scenario_loaded:
            state = self._client.scenario.getRunningState()
            self.button_exit.setEnabled(False)
            self.updateButtonMode(state)
            if state == STATE.RUNNING:
                self.label_state.setText("RUNNING - " + self.selected_scenario_name)
            elif state == STATE.PAUSED:
                self.label_state.setText("PAUSE - " + self.selected_scenario_name)
            
            # Fix scenario selector
            self.list_scenarios.setCurrentRow(self.selected_scenario)
            
            # Update current step
            current_step = self._client.scenario.getCurrentStep()
            self.list_steps.setCurrentRow(current_step)
        else:
            self.label_state.setText("STOPPED")
            self.list_steps.clear()
            self.updateButtonMode(0)

        self._refresh_runtime_display(is_scenario_loaded, state)

        # Update LPIPS comparison display
        self._update_lpips_display()

        # Update Camera image (throttled to ~30fps to reduce unnecessary copies)
        _now = time.monotonic()
        if _now - self._last_cam_update >= 0.033:
            self._last_cam_update = _now
            self._updateCameraDisplay()

    def _updateCameraDisplay(self):
        """Update the camera display widget with latest frame (~30fps)."""
        frame = self._client.mp_thread.getImage()
        if frame is None:
            return

        # Draw a trapezoid of projector area
        if config.use_camera_projection:
            cv2.line(frame, config.DISPLAY_POINTS[0], config.DISPLAY_POINTS[1], (255, 0, 0), 3)
            cv2.line(frame, config.DISPLAY_POINTS[0], config.DISPLAY_POINTS[2], (255, 0, 0), 3)
            cv2.line(frame, config.DISPLAY_POINTS[3], config.DISPLAY_POINTS[1], (255, 0, 0), 3)
            cv2.line(frame, config.DISPLAY_POINTS[3], config.DISPLAY_POINTS[2], (255, 0, 0), 3)

        # Overlay YOLO detection if available
        yolo_overlay = self._client.yolo_thread.getOverlayImage()
        if yolo_overlay is not None:
            frame = overlay_black_background_image(frame, yolo_overlay)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.label_camera.setPixmap(pixmap.scaled(
            self.label_camera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # Logger callback function
    def _emitLog(self, text):
        """Thread-safe: emit signal from any thread."""
        self._log_signal.emit(text)

    def _addLogSlot(self, text):
        """Slot: always runs on the GUI thread."""
        self.list_logs.addItem(text)
        self.list_logs.scrollToBottom()

    # Backward-compatible alias
    def addLog(self, text):
        self._emitLog(text)

    # Update button mode
    def updateButtonMode(self, state):
        if self.current_button_mode != state:
            self.current_button_mode = state
            if state == STATE.STOPPED:
                self.button_start.setEnabled(True)
                self.button_pause.setEnabled(False)
                self.button_stop.setEnabled(False)
                self.button_pause.setText("PAUSE")
                self.checkbox_record_video.setEnabled(True)

            elif state == STATE.RUNNING:
                self.button_start.setEnabled(False)
                self.button_pause.setEnabled(True)
                self.button_stop.setEnabled(True)
                self.button_pause.setText("PAUSE")
                self.checkbox_record_video.setEnabled(False)

            elif state == STATE.PAUSED:
                self.button_start.setEnabled(False)
                self.button_pause.setEnabled(True)
                self.button_stop.setEnabled(True)
                self.button_pause.setText("RESUME")
                self.checkbox_record_video.setEnabled(False)

    # Start button
    def startButtonFunction(self):
        self._client.loadScenario(self.selected_scenario_name)
        self.displayStepList()

    # Pause button
    def pauseButtonFunction(self):
        if self._client.isScenarioLoaded():
            sh = self._client.scenario
            if sh.getRunningState() == STATE.PAUSED:
                logger.info("RESUME scenario")
                sh.resume()

            elif sh.getRunningState() == STATE.RUNNING:
                logger.info("PAUSE scenario")
                sh.pause()

    # Stop button
    def stopButtonFunction(self):
        self._client.unloadScenario()

    # VLM Request button
    def vlmRequestButtonFunction(self):
        if self._client.isScenarioLoaded():
            self._client.scenario.triggerVLMFeedback()
    
    # Stop SULIVAN client when close button in clicked
    def closeEvent(self, QCloseEvent):
        if not self._client.is_exit:
            self._client.stop()
        QCloseEvent.accept()

    # Take screenshot
    def takeScreenshot(self):
        try:
            logger.info("Take screenshot")
            screenshot_date = time.strftime("%Y-%m-%d_%H-%M-%S")
            screenshot_dir = get_log_dir() + "/screenshots/"
            if not os.path.exists(screenshot_dir):
                os.makedirs(screenshot_dir)
            
            camera_raw = self._client.camera_thread.getRawImage()
            camera_masked = self._client.camera_thread.getImage()
            mp_overlay = self._client.mp_thread.getImage()
            yolo_overlay = self._client.yolo_thread.getOverlayImage()
            camera_with_detect_overlay = overlay_black_background_image(mp_overlay, yolo_overlay)
            projector_image = self._client.getDisplayImage()

            cv2.imwrite(screenshot_dir + f"{screenshot_date}_camera_raw.png", camera_raw)
            cv2.imwrite(screenshot_dir + f"{screenshot_date}_camera_masked.png", camera_masked)
            cv2.imwrite(screenshot_dir + f"{screenshot_date}_detect_overlay.png", camera_with_detect_overlay)
            cv2.imwrite(screenshot_dir + f"{screenshot_date}_src.png", projector_image)
            
        except Exception as e:
            logger.info(f"Take screenshot failed: {e}")

    # Calibrate Projector button
    def calibrateProjectorFunction(self):
        logger.info("Start Projector Calibration")
        self._client.camera_thread.start_calibration()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Up:
            self.action_restart_step()
        elif key == Qt.Key_Down:
            self.action_complete()
        else:
            super().keyPressEvent(event)

# ============ Main Entry point ============
if __name__ == "__main__":
    # HiDPI scaling is enabled by default in Qt6/PySide6

    client = SULIVAN_Client()
    app = QApplication(sys.argv)
    app.setFont(QFont("Noto Sans KR"))
    instructor_window = InstructorStation(client=client)
    instructor_window.show()
    app.exec()