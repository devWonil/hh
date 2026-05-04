# ui/instructor_station_ui.py
# ──────────────────────────────────────────────────────────────────
# Pure-Python replacement for instructor_station.ui
# All widget creation, layout, font, and size-policy live here.
# The main InstructorStation class only needs to call setupUi(self)
# and then connect signals / add business logic.
# ──────────────────────────────────────────────────────────────────

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QListWidget,
    QGroupBox, QCheckBox, QStatusBar,
    QScrollArea, QFrame, QSizePolicy, QMenu,
    QHBoxLayout, QVBoxLayout, QGridLayout,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QAction

from ui.styles import (
    FONT_TITLE, FONT_GROUPBOX, FONT_LIST, FONT_LOG,
    FONT_BUTTON_BIG, FONT_BUTTON_MED, FONT_BUTTON_CAL,
    FONT_CHECKBOX,
    FONT_RUNTIME_CAP, FONT_RUNTIME_VAL,
)


class Ui_InstructorStation:
    """Build every widget and lay them out responsively."""

    # ── public entry point ───────────────────────────────────────
    def setupUi(self, window: QMainWindow):
        window.setWindowTitle("SULIVAN Instructor Station")
        window.setMinimumSize(1440, 1000)

        # Central widget
        self.centralwidget = QWidget()
        window.setCentralWidget(self.centralwidget)

        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.setSizeGripEnabled(False)
        window.setStatusBar(self.status_bar)

        # ── Create all widgets ───────────────────────────────────
        self._create_top_bar_widgets()
        self._create_left_column_widgets()
        self._create_center_column_widgets()
        self._create_right_column_widgets()
        self._create_status_bar_widgets()

        # ── Assemble layouts ─────────────────────────────────────
        root_layout = QVBoxLayout(self.centralwidget)
        root_layout.addLayout(self._build_top_bar())
        root_layout.addLayout(self._build_content_area())

    # ═════════════════════════════════════════════════════════════
    #  Widget creation  (grouped by screen region)
    # ═════════════════════════════════════════════════════════════

    def _create_top_bar_widgets(self):
        # State labels
        self.explain_state = QLabel("SULIVAN Client state: ")
        self.explain_state.setFont(FONT_TITLE)

        self.label_state = QLabel("")
        self.label_state.setFont(FONT_TITLE)
        self.label_state.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Buttons
        self.button_screenshot = QPushButton("Screenshot")
        self.button_screenshot.setFont(FONT_BUTTON_MED)
        self.button_screenshot.setFixedHeight(60)
        self.button_screenshot.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.button_exit = QPushButton("EXIT")
        self.button_exit.setFont(FONT_BUTTON_BIG)
        self.button_exit.setFixedHeight(60)
        self.button_exit.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)

    # ── Left column ──────────────────────────────────────────────

    def _create_left_column_widgets(self):
        # Scenario group
        self.scenario_groupbox = QGroupBox("Scenario list")
        self.scenario_groupbox.setFont(FONT_GROUPBOX)
        self.scenario_groupbox.setMinimumWidth(500)

        self.list_scenarios = QListWidget()
        self.list_scenarios.setFont(FONT_LIST)
        QVBoxLayout(self.scenario_groupbox).addWidget(self.list_scenarios)

        # Step group
        self.step_groupbox = QGroupBox("Step list")
        self.step_groupbox.setFont(FONT_GROUPBOX)
        self.step_groupbox.setMinimumWidth(500)

        self.list_steps = QListWidget()
        self.list_steps.setFont(FONT_LIST)
        QVBoxLayout(self.step_groupbox).addWidget(self.list_steps)

    # ── Center column ────────────────────────────────────────────

    def _create_center_column_widgets(self):
        # Camera group (with LPIPS panel)
        self.camera_groupbox = QGroupBox("Camera")
        self.camera_groupbox.setFont(FONT_GROUPBOX)

        self.label_camera = QLabel()
        self.label_camera.setAlignment(Qt.AlignCenter)
        self.label_camera.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.label_camera.setMinimumSize(1, 1)

        self._init_lpips_panel()

        camera_layout = QVBoxLayout(self.camera_groupbox)
        camera_layout.addWidget(self.label_camera, stretch=1)
        camera_layout.addWidget(self.lpips_panel)

        # Log group
        self.log_groupbox = QGroupBox("Logs")
        self.log_groupbox.setFont(FONT_GROUPBOX)

        self.list_logs = QListWidget()
        self.list_logs.setFont(FONT_LOG)
        QVBoxLayout(self.log_groupbox).addWidget(self.list_logs)

    # ── Right column ─────────────────────────────────────────────

    def _create_right_column_widgets(self):
        # Control buttons
        self.button_start = QPushButton("START")
        self.button_start.setFont(FONT_BUTTON_BIG)
        self.button_start.setFixedHeight(80)
        self.button_start.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.button_pause = QPushButton("PAUSE")
        self.button_pause.setFont(FONT_BUTTON_BIG)
        self.button_pause.setFixedHeight(80)
        self.button_pause.setEnabled(False)
        self.button_pause.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.button_stop = QPushButton("STOP")
        self.button_stop.setFont(FONT_BUTTON_BIG)
        self.button_stop.setFixedHeight(80)
        self.button_stop.setEnabled(False)
        self.button_stop.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)


        # Runtime panel
        self._build_runtime_panel()

        # Check-boxes
        self.checkbox_record_video = QCheckBox("Record video")
        self.checkbox_record_video.setFont(FONT_CHECKBOX)

        self.checkbox_rgb_masking = QCheckBox("Color Refinement")
        self.checkbox_rgb_masking.setFont(FONT_CHECKBOX)

        self.checkbox_compensate_hls = QCheckBox("Lightness Compensation")
        self.checkbox_compensate_hls.setFont(FONT_CHECKBOX)

        # Calibrate
        self.button_calibrate = QPushButton("Calibrate Projector")
        self.button_calibrate.setFont(FONT_BUTTON_CAL)

        # VLM Request
        self.button_vlm_request = QPushButton("VLM Request")
        self.button_vlm_request.setFont(FONT_BUTTON_CAL)

    # ═════════════════════════════════════════════════════════════
    #  Composite sub-builders
    # ═════════════════════════════════════════════════════════════

    def _build_runtime_panel(self):
        self.runtime_panel = QGroupBox("Elapsed Time")
        self.runtime_panel.setFont(FONT_GROUPBOX)

        grid = QGridLayout(self.runtime_panel)
        grid.setContentsMargins(12, 12, 12, 12)
        grid.setHorizontalSpacing(16)

        total_cap = QLabel("Total")
        total_cap.setFont(FONT_RUNTIME_CAP)
        total_cap.setAlignment(Qt.AlignCenter)
        step_cap = QLabel("Current")
        step_cap.setFont(FONT_RUNTIME_CAP)
        step_cap.setAlignment(Qt.AlignCenter)

        self.runtime_total_value = QLabel("--:--")
        self.runtime_total_value.setFont(FONT_RUNTIME_VAL)
        self.runtime_total_value.setAlignment(Qt.AlignCenter)
        self.runtime_total_value.setMinimumHeight(60)

        self.runtime_step_value = QLabel("--:--")
        self.runtime_step_value.setFont(FONT_RUNTIME_VAL)
        self.runtime_step_value.setAlignment(Qt.AlignCenter)
        self.runtime_step_value.setMinimumHeight(60)

        grid.addWidget(total_cap, 0, 0)
        grid.addWidget(step_cap, 0, 1)
        grid.addWidget(self.runtime_total_value, 1, 0)
        grid.addWidget(self.runtime_step_value, 1, 1)

    def _init_lpips_panel(self):
        """LPIPS comparison display panel (hidden by default)."""
        self.lpips_panel = QWidget()
        self.lpips_panel.setFixedHeight(140)
        self.lpips_panel.setVisible(False)

        lpips_layout = QHBoxLayout(self.lpips_panel)
        lpips_layout.setContentsMargins(5, 5, 5, 5)
        lpips_layout.setSpacing(10)

        self.lpips_scroll_area = QScrollArea()
        self.lpips_scroll_area.setWidgetResizable(True)
        self.lpips_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.lpips_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.lpips_scroll_area.setFrameShape(QFrame.NoFrame)

        self.lpips_container = QWidget()
        self.lpips_container_layout = QHBoxLayout(self.lpips_container)
        self.lpips_container_layout.setContentsMargins(0, 0, 0, 0)
        self.lpips_container_layout.setSpacing(20)
        self.lpips_container_layout.addStretch()

        self.lpips_scroll_area.setWidget(self.lpips_container)
        lpips_layout.addWidget(self.lpips_scroll_area)

        self.lpips_display_widgets = {}

    def _create_status_bar_widgets(self):
        self.status_bar.showMessage("Ready")

    # ═════════════════════════════════════════════════════════════
    #  Layout assembly
    # ═════════════════════════════════════════════════════════════

    def _build_top_bar(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.addWidget(self.explain_state)
        layout.addWidget(self.label_state)
        layout.addStretch(1)
        layout.addWidget(self.button_screenshot)
        layout.addWidget(self.button_exit)
        return layout

    def _build_content_area(self) -> QHBoxLayout:
        content = QHBoxLayout()
        content.addLayout(self._build_left_column(), 3)
        content.addLayout(self._build_center_column(), 6)
        content.addWidget(self._build_right_column())
        return content

    def _build_left_column(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.addWidget(self.scenario_groupbox, 1)
        layout.addWidget(self.step_groupbox, 1)
        return layout

    def _build_center_column(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.addWidget(self.camera_groupbox, 2)
        layout.addWidget(self.log_groupbox, 1)
        return layout

    def _build_right_column(self) -> QWidget:
        container = QWidget()
        container.setFixedWidth(300)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Control buttons
        layout.addWidget(self.button_start)
        layout.addWidget(self.button_pause)
        layout.addWidget(self.button_stop)
        layout.addWidget(self.runtime_panel)

        # Check-boxes
        layout.addWidget(self.checkbox_record_video)
        layout.addWidget(self.checkbox_rgb_masking)
        layout.addWidget(self.checkbox_compensate_hls)

        layout.addWidget(self.button_calibrate)
        layout.addWidget(self.button_vlm_request)
        layout.addStretch(1)

        return container
