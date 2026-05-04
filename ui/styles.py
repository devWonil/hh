# ui/styles.py  –  Font & style constants for Instructor Station UI

from PySide6.QtGui import QFont


# ── Fonts ──────────────────────────────────────────────────────────
def _bold(font: QFont) -> QFont:
    font.setBold(True)
    return font


FONT_TITLE       = _bold(QFont("Noto Sans KR Black", 24))   # window title labels
FONT_GROUPBOX    = QFont("Noto Sans KR Medium", 18)          # group-box headers
FONT_LIST        = QFont("Noto Sans KR", 24)       # list widgets
FONT_LOG         = QFont("Noto Sans KR", 10)       # log list
FONT_BUTTON_BIG  = _bold(QFont("Noto Sans KR Black", 24))    # START / PAUSE / STOP / EXIT
FONT_BUTTON_MED  = _bold(QFont("Noto Sans KR Black", 18))    # Screenshot
FONT_BUTTON_CAL  = _bold(QFont("Noto Sans KR Black", 16))    # Calibrate
FONT_CHECKBOX    = QFont("Noto Sans KR", 16)                  # check-boxes
FONT_FORM        = QFont("Noto Sans KR Medium", 14)           # login form fields
FONT_LABEL       = QFont("Noto Sans KR", 14)                  # generic labels
FONT_LABEL_MSG   = QFont("Noto Sans KR Medium", 12)           # login message
FONT_RUNTIME_CAP = QFont("Noto Sans KR", 14)                  # runtime captions
FONT_RUNTIME_VAL = _bold(QFont("Noto Sans KR", 28))           # runtime values
FONT_STATUS_LBL  = QFont("Noto Sans KR", 14)                  # status-bar label
FONT_STATUS_BAR  = QFont("Noto Sans KR", 12)                  # status-bar progress
FONT_LPIPS_TITLE = _bold(QFont("Noto Sans KR", 9))            # LPIPS section titles
FONT_LPIPS_VALUE = QFont("Noto Sans KR", 11)                  # LPIPS score values
