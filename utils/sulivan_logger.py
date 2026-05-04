#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import os
import datetime as dt
import logging

LOG_FORMAT_FILE = '%(asctime)s [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] %(message)s'
LOG_FORMAT_CONSOLE = '%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Module-level state (set by setup_logging)
_log_dir = None
_initialized = False


class CallbackHandler(logging.Handler):
    """logging.Handler subclass that forwards formatted log records to a callback."""

    def __init__(self, callback, level=logging.NOTSET):
        super().__init__(level)
        self.callback = callback

    def emit(self, record):
        try:
            msg = self.format(record)
            self.callback(msg)
        except Exception:
            self.handleError(record)


def setup_logging(logs_dir: str, debug_mode: bool = False) -> str:
    """Configure the root logger with file and stream handlers.

    Should be called once from the application entry point (main.py)
    before any logging calls are made.

    Returns:
        The path to the session-specific log directory.
    """
    global _log_dir, _initialized

    if _initialized:
        return _log_dir

    log_level = logging.DEBUG if debug_mode else logging.INFO

    # Create session log directory
    os.makedirs(logs_dir, exist_ok=True)
    _log_dir = os.path.join(logs_dir, dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(_log_dir, exist_ok=True)

    log_filename = os.path.join(_log_dir, 'sulivan.log')

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(log_level)

    file_formatter = logging.Formatter(LOG_FORMAT_FILE, datefmt=DATE_FORMAT)
    console_formatter = logging.Formatter(LOG_FORMAT_CONSOLE, datefmt=DATE_FORMAT)

    fh = logging.FileHandler(log_filename, encoding='utf-8')
    fh.setLevel(log_level)
    fh.setFormatter(file_formatter)

    sh = logging.StreamHandler()
    sh.setLevel(log_level)
    sh.setFormatter(console_formatter)

    root.addHandler(fh)
    root.addHandler(sh)

    _initialized = True
    return _log_dir


def get_log_dir() -> str:
    """Return the current session log directory path."""
    if _log_dir is None:
        raise RuntimeError(
            "Logging has not been initialized. Call setup_logging() first."
        )
    return _log_dir


def add_callback(callback, level=logging.INFO):
    """Register a callback as a log handler on the root logger.

    The callback receives a single string argument formatted as
    ``[LEVEL] message``.  This replaces the old addCallback() API
    with a proper logging.Handler.

    Returns:
        The created CallbackHandler (can be removed later if needed).
    """
    handler = CallbackHandler(callback, level)
    handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(handler)
    return handler