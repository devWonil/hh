#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import time
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"  # Hide welcome message
import pygame

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from client import SULIVAN_Client

import logging

logger = logging.getLogger(__name__)
from utils.shape import OBB


class Image:
    """Simple oriented image element that can be drawn on the scenario screen."""

    def __init__(
        self,
        client: "SULIVAN_Client",
        name: str,
        image_path: str,
        bbox,
        angle_deg: float = 0,
        display_delay: int | float = 0,
        visible: bool = True,
    ):
        self._client = client
        self.scenario = client.scenario

        self.name = name
        self.bbox = self._normalize_bbox(bbox)
        self.angle_deg = angle_deg or 0
        self.display_delay_ms = int(display_delay) if display_delay else 0
        self.visible = bool(visible)

        self._source_surface: pygame.Surface | None = None
        self.image_path = ""
        self.obb: OBB | None = None

        self.set_image(image_path)
        logger.debug(f"Image '{self.name}' initialized.")

    @staticmethod
    def _normalize_bbox(bbox):
        try:
            x, y, w, h = bbox
        except (TypeError, ValueError):
            raise ValueError("Image bbox must contain four numeric values (X, Y, W, H).")
        return (round(x), round(y), round(w), round(h))

    def _rebuild_obb(self):
        if self._source_surface is None:
            return

        x, y, w, h = self.bbox
        center_x = x + w / 2
        center_y = y + h / 2

        self.obb = OBB(center_x, center_y, w, h, self.angle_deg)
        self.obb.add_image("default", self._source_surface)
        self.obb.set_active_image("default")

    def set_bbox(self, bbox):
        """Update the bounding box in pixel coordinates."""
        self.bbox = self._normalize_bbox(bbox)
        self._rebuild_obb()

    def set_angle(self, angle_deg):
        """Update the rotation angle of the image."""
        self.angle_deg = angle_deg or 0
        if self.obb is not None:
            self.obb.update_angle(self.angle_deg)

    def set_image(self, image_path: str):
        """Load image from scenario-aware path and apply to the OBB."""
        if not image_path:
            raise ValueError("Image path must be provided for Image object.")

        try:
            resolved_path = self.scenario.path_util.getPath(image_path)
        except FileNotFoundError as exc:
            logger.error(str(exc))
            raise

        surface = pygame.image.load(resolved_path).convert_alpha()
        self._source_surface = surface
        self.image_path = image_path
        self._rebuild_obb()

    def set_visible(self, visible: bool):
        self.visible = bool(visible)

    def set_display_delay(self, milliseconds: int | float):
        self.display_delay_ms = int(milliseconds) if milliseconds else 0

    def draw(self, surface: pygame.Surface):
        if not self.visible or self.obb is None:
            return

        if self.display_delay_ms > 0:
            elapsed_ms = (time.time() - self.scenario.step_start_time) * 1000
            if elapsed_ms < self.display_delay_ms:
                return

        self.obb.draw(surface)
