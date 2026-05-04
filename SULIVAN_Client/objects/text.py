#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"  # Hide welcome message

import pygame

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from client import SULIVAN_Client

import utils.coord_utils as coord
import logging

logger = logging.getLogger(__name__)


DEFAULT_FONT_PATH = "assets/fonts/NotoSansKR-Medium.ttf"
DEFAULT_COLOR = (255, 255, 255)
VALID_ALIGNS = {"left", "center", "right"}


class Text:
    """Renderable text that follows scenario lifecycle."""

    def __init__(self,
                 client: "SULIVAN_Client",
                 name: str,
                 text: str,
                 position_world,
                 size_world,
                 align: str | None = None,
                 color=DEFAULT_COLOR):
        self._client = client
        self.scenario = client.scenario
        self.name = name
        self.color = self._normalize_color(color)

        self.align = self._normalize_align(align)
        self.position_px = self._world_to_pixel_point(position_world)
        self.font = self._create_font(size_world)

        self.text_surface = None
        self.text_rect = None
        self.setText(text)

    def _normalize_align(self, align):
        if align is None:
            return "left"

        align_lower = align.lower()
        if align_lower in VALID_ALIGNS:
            return align_lower

        raise ValueError(
            f"Invalid align value '{align}' for text '{self.name}'. Supported: {sorted(VALID_ALIGNS)}"
        )

    def _normalize_color(self, color):
        if color is None:
            return DEFAULT_COLOR

        if isinstance(color, str):
            hex_value = color.lstrip('#')
            if len(hex_value) != 6:
                raise ValueError(f"Color '{color}' must be a hex string like '#RRGGBB'.")
            try:
                rgb = tuple(int(hex_value[i:i + 2], 16) for i in range(0, 6, 2))
            except ValueError as err:
                raise ValueError(f"Color '{color}' contains invalid hex digits.") from err
            return rgb

        if isinstance(color, (list, tuple)):
            if len(color) != 3:
                raise ValueError(f"Color for text '{self.name}' must have three components.")
            try:
                rgb = tuple(int(component) for component in color)
            except (TypeError, ValueError) as err:
                raise ValueError("RGB color components must be integers.") from err
            if any(component < 0 or component > 255 for component in rgb):
                raise ValueError("RGB color components must be between 0 and 255.")
            return rgb

        raise ValueError(
            "Color must be provided as an RGB list/tuple or hex string like '#RRGGBB'."
        )

    def _world_to_pixel_point(self, position_world):
        if position_world is None or len(position_world) != 2:
            raise ValueError(f"Text '{self.name}' must define position as [X, Y].")

        x_cm, y_cm = position_world
        px = round(coord.fromWorldToPixelWidth(x_cm))
        py = round(coord.fromWorldToPixelHeight(y_cm))
        return px, py

    def _create_font(self, size_world):
        if size_world is None:
            raise ValueError(f"Text '{self.name}' must define size in cm.")

        size_px = coord.fromWorldToPixelHeight(size_world)
        size_px = max(1, int(round(size_px)))

        try:
            return pygame.font.Font(DEFAULT_FONT_PATH, size_px)
        except FileNotFoundError:
            logger.error(f"Font file not found: {DEFAULT_FONT_PATH}. Using system default font.")
            return pygame.font.SysFont(None, size_px)

    def setText(self, new_text: str):
        """Update the displayed text and recalculate surface and rect."""
        if new_text is None:
            new_text = ""
        self._text = new_text

        lines = new_text.splitlines() or [""]
        line_height = self.font.get_linesize()
        line_height = max(1, line_height)

        max_width = 1
        rendered_lines = []
        for line in lines:
            rendered = self.font.render(line, True, self.color)
            rendered_lines.append(rendered)
            max_width = max(max_width, rendered.get_width())

        total_height = line_height * len(rendered_lines)
        surface = pygame.Surface((max_width, total_height), pygame.SRCALPHA)

        y_offset = 0
        for rendered in rendered_lines:
            surface.blit(rendered, (0, y_offset))
            y_offset += line_height

        self.text_surface = surface
        self._update_rect()

    def _update_rect(self):
        if self.text_surface is None:
            self.text_rect = None
            return

        rect = self.text_surface.get_rect()
        if self.align == "left":
            rect.midleft = self.position_px
        elif self.align == 'center':
            rect.center = self.position_px
        elif self.align == "right":
            rect.midright = self.position_px
        else:
            # Fallback safety: default to left
            rect.midleft = self.position_px

        self.text_rect = rect

    def draw(self, surface):
        if self.text_surface is None or self.text_rect is None:
            return
        surface.blit(self.text_surface, self.text_rect)
