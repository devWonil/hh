#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import pygame
import math

import logging

logger = logging.getLogger(__name__)

class OBB:
    def __init__(self, center_x, center_y, width, height, angle_deg=0,
                 color=None, outline_color=None, outline_thickness=None, images=None):
        self.center = pygame.math.Vector2(center_x, center_y)
        self.width = width
        self.height = height
        self.angle_deg = angle_deg  # Angle in degrees
        
        self.images = images if images is not None else {}
        self.scaled_images = {}
        self.active_image_key = None
        self.active_scaled_image = None

        if self.images:
            for key, img in self.images.items():
                self.scaled_images[key] = pygame.transform.smoothscale(img, (self.width, self.height))
            # Set the first image as the default active image
            self.active_image_key = next(iter(self.images))
            self.active_scaled_image = self.scaled_images[self.active_image_key]

        self.points = self._calculate_points() # 4 vertices
        self.axes = self._calculate_axes()   # 2 axis vectors
        self.color = color
        self.outline_color = outline_color
        self.outline_thickness = outline_thickness

    def _calculate_points(self):
        """Calculate 4 vertices based on center, size, and angle."""
        angle_rad = math.radians(-self.angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        half_w = self.width / 2
        half_h = self.height / 2

        # Local vertex coordinates when not rotated
        local_points = [
            pygame.math.Vector2(-half_w, -half_h),
            pygame.math.Vector2(half_w, -half_h),
            pygame.math.Vector2(half_w, half_h),
            pygame.math.Vector2(-half_w, half_h)
        ]

        # Rotate each vertex and move to center position
        world_points = []
        for p in local_points:
            rotated_x = p.x * cos_a - p.y * sin_a
            rotated_y = p.x * sin_a + p.y * cos_a
            world_points.append(self.center + pygame.math.Vector2(rotated_x, rotated_y))
            
        return world_points

    def _calculate_axes(self):
        """Calculate the two axes (unit vectors) of the OBB."""
        p0, p1, p3 = self.points[0], self.points[1], self.points[3]
        axis1 = (p1 - p0).normalize()
        axis2 = (p3 - p0).normalize()
        return [axis1, axis2]
    
    def update_angle(self, angle_deg):
        """Update the angle and recalculate vertices and axes."""
        self.angle_deg = angle_deg
        self.points = self._calculate_points()
        self.axes = self._calculate_axes()

    def set_active_image(self, key):
        """Set the image to display."""
        if key in self.scaled_images:
            self.active_image_key = key
            self.active_scaled_image = self.scaled_images[key]
        else:
            logger.warning(f"Image key '{key}' not found.")

    def add_image(self, key, image_surface):
        """Add an image with a key and scale it to size."""
        if image_surface:
            self.images[key] = image_surface
            self.scaled_images[key] = pygame.transform.smoothscale(image_surface, (self.width, self.height))
            # Set as active image if it's the first image
            if self.active_image_key is None:
                self.set_active_image(key)

    def get_active_image_key(self):
        """Return the key of the active image."""
        return self.active_image_key

    def draw(self, surface):
        if self.active_scaled_image:
            rotated_image = pygame.transform.rotate(self.active_scaled_image, self.angle_deg)
            rect = rotated_image.get_rect(center=self.center)
            surface.blit(rotated_image, rect.topleft)
        
        elif self.color is not None:
            pygame.draw.polygon(surface, self.color, self.points, 0) # Filled polygon
        
        if self.outline_color is not None and self.outline_thickness is not None:
            pygame.draw.polygon(surface, self.outline_color, self.points, self.outline_thickness) # Outline

class Circle:
    def __init__(self, x, y, radius, color=None, outline_color=None, outline_thickness=None):
        self.center = pygame.math.Vector2(x, y)
        self.radius = radius
        self.color = color
        self.outline_color = outline_color
        self.outline_thickness = outline_thickness

    def project(self, axis):
        """Projects the circle onto the given axis and returns the minimum and maximum values."""
        center_projection = self.center.dot(axis)
        min_proj = center_projection - self.radius
        max_proj = center_projection + self.radius
        return min_proj, max_proj

    def update_pos(self, x, y):
        self.center.x = x
        self.center.y = y

    def draw(self, surface):
        if self.color is not None:
            pygame.draw.circle(surface, self.color, (int(self.center.x), int(self.center.y)), self.radius)
        if self.outline_color is not None and self.outline_thickness is not None:
            pygame.draw.circle(surface, self.outline_color, (int(self.center.x), int(self.center.y)), self.radius, self.outline_thickness)

def _project_obb_on_axis(obb: OBB, axis: pygame.math.Vector2):
    """Project all OBB vertices onto the axis, return (min, max)."""
    projections = [p.dot(axis) for p in obb.points]
    return min(projections), max(projections)

def check_collision_obb_obb(obb1: OBB, obb2: OBB) -> bool:
    """Detects collision between two OBBs using the Separating Axis Theorem."""
    for axis in obb1.axes + obb2.axes:
        min1, max1 = _project_obb_on_axis(obb1, axis)
        min2, max2 = _project_obb_on_axis(obb2, axis)
        if max1 < min2 or max2 < min1:
            return False
    return True

def check_collision_obb_circle(obb: OBB, circle: Circle) -> bool:
    """Detects collision between OBB and circle."""
    
    # 1. Transform the circle's center to OBB's local coordinate system
    #    Consider the OBB's center as the origin.
    vec_c_to_o = circle.center - obb.center # Vector from OBB center to circle center
    
    # 2. Project vec_c_to_o vector onto OBB's two axes (dot product) to get local coordinates
    #    This becomes the circle's center position in OBB's local coordinate system
    circle_local_x = vec_c_to_o.dot(obb.axes[0])
    circle_local_y = vec_c_to_o.dot(obb.axes[1])

    # 3. Find the closest point on OBB to the circle's center in OBB's local coordinate system
    half_w, half_h = obb.width / 2, obb.height / 2
    
    closest_x = max(-half_w, min(circle_local_x, half_w))
    closest_y = max(-half_h, min(circle_local_y, half_h))
    
    closest_point_local = pygame.math.Vector2(closest_x, closest_y)
    circle_center_local = pygame.math.Vector2(circle_local_x, circle_local_y)
    
    # 4. Calculate the distance between the closest point and circle's center
    distance_squared = (circle_center_local - closest_point_local).length_squared()
    
    # 5. Collision occurs if distance is less than or equal to circle's radius
    return distance_squared <= circle.radius**2