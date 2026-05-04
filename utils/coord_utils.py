#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

# There are 3 coordinates in this system.
#  1. World coord (Unit: cm) - Workspace
#  2. Pixel coord (Unit: pixel) - Area of projection
#  3. Camera coord (Unit: pixel) - FOV of camera

import config
import numpy as np

M_cp = config.M_cp
M_pc = np.linalg.inv(M_cp)

"""
 From real-world coordinate to pixel(projector) coordinate
"""
def fromWorldToPixelWidth(width):
    '''Calculate pixel coordinate from real coordinate. (Width)'''
    return width  * (config.PIXELS_WIDTH  / config.LENGTH_WIDTH)
def fromWorldToPixelHeight(height):
    '''Calculate pixel coordinate from real coordinate. (Height)'''
    return height * (config.PIXELS_HEIGHT / config.LENGTH_HEIGHT)

# combined code
def fromWorldToPixelBbox(bbox):
    '''Calculate pixel coordinate from real coordinate. (bounding box: [X, Y, W, H])'''
    bbox_result = []
    bbox_result.append(round(fromWorldToPixelWidth( bbox[0])))
    bbox_result.append(round(fromWorldToPixelHeight(bbox[1])))
    bbox_result.append(round(fromWorldToPixelWidth( bbox[2])))
    bbox_result.append(round(fromWorldToPixelHeight(bbox[3])))

    return bbox_result


"""
 From camera coordinate to pixel(projector) coordinate
"""
def fromCameraToPixelPoint(point, rounded=False):
    '''Calculate pixel coordinate from camera coordinate. (point: [X, Y])'''
    # If not using camera projection, return the original point
    if not config.use_camera_projection:
        return point

    point_homo = np.ones((3, 1))
    point_homo[0] = point[0]
    point_homo[1] = point[1]

    calc_point = M_cp @ point_homo
    calc_point /= calc_point[2] # Normalize

    if rounded:
        return [round(calc_point[0,0]), round(calc_point[1,0])]
    else:
        return [calc_point[0,0], calc_point[1,0]]

def fromCameraToPixelBbox(bbox):
    ''' Calculate projector-based coord from camera-based coord (bbox: [X, Y, W, H]) '''
    bbox_result = []

    point1 = bbox[0:2]
    point2 = [bbox[0] + bbox[2], bbox[1] + bbox[3]]

    point1_pixel = fromCameraToPixelPoint(point1)
    point2_pixel = fromCameraToPixelPoint(point2)


    bbox_result.append(round(point1_pixel[0]))
    bbox_result.append(round(point1_pixel[1]))
    bbox_result.append(round(point2_pixel[0] - point1_pixel[0]))
    bbox_result.append(round(point2_pixel[1] - point1_pixel[1]))

    return bbox_result

"""
 From pixel(projector) coordinate to camera coordinate
"""
def fromPixelToCameraPoint(point, rounded=False):
    '''Calculate pixel coordinate from camera coordinate. (point: [X, Y])'''
    # If not using camera projection, return the original point
    if not config.use_camera_projection:
        return point

    point_homo = np.ones((3, 1))
    point_homo[0] = point[0]
    point_homo[1] = point[1]

    calc_point = M_pc @ point_homo
    calc_point /= calc_point[2] # Normalize

    if rounded:
        return [round(calc_point[0,0]), round(calc_point[1,0])]
    else:
        return [calc_point[0,0], calc_point[1,0]]
    
def fromPixelToCameraBbox(bbox):
    ''' Calculate camera-based coord from projector-based coord (bbox: [X, Y, W, H]) '''
    bbox_result = []

    point1 = bbox[0:2]
    point2 = [bbox[0] + bbox[2], bbox[1] + bbox[3]]

    point1_pixel = fromPixelToCameraPoint(point1)
    point2_pixel = fromPixelToCameraPoint(point2)


    bbox_result.append(round(point1_pixel[0]))
    bbox_result.append(round(point1_pixel[1]))
    bbox_result.append(round(point2_pixel[0] - point1_pixel[0]))
    bbox_result.append(round(point2_pixel[1] - point1_pixel[1]))

    return bbox_result

def fromPixelToCameraCorners(corners):
    '''Transform a list of [x, y] corner points from projector pixel coords to camera coords.'''
    return [fromPixelToCameraPoint(p, rounded=True) for p in corners]