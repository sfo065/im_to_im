import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from shapely import LineString
sys.path.append(os.path.abspath('../raw_to_map'))
from sfkb_to_image_coordinates.sfkb_to_graph_dup import get_camera_extrinsics

line = lambda a, b, c, x:  -(a*x + c)/b

def camera_matrix(image_id, inpho_path):
    '''
    Calculates the camera matrix for the camera used in the image 
    given by the image id. 
    '''
    #image dims
    ny, nx = 26460, 17004

    #intrinsics:
    focal_length = int(100.5*1e-3/4e-6) # image coordinates
    ppa = np.array((nx/2, ny/2)) # image coordinates

    #extrinsics:
    cx, cy, cz, R = get_camera_extrinsics(image_id, inpho_path) #camera posistion  and rotation in UTM-coordinates
    C = np.array((cx, cy, cz)).reshape(-1, 1) #camera postion in UTM
    
    extrinsic_matrix = np.vstack([np.hstack([R, -R@C]),np.array((0, 0, 0, 1))])
    intrinsic_matrix = np.array(((focal_length, 0, ppa[0], 0),(0, focal_length, ppa[1], 0),(0, 0, 1, 0)))
    CM = intrinsic_matrix@extrinsic_matrix
    Tx = np.array([[-1,  0,  nx],
                   [ 0,  1,   0],
                   [ 0,  0,   1]])
    CM = Tx@CM
    
    return CM

def fundamental_matrix(P1, P2):
    # Compute the camera center C1 of P1
    _, _, Vt = np.linalg.svd(P1)
    C1 = Vt[-1]  # Last row of Vt corresponds to the null space
    C1 = C1 / C1[-1]  # Normalize to make it homogeneous

    # Compute epipole e2 in the second camera
    e2 = P2 @ C1
    e2 = e2 / e2[-1]  # Normalize to make it homogeneous

    # Skew-symmetric matrix for e2
    e2_skew = np.array([
        [0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0]
    ])

    # Compute the fundamental matrix
    F = e2_skew @ P2 @ np.linalg.pinv(P1)
    
    return F #/ F[-1, -1]  # Normalize the fundamental matrix

def line_coeffs_to_shapely(a, b, c):
    if b != 0:  # Non-vertical line
        # Generate two points
        x1, y1 = 0, -c / b
        x2, y2 = 1, -(a + c) / b
    else:  # Vertical line
        x1 = x2 = -c / a
        y1, y2 = 0, 1

    # Create a Shapely LineString
    return LineString([(x1, y1), (x2, y2)])

def epipolar_line(points, iid1, iid2, root='../AgderOst'):
    '''
    Compute the epipolar lines corresponding to a set of points in image 1.

    Parameters:
    - points: list of 2D points in image 1, represented as [x, y] coordinates.
    - iid1: identifier for the first image (image 1).
    - iid2: identifier for the second image (image 2).
    - root: root directory containing the 'settings.json' file (default: '../AgderOst').

    Returns:
    - output: list of coefficients [a, b, c] for the epipolar lines in image 2.
              Each line is represented by the equation: ax + by + c = 0.
    '''
    # Load settings from the JSON file located at the specified root directory.
    with open(os.path.join(root, 'settings.json'), 'r') as f:
        settings = json.load(f)
    
    # Retrieve the camera matrices for image 1 (P1) and image 2 (P2).
    P1 = camera_matrix(iid1, settings['inpho_path'])
    P2 = camera_matrix(iid2, settings['inpho_path'])
    
    # Compute the fundamental matrix (F) that relates the two images.
    F = fundamental_matrix(P1, P2)
    
    # Initialize an output list to store the epipolar line coefficients.
    output = list()
    
    # Iterate over each point in the provided list of points from image 1.
    for i, point in enumerate(points):
        # Convert the point to homogeneous coordinates by appending a 1.
        homogenous_point = np.hstack([point, np.array(1)])
        
        # Compute the coefficients [a, b, c] of the epipolar line in image 2.
        coeffs = F @ homogenous_point
        
        # Append the computed coefficients to the output list.
        output.append(coeffs)
    
    # Return the list of epipolar line coefficients.
    return output

