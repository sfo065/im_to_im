import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from shapely import LineString, Point, Polygon
from shapely.geometry import Polygon
sys.path.append(os.path.abspath('../raw_to_map'))
from sfkb_to_image_coordinates.sfkb_to_graph_dup import get_camera_extrinsics, utm_to_image
from tqdm import tqdm
from collections import defaultdict
import rasterio
from rasterio.windows import Window

def point_exists_in_dict(existing_dict, new_point, tolerance):
    for image_id, point in existing_dict.items():
        if point.distance(new_point) <= tolerance:
            return image_id  # Return the key (image) where the point matches
    return None

# Function to add a match to the list of dictionaries
def add_match(match_list, im1, pt1, im2, pt2, tolerance=1e-6):
    match_found = False
    for match_dict in match_list:
        # Check if pt1 already exists in the match dictionary
        existing_image = point_exists_in_dict(match_dict, pt1, tolerance)
        if existing_image:
            # Add the new point-image pair (im2: pt2) to the existing match
            match_dict[im2] = pt2
            match_found = True
            break
        # Also check if pt2 exists (and add pt1 if necessary)
        existing_image = point_exists_in_dict(match_dict, pt2, tolerance)
        if existing_image:
            match_dict[im1] = pt1
            match_found = True
            break
    
    # If no existing match was found, create a new match dictionary
    if not match_found:
        new_match = {im1: pt1, im2: pt2}
        match_list.append(new_match)
    
def add_single_point(match_list, im1, p1, tolerance=1e-6):
    """
    Adds a single point {im1: p1} to the match list if it does not already exist.

    Args:
        match_list: List of dictionaries storing matched points.
        im1: Image ID of the point.
        p1: Shapely Point to be added.
        tolerance: Distance tolerance to determine point equality.

    Returns:
        None (modifies match_list in place).
    """
    point_exists = False

    # Iterate through all dictionaries in match_list
    for match_dict in match_list:
        for image_id, point in match_dict.items():
            # Check if the current point already exists (within tolerance)
            if image_id == im1 and point.distance(p1) <= tolerance:
                point_exists = True
                break  # Exit inner loop
        if point_exists:
            break  # Exit outer loop if point is found

    # If the point doesn't exist, add it as a new dictionary
    if not point_exists:
        match_list.append({im1: p1})


def load_buildings(folder_path="../results/objects"):
    # List to store all loaded objects
    loaded_objects = []
    
    # Iterate over all files in the folder
    for file_name in tqdm(os.listdir(folder_path)):
        # Check if the file has a .json extension
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                loaded_object = json.load(file)                
                loaded_objects.append(loaded_object)
    print(f'Loaded {len(loaded_objects)} building objects from {folder_path}' )
    return loaded_objects


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
    if any(isinstance(point, Point) for point in points):
        points = [np.array(point.xy).squeeze() for point in points if isinstance(point, Point)]
    else:
        pass
    # Load settings from the JSON file located at the specified root directory.
    with open(os.path.join(root, 'settings.json'), 'r') as f:
        settings = json.load(f)
    
    # Retrieve the camera matrices for image 1 (P1) and image 2 (P2).
    P1 = camera_matrix(iid1, settings['inpho_path'])
    P2 = camera_matrix(iid2, settings['inpho_path'])
    
    # Compute the fundamental matrix (F) that relates the two images.
    F = fundamental_matrix(P1, P2)
    
    # Initialize an output list to store the epipolar line coefficients.
    output = {}
    x = np.array((0, settings['image_width']))
    # Iterate over each point in the provided list of points from image 1.
    for i, point in enumerate(points):
        # Convert the point to homogeneous coordinates by appending a 1.
        homogenous_point = np.hstack([point, np.array(1)])
        
        # Compute the coefficients [a, b, c] of the epipolar line in image 2.
        a, b, c = F @ homogenous_point
        y = -(a*x + c)/b    
        
        # Append the computed coefficients to the output list.
        output[Point(point)] = LineString(((x[0], y[0]), (x[1], y[1])))
          
    
    return output


def overlapping_polygons(sosi_path):

    sosi_path = os.path.join(sosi_path, 'DEKNINGSOVERSIKT')
    sosi_file = next((f for f in os.listdir(sosi_path) if f.endswith('Vertikalbilde.sos')), None)
    sosi_path = os.path.join(sosi_path, sosi_file)

    coverage_dict = {}
    current_id = None
    current_coords = []
    coords_start = False

    # Parse the SOSI file
    with open(sosi_path, 'r', encoding='latin-1') as infile:
        for line in infile:
            line = line.strip()
            if line.startswith('.FLATE'):
                if current_id and current_coords:
                    coverage_dict[current_id] = current_coords
                current_id = None
                current_coords = []
            elif line.startswith('...BILDEFILRGB'):
                try:
                    current_id = line.split('"')[1].split('.')[0]
                except:
                    current_id = line.split(' ')[1].split('.')[0]
            elif line.startswith('..NØ'):
                coords_start = True
            elif coords_start and line[0].isdigit():
                coords = [float(coord) for coord in line.split()][::-1]
                if len(coords) == 2:  # Ensure we only add pairs of coordinates
                    current_coords.append(coords)
            elif line.startswith('.KURVE'):
                coords_start = False

    # Add the last image if it exists
    if current_id and current_coords:
        coverage_dict[current_id] = current_coords

    # Process coverage areas
    for image_id, coords in coverage_dict.items():

        coverage_dict[image_id] = Polygon(np.array(coords))
    
    image_ids = list(coverage_dict.keys())
    overlapping_ids = {id:list() for id in image_ids}
    
    for i, id1 in enumerate(image_ids):
        area = coverage_dict[id1].area
        poly1 = coverage_dict[id1]
        for id2 in image_ids[i + 1:]:  # Avoid redundant checks
            poly2 = coverage_dict[id2]
            intersection = poly1.intersection(poly2)
            if not intersection.is_empty and intersection.area/area > 0.05:
                overlapping_ids[id1].append(id2)
                overlapping_ids[id2].append(id1)

    return coverage_dict, overlapping_ids

    

def simplify_polygon(polygon):
    """
    Simplifies a Shapely polygon by retaining only its extreme points (four corners).

    Args:
        polygon: A Shapely Polygon object.

    Returns:
        A simplified Shapely Polygon with four corners.
    """
    # Extract exterior coordinates as a list of (x, y) tuples
    x_coords, y_coords = polygon.exterior.xy
    points = list(zip(x_coords, y_coords))

    # Identify extreme points
    min_x = min(points, key=lambda p: p[0])
    max_x = max(points, key=lambda p: p[0])
    min_y = min(points, key=lambda p: p[1])
    max_y = max(points, key=lambda p: p[1])

    # Collect the four corners
    extreme_points = {min_x, max_x, min_y, max_y}

    # Convert set back to a list for ordered construction
    simplified_points = list(extreme_points)
    simplified_points.append(simplified_points[0])  # Close the polygon

    # Create and return the simplified polygon
    return Polygon(simplified_points)

def snap_to_border(points, tolerance=1000):
    x_min = 0
    x_max = 17004
    y_min = 0
    y_max = 26460
    for i, (x, y) in enumerate(points):
        # Adjust x values
        if abs(x - x_min) <= tolerance:
            points[i, 0] = x_min
        elif abs(x - x_max) <= tolerance:
            points[i, 0] = x_max

        # Adjust y values
        if abs(y - y_min) <= tolerance:
            points[i, 1] = y_min
        elif abs(y - y_max) <= tolerance:
            points[i, 1] = y_max

    return points

def get_overlap(iid1, iid2):
    '''
    finds the overlap between images with ids given in args. 
    returns the overlapping area as a shapely polygon in the coordinates of image 1
    '''
    with open('../AgderOst/settings.json') as f:
        settings = json.load(f)
    coverage, overlap = overlapping_polygons('../AgderOst/FG-14583_AgderØstGSD07')
    if iid2 in overlap[iid1]:

        intersect = simplify_polygon(coverage[iid1].intersection(coverage[iid2]))
        utm_box = np.vstack([np.vstack(intersect.exterior.xy), np.zeros(5)]).T
        box = Polygon(snap_to_border(utm_to_image(utm_box, iid1, settings['inpho_path']))).convex_hull
    else:
        box = None
    return box

def find_points_in_overlap(points1, points2, iid1, iid2):
    box1 = get_overlap(iid1, iid2)
    box2 = get_overlap(iid2, iid1)
    if box1:
        points_within_box1 = [point for point in points1 if point.within(box1)]
        points_within_box2 = [point for point in points2 if point.within(box2)]
        return points_within_box1, points_within_box2
    else:
        return None
    
def match_points(p1, p2, iid1, iid2, tol=1e-2):
    """
    Matches points from image 1 to points in image 2 based on epipolar geometry.

    Args:
        p1: List of Shapely points in image 1.
        p2: List of Shapely points in image 2.
        iid1: Image ID of image 1.
        iid2: Image ID of image 2.

    Returns:
        single_match: List of unique point matches as dictionaries.
        mult_match: List of multiple matches as dictionaries.
        no_match: List of points from image 1 with no match.
    """
    overlap = find_points_in_overlap(p1, p2, iid1, iid2)
    if overlap:
        points1, points2 = overlap
        lines = epipolar_line(points2, iid2, iid1)
        matches = defaultdict(set)

        # Track which points in points1 are matched
        matched_points = set()

        # Match points within the tolerance
        for pt1 in points1:
            for pt2, line in lines.items():
                if pt1.distance(line) < tol:
                    matches[pt2].add(pt1)
                    matched_points.add(pt1)

        matches = dict(matches)
    
        # Separate into single and multiple matches
        single_matches = {}
        mult_matches = {}
        for pt2, pt1_set in matches.items():
            if len(pt1_set) == 1:
                pt1 = list(pt1_set)[0]
                single_matches[pt1] = pt2
            elif len(pt1_set) > 1:
                mult_matches[tuple(pt1_set)] = pt2

        # Find unmatched points in points1
        unmatched_points = set(points1) - matched_points

        return single_matches, mult_matches, unmatched_points
        
    
    else: 
        print (f'Images {iid1} and {iid2} do not overlap')
        return

def get_true_matches(iid1, iid2, buildings):
    true_matches = dict()
    for building in buildings:
        iids = building['corners'].keys()
        if iid1 in iids and iid2 in iids:
            p1 = Point(np.round(np.mean(np.array(list(building['corners'][iid1])), axis=0), 6))
            p2 = Point(np.round(np.mean(np.array(list(building['corners'][iid2])), axis=0), 6))
            true_matches[p1] = p2
    return true_matches

def main(overlap, buildings_pt, print_metrics=False, buildings_true=None):
    # Set to track processed pairs
    processed_pairs = set()
    matches = []
    mult_matches = []

    # Iterate over the dictionary and process overlaps
    for iid1, overlapping_iids in overlap.items():
        for iid2 in overlapping_iids:
            # Create a pair tuple (ordered to avoid duplication)
            pair = tuple(sorted([int(iid1.split('_')[-1]), int(iid2.split('_')[-1])]))
            # Check if the pair has already been processed
            if pair not in processed_pairs:
                # Call the matching function
                
                inferred_matches, mult_match, unmatched = match_points(buildings_pt[iid1], buildings_pt[iid2], iid1, iid2)
                # Add the pair to the processed set
                processed_pairs.add(pair)
                
                if print_metrics:
                    true_matches = get_true_matches(iid1, iid2, buildings_true)                    
                    TP = 0
                for m in inferred_matches:
                    add_match(matches, iid1, m, iid2, inferred_matches[m])
                    if print_metrics:
                        if inferred_matches[m] == true_matches[m]:
                            TP += 1

                for p in unmatched:
                    add_single_point(matches, iid1, p)

                if len(mult_match) > 0:
                    mult_matches.append({iid1:tuple(mult_match.keys()), iid2:tuple(mult_match.values())})
                
                if print_metrics:
                    mult_tot = sum([len(m) for m in mult_match])
                    precision = round(TP/len(inferred_matches), 4)
                    recall = round(TP/len(true_matches), 4)
                    F1 = round(2*precision*recall/(precision+recall), 4)
                    
                if print_metrics:
                    print (f'{iid1.split("_")[-1]}-{iid2.split("_")[-1]}: Precision:{precision} -- Recall {recall} --F1: {F1} -- N matches: {len(inferred_matches)} -- mult match:{mult_tot}\n')
    return matches, mult_matches

def plot_matches(match, width, cog_path):
    n = len(match)
    _, axs = plt.subplots(1, n, figsize=(4*n, 4))
    for ax, iid in zip(axs.flatten(),match):
        x, y = match[iid].xy
        window = Window(x[0]-width/2, y[0]-width/2, width, width)

        with rasterio.open(f"{cog_path}/{iid}.tif") as src:
            image = src.read(window=window)
            image = np.moveaxis(image, 0, -1)
        ax.imshow(image)
        ax.scatter(width/2, width/2, c='red')
