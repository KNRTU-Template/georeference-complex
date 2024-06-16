from rasterio import Affine
import numpy as np
import cv2
from multiprocessing.pool import ThreadPool
import time
import math
from asift import Timer, image_resize, init_feature, filter_matches, affine_detect
import datetime

MAX_SIZE = 756

FLANN_INDEX_KDTREE = 1

FLANN_INDEX_LSH = 6

def linear_stretch(band, min_percent=2, max_percent=98):
    min_val = np.percentile(band, min_percent)
    max_val = np.percentile(band, max_percent)
    return np.clip((band - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)

def check_and_resize(orig_crop_, ratio=1):
    if orig_crop_.shape[0] > MAX_SIZE or orig_crop_.shape[1] > MAX_SIZE:
        ratio = MAX_SIZE / min(orig_crop_.shape[:2])
        crop_ = cv2.resize(orig_crop_, (int(orig_crop_.shape[1] * ratio), int(orig_crop_.shape[0] * ratio)), interpolation=cv2.INTER_AREA)
    else:
        crop_ = orig_crop_
    return crop_, ratio

def crop_map(image, crop_height, crop_width, overlap):
    _, height, width = image.shape
    step_y = crop_height - overlap
    step_x = crop_width - overlap
    return [(image[:, y:min(y + crop_height, height), x:min(x + crop_width, width)], x, y)
            for y in range(0, height - overlap, step_y) for x in range(0, width - overlap, step_x)]

def get_adjusted_affine(geo_transform, x_offset, y_offset):
    original_affine = Affine.from_gdal(*geo_transform)
    adjusted_affine = original_affine * Affine.translation(x_offset, y_offset)
    return adjusted_affine

def get_geo_corners(corners, geo_transform, offsets, scale):
    affine_transform = get_adjusted_affine(geo_transform, *offsets)
    geo_corners = [affine_transform * (x * scale, y * scale) for x, y in corners]
    return geo_corners

def calculate_center(corners):
    x_coords, y_coords = zip(*corners)
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    return center_x, center_y

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def pixel_to_coord(x, y, geo_transform, scale_x=10, scale_y=10):
    xoff, a, b, yoff, d, e = geo_transform
    xp = scale_x * x - b * y + xoff
    yp = d * x + -scale_y * y + yoff
    return (xp, yp)

def pixel_to_coord2(top_left, x, y, geo_transform, scale_x=10, scale_y=10):
    xoff, a, b, yoff, d, e = geo_transform
    xp = scale_x * x - b * y + top_left[0]
    yp = d * x + -scale_y * y + top_left[1]
    return (xp, yp)

def polygon_area(coords):
    n = len(coords)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]
    return abs(area) / 2

def crop_map(image, crop_height, crop_width):
    _, height, width = image.shape
    return [(image[:, y:min(y + crop_height, height), x:min(x + crop_width, width)], x, y)
            for y in range(0, height, crop_height) for x in range(0, width, crop_width)]

results = []

start = time.time()

def find_coordinates(layout, crop_arr):
    start = time.time()
    start_dt = datetime.datetime.now()
    
    gt = layout.GetGeoTransform()
    full_layout_arr = layout.ReadAsArray()
    part_size = 3000
    layout_crops = crop_map(full_layout_arr, part_size, part_size)

    crop_normalized = np.stack([linear_stretch(band) for band in crop_arr], axis=-1)
    orig_crop_ = cv2.cvtColor(crop_normalized, cv2.COLOR_RGB2GRAY)
    crop_, ratio_1 = check_and_resize(orig_crop_)

    best_distance = float('inf')
    best_part_id = None
    best_geo_corners = np.zeros((4, 2))

    detector_name = "sift-flann"
    detector, matcher = init_feature(detector_name)

    for i, (layout_arr, offset_x, offset_y) in enumerate(layout_crops):
        print(offset_x, offset_y,)
        print(layout_arr.shape)
        layout_crop_normalized = np.stack([linear_stretch(band) for band in layout_arr[:3]], axis=-1)
        orig_layout_ = cv2.cvtColor(layout_crop_normalized, cv2.COLOR_RGB2GRAY)
        layout_, ratio_2 = check_and_resize(orig_layout_)

        with ThreadPool(processes=cv2.getNumberOfCPUs()) as pool:
            kp1, desc1 = affine_detect(detector, crop_, pool)
            kp2, desc2 = affine_detect(detector, layout_, pool)
            raw_matches = matcher.knnMatch(desc1, desc2, k=2)

        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches, ratio=0.7)
        print(i, len(kp_pairs))
        if len(p1) >= 4:
            # for index in range(len(p1)):
            #     pt = p1[index]
            #     p1[index] = pt / ratio_1

            for index in range(len(p2)):
                pt = p2[index]
                p2[index] = pt / ratio_2

            for index in range(len(kp_pairs)):
                element = kp_pairs[index]
                kp1, kp2 = element

                new_kp1 = cv2.KeyPoint(kp1.pt[0] / ratio_1, kp1.pt[1] / ratio_1, kp1.size)
                new_kp2 = cv2.KeyPoint(kp2.pt[0] / ratio_2, kp2.pt[1] / ratio_2, kp2.size)

                kp_pairs[index] = (new_kp1, new_kp2)

            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
            if H is not None:
                top_left = pixel_to_coord(offset_x, offset_y, gt)
                top_right = pixel_to_coord(offset_x + part_size, offset_y, gt)
                bottom_right = pixel_to_coord(offset_x + part_size, offset_y + part_size, gt)
                bottom_left = pixel_to_coord(offset_x, offset_y + part_size, gt)

                part_geo_corners = [top_left, top_right, bottom_right, bottom_left]

                crop_corners = np.array([[0, 0],
                         [crop_.shape[1]-1, 0],
                         [crop_.shape[1]-1, crop_.shape[0]-1],
                         [0, crop_.shape[0]-1]], dtype='float32').reshape(-1, 1, 2)

                transformed_corners = cv2.perspectiveTransform(crop_corners, H)

                geo_corners = [pixel_to_coord2(top_left, corner[0][0], corner[0][1], gt, 10, 10) for corner in transformed_corners]
                print(part_geo_corners)
                print(geo_corners)

                crop_center = calculate_center(geo_corners)

                part_center = calculate_center(part_geo_corners)

                distance = euclidean_distance(crop_center, part_center)

                geo_corners = np.array(geo_corners)

                # area = polygon_area(geo_corners)

                # print(distance / area)

                # if distance / area < best_distance:
                #     best_distance = distance / area
                #     best_part_id = i
                #     best_geo_corners = geo_corners

                # area = polygon_area(geo_corners)

                print(distance / len(kp_pairs))

                if distance / len(kp_pairs) < best_distance:
                    best_distance = distance / len(kp_pairs)
                    best_part_id = i
                    best_geo_corners = np.round(geo_corners, 3)
                
                print("***************************")
                print(H)
                print(transformed_corners)
                print("***************************")

                print("------------------------------\n")
                # results.append([geo_corners[0][0], geo_corners[0][1]])
                # pd.DataFrame(results,
                #   columns=["corner1_x", "corner1_y"]).to_csv("/content/drive/MyDrive/LCT/results_3_1_49.csv", index=False)

    print("THE BEST")
    finish = time.time()
    finish_dt = datetime.datetime.now()
    exec_time = round(finish - start, 3)
    print(best_part_id)
    print(best_distance)
    print(best_geo_corners)
    print(exec_time)

    return best_geo_corners, start_dt, finish_dt