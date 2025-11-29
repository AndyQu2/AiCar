import cv2
import numpy as np


def get_interest_region(edges, color='yellow'):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    if color == 'yellow':
        polygon = np.array([[(0, height * 1 / 2),
                            (width * 1 / 2, height * 1 / 2),
                            (width * 1 / 2, height),
                            (0, height)]], dtype=np.int32)
    else:
        polygon = np.array([[(width * 1 / 2, height * 1 / 2),
                             (width, height * 1 / 2),
                             (width, height),
                             (0, height)]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edge = cv2.bitwise_and(edges, mask)
    return cropped_edge

def detect_line(edges):
    rho = 1
    angle = np.pi / 180
    min_thr = 10
    line = cv2.HoughLinesP(edges, rho, angle, min_thr, np.array([]),
                           minLineLength=8, maxLineGap=8)
    return line


def make_point(frame, lines):
    height, width, _ = frame.shape
    slope, intercept = lines
    y1 = height
    y2 = int(y1 * 1 / 2)
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [x1, y1, x2, y2]

def average_lines(frame, lines, direction='left'):
    lane_line = []

    if lines is None:
        print("No lines detected")
        return lane_line

    height, width, _ = frame.shape
    fits = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue

            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if direction == 'left' and slope < 0:
                fits.append((slope, intercept))
            elif direction == 'right' and slope > 0:
                fits.append((slope, intercept))

    if len(fits) > 0:
        fit_average = np.average(fits, axis=0)
        lane_line.append(make_point(frame, fit_average))

    return lane_line

def display_lines(frame, lines, line_color=(0, 0, 255), line_width=2):
    line_img = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), line_color, line_width)

    line_img = cv2.addWeighted(frame, 0.8, line_img, 1, 1)
    return line_img