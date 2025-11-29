import math

import cv2
import gym
import gym_donkeycar
import numpy as np

from utilities import get_interest_region, detect_line, average_lines

NUM_GET = 30000

def get_training_data(loop_count, output_dir):
    picture_index = 0
    env = gym.make('donkey-generated-roads-v0')
    env.reset()

    action = np.array([0, 0.1])
    obv, reward, done, info = env.step(action)
    frame = cv2.cvtColor(obv, cv2.COLOR_RGB2BGR)

    for i in range(loop_count):
        height, width, _ = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([15, 40, 40])
        upper_yellow = np.array([45, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        yellow_edge = cv2.Canny(yellow_mask, 200, 400)
        white_edge = cv2.Canny(white_mask, 200, 400)

        yellow_road = get_interest_region(yellow_edge, color='yellow')
        white_road = get_interest_region(white_edge, color='white')

        yellow_lines = detect_line(yellow_road)
        yellow_lane = average_lines(frame, yellow_lines, direction='left')

        white_lines = detect_line(white_road)
        white_lane = average_lines(frame, white_lines, direction='right')

        x_offset = 0
        y_offset = 0
        if len(yellow_lane) > 0 and len(white_lane) > 0:
            left_x1, left_y1, left_x2, left_y2 = yellow_lane[0]
            right_x1, right_y1, right_x2, right_y2 = white_lane[0]

            mid = int(width / 2)
            x_offset = (left_x2 + right_x2) / 2 - mid
            y_offset = int(height / 2)

        elif len(yellow_lane) > 0:
            left_x1, left_y1, left_x2, left_y2 = yellow_lane[0]
            x_offset = left_x2 - left_x1
            y_offset = int(height / 2)

        elif len(white_lane) > 0:
            right_x1, right_y1, right_x2, right_y2 = white_lane[0]
            x_offset = right_x2 - right_x1
            y_offset = int(height / 2)

        else:
            print("No line detected!")
            action = np.array([0, 0.1])
            obv, reward, done, info = env.step(action)
            frame = cv2.cvtColor(obv, cv2.COLOR_RGB2BGR)
            env.reset()
            continue

        angle_to_mid_radian = math.atan(x_offset / y_offset)
        angle_to_mid_deg = int(angle_to_mid_radian * 180 / math.pi)
        steering_angle = angle_to_mid_deg / 45.0
        action = np.array([steering_angle, 0.1])

        img_path = output_dir + "{:d}_{:.4f}.jpg".format(picture_index, steering_angle)
        cv2.imwrite(img_path, frame)
        picture_index += 1

        obv, reward, done, info = env.step(action)
        frame = cv2.cvtColor(obv, cv2.COLOR_RGB2BGR)
    return None

get_training_data(NUM_GET, 'data\\images\\')
print("Program finished")