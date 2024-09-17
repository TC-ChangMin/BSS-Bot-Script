import cv2 as cv
import numpy as np
import pydirectinput
import pygetwindow as gw
from time import time, sleep
from windowcapture import WindowCapture
import keyboard

sleep(2)

def rotate_image(image, angle):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return rotated_image

def combine_boxes(rectangles, distance_threshold=10):
    combined_boxes = []
    used = [False] * len(rectangles)

    for i, (x1, y1, w1, h1) in enumerate(rectangles):
        if used[i]:
            continue
        x2, y2, w2, h2 = x1, y1, w1, h1
        for j, (x3, y3, w3, h3) in enumerate(rectangles):
            if i != j and not used[j]:
                if abs(x3 - x2) < distance_threshold and abs(y3 - y2) < distance_threshold:
                    x2 = min(x2, x3)
                    y2 = min(y2, y3)
                    x2_w = max(x2 + w2, x3 + w3)
                    y2_h = max(y2 + h2, y3 + h3)
                    w2 = x2_w - x2
                    h2 = y2_h - y2
                    used[j] = True
        combined_boxes.append([x2, y2, w2, h2])
    return combined_boxes

def is_gray_pixel(pixel, gray_threshold=30):
    return abs(pixel[0] - pixel[1]) < gray_threshold and abs(pixel[1] - pixel[2]) < gray_threshold

def has_majority_gray_pixels(frame, bbox, gray_threshold=30, gray_ratio_threshold=0.7):
    x, y, w, h = bbox
    roi = frame[y:y+h, x:x+w]
    gray_pixels = np.sum([is_gray_pixel(pixel) for pixel in roi.reshape(-1, 3)])
    total_pixels = roi.size // 3
    gray_ratio = gray_pixels / total_pixels
    return gray_ratio > gray_ratio_threshold

def detect_objects(frame, reference_image_paths, window_size, last_detection_time):
    output_frame = frame.copy()
    rectangles = []

    # Convert frame to grayscale if it's in color
    if len(frame.shape) == 3:
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    else:
        frame_gray = frame

    for reference_image_path in reference_image_paths:
        needle_img = cv.imread(reference_image_path, cv.IMREAD_GRAYSCALE)
        if needle_img is None:
            print(f"Error: '{reference_image_path}' could not be loaded.")
            continue

        result = cv.matchTemplate(frame_gray, needle_img, cv.TM_CCOEFF_NORMED)
        threshold = 0.3
        locations = np.where(result >= threshold)
        locations = list(zip(*locations[::-1]))

        for loc in locations:
            top_left = loc
            bottom_right = (top_left[0] + needle_img.shape[1], top_left[1] + needle_img.shape[0])
            rectangles.append([top_left[0], top_left[1], needle_img.shape[1], needle_img.shape[0]])

    rectangles, _ = cv.groupRectangles(rectangles, groupThreshold=1, eps=0.5)
    combined_boxes = combine_boxes(rectangles)

    current_time = time()
    if current_time - last_detection_time >= 3:  # Only process boxes every 1 second
        for (x, y, w, h) in combined_boxes[:1]:  # Ensure only 1 box is processed
            if not has_majority_gray_pixels(frame, (x, y, w, h)):
                cv.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color for combined boxes
                cv.drawMarker(output_frame, (x + w // 2, y + h // 2), (0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=20, thickness=2)

                # Calculate the center of the box and the center of the screen
                box_center = (x + w // 2, y + h // 2)
                screen_center = (450, 350)  # Center of the Roblox window

                # Calculate distance
                x_dist = box_center[0] - screen_center[0]
                y_dist = screen_center[1] - box_center[1]

                # Print distance for debugging
                print(f"Distance: ({x_dist}, {y_dist})")

                # Move character with WASD keys
                move_character(x_dist, y_dist)

        # Update last detection time
        last_detection_time = current_time

    return output_frame, last_detection_time

def move_character(x_dist, y_dist):
    # Determine movement direction based on distance
    if x_dist > 0:
        pydirectinput.keyDown('d')
        sleep(abs(x_dist) / 1000)  # Adjust movement speed based on distance
        pydirectinput.keyUp('d')
        sleep(0.25)

        pydirectinput.keyDown('a')
        sleep(abs(x_dist) / 1000)  # Adjust movement speed based on distance
        pydirectinput.keyUp('a')
        sleep(0.25)

    elif x_dist < 0:
        pydirectinput.keyDown('a')
        sleep(abs(x_dist) / 1000)  # Adjust movement speed based on distance
        pydirectinput.keyUp('a')
        sleep(0.25)

        pydirectinput.keyDown('d')
        sleep(abs(x_dist) / 1000)  # Adjust movement speed based on distance
        pydirectinput.keyUp('d')
        sleep(0.25)


    if y_dist > 0:
        pydirectinput.keyDown('s')
        sleep(abs(y_dist) / 1000)  # Adjust movement speed based on distance
        pydirectinput.keyUp('s')
        sleep(0.25)

        pydirectinput.keyDown('w')
        sleep(abs(x_dist) / 1000)  # Adjust movement speed based on distance
        pydirectinput.keyUp('w')
        sleep(0.25)


    elif y_dist < 0:
        pydirectinput.keyDown('w')
        sleep(abs(y_dist) / 1000)  # Adjust movement speed based on distance
        pydirectinput.keyUp('w')
        sleep(0.25)

        pydirectinput.keyDown('s')
        sleep(abs(x_dist) / 1000)  # Adjust movement speed based on distance
        pydirectinput.keyUp('s')
        sleep(0.25)


# Load the reference images for object detection
reference_image_paths = ['images/bomb1.png', 'images/bomb2.png', 'images/bomb3.png']

# Initialize the WindowCapture class
wincap = WindowCapture('Roblox')

# Timing control
last_detection_time = 0

# Main loop to capture and process video frames
sleep(10)  # Pause for 10 seconds before running the loop

loop_time = time()
while True:
    screenshot = wincap.get_screenshot()
    processed_frame, last_detection_time = detect_objects(screenshot, reference_image_paths, (900, 700), last_detection_time)  # Roblox window size
    cv.imshow('Matches', processed_frame)

    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # Exit the loop if 'q' is pressed
    if keyboard.is_pressed('q'):
        break
    
cv.destroyAllWindows()
print('Done.')
