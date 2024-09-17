import cv2 as cv
import numpy as np
from time import time
from windowcapture import WindowCapture

# Load the reference images for object detection
reference_images = ['images/bomb2.png']

# Initialize the WindowCapture class
wincap = WindowCapture('Roblox')

# Initialize a dictionary to track detection status
detection_status = {}

def detect_objects(frame, reference_images):
    output_frame = frame.copy()
    rectangles = []

    for reference_path in reference_images:
        needle_img = cv.imread(reference_path, cv.IMREAD_UNCHANGED)
        if needle_img is None:
            print(f"Error: '{reference_path}' could not be loaded.")
            detection_status[reference_path] = "Load Error"
            continue

        # Ensure the reference image and frame are both in the same format (8-bit 3-channel)
        if needle_img.ndim == 2:  # If grayscale
            needle_img = cv.cvtColor(needle_img, cv.COLOR_GRAY2BGR)
        elif needle_img.shape[2] == 4:  # If with alpha channel
            needle_img = cv.cvtColor(needle_img, cv.COLOR_BGRA2BGR)

        if frame.ndim == 2:  # If grayscale
            frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:  # If with alpha channel
            frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

        # Convert both images to the same data type if necessary
        needle_img = np.asarray(needle_img, dtype=np.uint8)
        frame = np.asarray(frame, dtype=np.uint8)

        result = cv.matchTemplate(frame, needle_img, cv.TM_CCOEFF_NORMED)
        threshold = 0.33
        locations = np.where(result >= threshold)
        locations = list(zip(*locations[::-1]))

        if locations:
            detection_status[reference_path] = "Detected"
            for loc in locations:
                top_left = loc
                bottom_right = (top_left[0] + needle_img.shape[1], top_left[1] + needle_img.shape[0])
                rectangles.append([top_left[0], top_left[1], needle_img.shape[1], needle_img.shape[0]])
        else:
            detection_status[reference_path] = "Not Detected"

    rectangles, _ = cv.groupRectangles(rectangles, groupThreshold=1, eps=0.5)

    for (x, y, w, h) in rectangles:
        cv.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        center_x = x + w // 2
        center_y = y + h // 2
        cv.drawMarker(output_frame, (center_x, center_y), (0, 0, 255), markerType=cv.MARKER_CROSS, markerSize=20, thickness=2)

    return output_frame

# Main loop to capture and process video frames
loop_time = time()
while True:
    screenshot = wincap.get_screenshot()
    processed_frame = detect_objects(screenshot, reference_images)
    cv.imshow('Matches', processed_frame)

    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()
print('Done.')