# Metro-Security-Violation-System


An automated, edge-optimized computer vision system designed to detect and deter wrong-way movement in secured areas (like metro stations). It uses YOLOv8 for real-time object tracking and triggers automated audio alerts and high-resolution photo captures when a violation is detected.

## Overview

This system tracks individuals moving across a camera's field of view. By dividing the frame into four distinct vertical zones, it monitors the sequential path of each tracked person. If someone moves against the designated flow of traffic (e.g., passing from Zone 1 to Zone 2 to Zone 3), the system flags them as a violator. 

To ensure high performance on edge hardware without sacrificing evidence quality, the system runs YOLOv8 tracking on a heavily downscaled video feed while saving the violation evidence in the camera's native high resolution.

## Key Features

* **Directional Zone Tracking:** Divides the frame into four vertical zones (20%, 40%, 60%, 80%) to accurately track movement sequences.
* **Dual-Resolution Processing:** Tracks objects at 320x240 for maximum FPS, but scales bounding boxes back to the original resolution to capture crystal-clear evidence photos.
* **Automated Audio Deterrent:** Uses `pygame` to generate an immediate, loud 1200Hz synthesized beep to alert security and deter the violator.
* **High-Res Evidence Capture:** Automatically saves a timestamped, high-quality JPEG of the violator with a red bounding box and center point drawn over them.
* **Real-Time Dashboard:** Displays a live video feed with overlaid tracking IDs, current movement direction arrows, FPS metrics, and violation counts.

## How It Works

1.  **Initialization:** The script loads the YOLOv8 model (using the NCNN format for optimization), initializes the audio mixer, and reads the video stream.
2.  **Downscaling:** Each frame is resized to a lower resolution to speed up inference time.
3.  **Tracking:** The `ultralytics` ByteTrack implementation tracks individuals across frames.
4.  **Zone Analysis:** The center point of each bounding box is calculated and assigned to one of four zones. The system keeps a rolling history of the last 10 zones visited by each ID.
5.  **Violation Check:** If an ID's sequence shows strict left-to-right movement across at least three zones (e.g., `1 -> 2 -> 3`), a violation is triggered.
6.  **Action:** The system plays the alert tone, maps the low-res bounding box back to the high-res frame, and saves the image to the `wrong_direction_violators` directory.

## Prerequisites

Ensure you have Python installed along with the following libraries:

* `opencv-python` (cv2)
* `numpy`
* `ultralytics`
* `pygame`

## Outputs

All captured violations are automatically saved in the auto-generated `wrong_direction_violators/` directory. Files are named using the track ID and a precise timestamp for easy sorting and auditing:
`violation_ID[X]_[YYYYMMDD]_[HHMMSS]_[ms].jpg`
