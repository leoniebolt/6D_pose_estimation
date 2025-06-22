import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import os

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = any(s.get_info(rs.camera_info.name) == 'RGB Camera' for s in device.sensors)
if not found_rgb:
    print("The demo requires a depth camera with a color sensor.")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Create folder to save images
save_path_color = "captured_images/rgb"
save_path_depth = "captured_images/depth"
os.makedirs(save_path_color, exist_ok=True)
os.makedirs(save_path_depth, exist_ok=True)
image_index = 0

print("Press SPACE to capture an image, or 'q' to quit.")

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # Show the camera stream
        cv.imshow('Depth Image', depth_image)
        cv.imshow('Color Image', color_image)
        # Capture key press
        key = cv.waitKey(1) & 0xFF
        if key == 32:  # Spacebar pressed
            color_filename = os.path.join(save_path_color, f"{image_index}.png")
            depth_filename = os.path.join(save_path_depth, f"{image_index}.png")
            cv.imwrite(color_filename, color_image)
            cv.imwrite(depth_filename, depth_image)
            print(f"Saved: {color_filename}, {depth_filename}")
            image_index += 1  # Increment index for next image
        elif key == ord('q'):  # Quit when 'q' is pressed
            break
finally:
    # Stop streaming
    pipeline.stop()
    cv.destroyAllWindows()
