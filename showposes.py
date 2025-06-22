import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

# ======= Utility Functions =======

def read_camera_matrix(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return np.array(data["K"])

def load_object_poses(path):
    with open(path, 'r') as file:
        return json.load(file)

def create_bounding_box(half_extents):
    x, y, z = half_extents
    return np.array([
        [-x, -y, -z],
        [ x, -y, -z],
        [ x,  y, -z],
        [-x,  y, -z],
        [-x, -y,  z],
        [ x, -y,  z],
        [ x,  y,  z],
        [-x,  y,  z],
    ])

def project_to_image_plane(points_3d, intrinsic_matrix):
    projected = intrinsic_matrix @ points_3d.T
    projected /= projected[2, :]
    return projected[:2, :].T.astype(int)

def render_box(image, projected_corners, color=(255, 255, 0), line_thickness=2):
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]
    for start, end in edges:
        pt1, pt2 = tuple(projected_corners[start]), tuple(projected_corners[end])
        cv2.line(image, pt1, pt2, color, line_thickness)

def render_coordinate_axes(image, transform, intrinsics, axis_length=0.05):
    origin = transform[:3, 3]
    axes = np.eye(3) * axis_length
    axis_endpoints = np.stack([
        origin,
        origin + transform[:3, :3] @ axes[:, 0],
        origin + transform[:3, :3] @ axes[:, 1],
        origin + transform[:3, :3] @ axes[:, 2]
    ])
    projected = project_to_image_plane(axis_endpoints, intrinsics)
    cv2.line(image, tuple(projected[0]), tuple(projected[1]), (0, 0, 255), 2)  # X-axis
    cv2.line(image, tuple(projected[0]), tuple(projected[2]), (0, 255, 0), 2)  # Y-axis
    cv2.line(image, tuple(projected[0]), tuple(projected[3]), (255, 0, 0), 2)  # Z-axis

# ======= Main Execution =======

def visualize():
    # --- File paths ---
    rgb_image_file = "megapose6d/local_data/examples/morobot/image_rgb.png"
    pose_data_file = "megapose6d/local_data/examples/morobot/outputs/object_data.json"
    camera_file = "megapose6d/local_data/examples/morobot/camera_data.json"
    output_file = "megapose6d/local_data/examples/morobot/visualizations/poses.png"

    # --- Half sizes of object bounding boxes in meters ---
    object_sizes = {
        "1A_gray":   (0.01045, 0.04, 0.0505),
        "1A_yellow": (0.01045, 0.04, 0.0505),
        "1B_yellow": (0.008,   0.04, 0.0505),
        "3B_grey":   (0.04,    0.008, 0.06),
    }

    # --- Load data ---
    image = cv2.imread(rgb_image_file)
    intrinsics = read_camera_matrix(camera_file)
    pose_entries = load_object_poses(pose_data_file)

    for entry in pose_entries:
        label = entry["label"]
        if label not in object_sizes:
            print(f"[WARNING] Missing dimensions for object '{label}'. Skipping...")
            continue

        half_extents = object_sizes[label]

        # Compute transformation matrix
        quaternion = entry["TWO"][0]
        translation = entry["TWO"][1]
        rotation_matrix = R.from_quat(quaternion).as_matrix()
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = translation

        # Generate and transform bounding box
        corners_3d = create_bounding_box(half_extents)
        corners_homogeneous = np.hstack([corners_3d, np.ones((8, 1))])
        transformed_corners = (transformation_matrix @ corners_homogeneous.T).T[:, :3]

        # Project to 2D and draw
        corners_2d = project_to_image_plane(transformed_corners, intrinsics)
        render_box(image, corners_2d)
        render_coordinate_axes(image, transformation_matrix, intrinsics)

        # Draw label
        center = np.mean(corners_2d, axis=0).astype(int)
        cv2.putText(image, label, tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Save final visualization
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    cv2.imwrite(output_file, image)
    print(f"Pose visualization saved at: {output_file}")

if __name__ == "__main__":
    visualize()
