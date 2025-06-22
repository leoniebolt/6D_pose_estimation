import os
import json
import shutil
import subprocess
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R

# === YOLO OBJECT DETECTION ===
def detect_objects():
    model_path = "yolo/my_model/my_model.pt"
    model = YOLO(model_path)

    input_dir = Path("data/rgb")
    output_dir = Path("data/yolo_detections")
    output_dir.mkdir(parents=True, exist_ok=True)

    class_labels = model.names

    for idx in range(10):
        image_path = input_dir / f"{idx}.png"
        if not image_path.exists():
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        results = model(str(image_path))
        annotated = results[0].plot()
        cv2.imwrite(str(output_dir / f"{idx}_detected.png"), annotated)

        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()

        detections = []
        for box, cls, score in zip(boxes, classes, scores):
            if score < 0.5:
                continue
            x1, y1, x2, y2 = map(int, box)
            label = class_labels.get(int(cls), "unknown")
            detections.append({"label": label, "bbox_modal": [x1, y1, x2, y2]})

        with open(output_dir / f"{idx}.json", "w") as f:
            json.dump(detections, f, indent=2)

    print("[YOLO] Object detection completed.")

# === UTILITY FUNCTIONS ===
def load_camera_intrinsics(filepath):
    with open(filepath, "r") as f:
        return np.array(json.load(f)["K"])

def load_pose_predictions(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def get_3d_bbox_vertices(half_dims):
    x, y, z = half_dims
    return np.array([[-x,-y,-z],[x,-y,-z],[x,y,-z],[-x,y,-z],
                     [-x,-y,z],[x,-y,z],[x,y,z],[-x,y,z]])

def project_3d_to_2d(points_3d, K):
    points_2d = K @ points_3d.T
    points_2d /= points_2d[2]
    return points_2d[:2].T.astype(int)

def draw_bounding_box(image, points_2d, color=(255, 255, 0), thickness=2):
    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        cv2.line(image, tuple(points_2d[a]), tuple(points_2d[b]), color, thickness)

def draw_axes(image, transform, K, length=0.05):
    origin = transform[:3, 3]
    axes = np.eye(3) * length
    points = np.stack([origin] + [origin + transform[:3,:3] @ a for a in axes])
    points_2d = project_3d_to_2d(points, K)

    cv2.line(image, tuple(points_2d[0]), tuple(points_2d[1]), (0,0,255), 2)  # X - red
    cv2.line(image, tuple(points_2d[0]), tuple(points_2d[2]), (0,255,0), 2)  # Y - green
    cv2.line(image, tuple(points_2d[0]), tuple(points_2d[3]), (255,0,0), 2)  # Z - blue

# === VISUALIZATION ===
def visualize_pose_estimates():
    rgb_path = "megapose6d/local_data/examples/morobot/image_rgb.png"
    pose_path = "megapose6d/local_data/examples/morobot/outputs/object_data.json"
    intrinsics_path = "megapose6d/local_data/examples/morobot/camera_intrinsic.json"
    out_path = "megapose6d/local_data/examples/morobot/visualizations/poses.png"

    object_dimensions = {
        "1A_gray": (0.01045, 0.04, 0.0505),
        "1A_yellow": (0.01045, 0.04, 0.0505),
        "1B_yellow": (0.008, 0.04, 0.0505),
        "3B_grey": (0.04, 0.008, 0.06),
    }

    image = cv2.imread(rgb_path)
    K = load_camera_intrinsics(intrinsics_path)
    objects = load_pose_predictions(pose_path)

    for obj in objects:
        name = obj["label"]
        if name not in object_dimensions:
            continue

        size = object_dimensions[name]
        quat, trans = obj["TWO"]
        rotation = R.from_quat(quat).as_matrix()

        tf = np.eye(4)
        tf[:3,:3] = rotation
        tf[:3,3] = trans

        vertices = get_3d_bbox_vertices(size)
        vertices_hom = np.hstack([vertices, np.ones((8,1))])
        transformed = (tf @ vertices_hom.T).T[:, :3]
        projected = project_3d_to_2d(transformed, K)

        draw_bounding_box(image, projected)
        draw_axes(image, tf, K)

        label_pos = tuple(np.mean(projected, axis=0).astype(int))
        cv2.putText(image, name, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, image)

# === FILE HANDLING ===
def run_command(cmd):
    subprocess.run(cmd, check=True)

def copy_file(src, dst):
    shutil.copy2(src, dst)

def replace_directory(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def make_dir(path):
    os.makedirs(path, exist_ok=True)

def verify_meshes():
    import trimesh
    mesh_dir = "data/morobot/meshes"
    for file in os.listdir(mesh_dir):
        if file.endswith(".ply"):
            trimesh.load(os.path.join(mesh_dir, file))

# === MAIN PIPELINE ===
def process_all_images():
    for idx in range(10):
        base = "megapose6d/local_data/examples/morobot"

        # Copy input files into example folder
        copy_file(f"data/rgb/{idx}.png", f"{base}/image_rgb.png")
        copy_file(f"data/depth/{idx}.png", f"{base}/image_depth.png")
        copy_file(f"data/yolo_detections/{idx}.json", f"{base}/inputs/object_data.json")
        copy_file("data/camera_intrinsic.json", f"{base}/camera_intrinsic.json")
        replace_directory("data/morobot/meshes", f"{base}/meshes")

        # Run inference and visualization commands
        run_command(["python", "-m", "megapose.scripts.run_inference_on_example", "morobot", "--vis-detections"])
        run_command(["python", "-m", "megapose.scripts.run_inference_on_example", "morobot", "--run-inference"])
        run_command(["python", "-m", "megapose.scripts.run_inference_on_example", "morobot", "--vis-outputs"])
        visualize_pose_estimates()

        # Save poses visualization with index
        make_dir(f"{base}/visualizations/pose")
        shutil.copy2(f"{base}/visualizations/poses.png", f"{base}/visualizations/pose/{idx}_poses.png")

        # Save YOLO detection image with index if exists
        yolo_img = f"data/yolo_detections/{idx}_detected.png"
        if os.path.exists(yolo_img):
            shutil.copy2(yolo_img, f"{base}/visualizations/{idx}_yolo_detected.png")

        # === Save all_results.png and object_data.json BEFORE cleanup ===
        all_results_src = f"{base}/visualizations/all_results.png"
        all_results_dst_dir = f"{base}/visualizations/all_results"
        make_dir(all_results_dst_dir)
        if os.path.exists(all_results_src):
            shutil.copy2(all_results_src, f"{all_results_dst_dir}/{idx}_all_results.png")

        object_data_src = f"{base}/outputs/object_data.json"
        object_data_dst_dir = f"{base}/visualizations/pose/object_data"
        make_dir(object_data_dst_dir)
        if os.path.exists(object_data_src):
            shutil.copy2(object_data_src, f"{object_data_dst_dir}/{idx}_object_data.json")

    # Clean up temporary files AFTER saving results
    cleanup_files = [
        "inputs/object_data.json",
        "outputs/object_data.json",
        "visualizations/all_results.png",
        "visualizations/contour_overlay.png",
        "visualizations/detections.png",
        "visualizations/mesh_overlay.png",
        "visualizations/poses.png",
        "image_rgb.png",
        "image_depth.png"
    ]
    for f in cleanup_files:
        full_path = f"megapose6d/local_data/examples/morobot/{f}"
        if os.path.exists(full_path):
            os.remove(full_path)
            print(f"[CLEANUP] Removed: {f}")
        else:
            print(f"[CLEANUP] Skipped (not found): {f}")

# === MAIN ENTRY POINT ===
if __name__ == "__main__":
    print("[START] Pipeline execution started.")
    detect_objects()
    verify_meshes()
    process_all_images()
    print("[DONE] All tasks completed successfully.")
