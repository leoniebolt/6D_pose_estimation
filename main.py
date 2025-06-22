import os
import json
import shutil
import subprocess
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R

# === YOLO DETECTION ===
def detect_and_save():
    model_path = "yolo/my_model/my_model.pt"
    detector = YOLO(model_path)

    input_dir = Path("data/rgb")
    output_dir = Path("data/yolo_detections")
    output_dir.mkdir(parents=True, exist_ok=True)

    class_labels = detector.names

    for idx in range(10):
        img_file = input_dir / f"{idx}.png"
        if not img_file.is_file():
            continue

        image = cv2.imread(str(img_file))
        if image is None:
            continue

        results = detector(str(img_file))

        annotated_image = results[0].plot()
        annotated_path = output_dir / f"{idx}_detected.png"
        cv2.imwrite(str(annotated_path), annotated_image)

        detections_list = []

        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        for bbox, cls_id, conf_score in zip(boxes, classes, confidences):
            if conf_score < 0.4:
                continue

            xmin, ymin, xmax, ymax = bbox.astype(int).tolist()
            label = class_labels.get(int(cls_id), "unknown")

            detections_list.append({
                "label": label,
                "bbox_modal": [xmin, ymin, xmax, ymax]
            })

        json_path = output_dir / f"{idx}.json"
        with open(json_path, "w") as json_file:
            json.dump(detections_list, json_file, indent=2)

    print("[YOLO] Detektionen abgeschlossen.")

# === UTILITY FUNCS ===
def load_intrinsics(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return np.array(data["K"])

def load_poses(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def get_3d_box_from_half_sizes(half_size):
    x, y, z = half_size
    return np.array([[-x,-y,-z],[x,-y,-z],[x,y,-z],[-x,y,-z],[-x,-y,z],[x,-y,z],[x,y,z],[-x,y,z]])

def project_points(points_3d, camera_matrix):
    points_2d = camera_matrix @ points_3d.T
    points_2d /= points_2d[2, :]
    return points_2d[:2, :].T.astype(int)

def draw_box(image, corners_2d, color=(255, 255, 0), thickness=2):
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i,j in edges:
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[j]), color, thickness)

def draw_axes(image, transform, K, length=0.05):
    origin = transform[:3, 3]
    axes = np.eye(3) * length
    points = np.stack([origin, origin + transform[:3,:3] @ axes[:,0],
                                origin + transform[:3,:3] @ axes[:,1],
                                origin + transform[:3,:3] @ axes[:,2]])
    projected = project_points(points, K)
    cv2.line(image, tuple(projected[0]), tuple(projected[1]), (0,0,255), 2)
    cv2.line(image, tuple(projected[0]), tuple(projected[2]), (0,255,0), 2)
    cv2.line(image, tuple(projected[0]), tuple(projected[3]), (255,0,0), 2)

# === VISUALIZATION ===
def visualize():
    image_path = "megapose6d/local_data/examples/morobot/image_rgb.png"
    pose_json_path = "megapose6d/local_data/examples/morobot/outputs/object_data.json"
    intrinsics_path = "megapose6d/local_data/examples/morobot/camera_intrinsic.json"
    output_path = "megapose6d/local_data/examples/morobot/visualizations/poses.png"

    box_dims = {
        "1A_gray": (0.01045, 0.04, 0.0505),
        "1A_yellow": (0.01045, 0.04, 0.0505),
        "1B_yellow": (0.008, 0.04, 0.0505),
        "3B_grey": (0.04, 0.008, 0.06),
    }

    image = cv2.imread(image_path)
    K = load_intrinsics(intrinsics_path)
    objects = load_poses(pose_json_path)

    for obj in objects:
        label = obj["label"]
        if label not in box_dims:
            continue

        half_size = box_dims[label]
        quat, trans = obj["TWO"]
        rot = R.from_quat(quat).as_matrix()
        transform = np.eye(4)
        transform[:3,:3] = rot
        transform[:3,3] = trans

        box = get_3d_box_from_half_sizes(half_size)
        box_h = np.hstack([box, np.ones((8,1))])
        box_world = (transform @ box_h.T).T[:, :3]

        box_2d = project_points(box_world, K)
        draw_box(image, box_2d)
        draw_axes(image, transform, K)

        center_2d = np.mean(box_2d, axis=0).astype(int)
        cv2.putText(image, label, tuple(center_2d), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)

# === PIPELINE ===
def run_command(command):
    subprocess.run(command, check=True)

def copy_file(src, dst):
    shutil.copy2(src, dst)

def copy_and_replace_folder(src_folder, dst_folder):
    if os.path.exists(dst_folder): shutil.rmtree(dst_folder)
    shutil.copytree(src_folder, dst_folder)

def ensure_dir_exists(path):
    os.makedirs(path, exist_ok=True)

def validate_meshes():
    import trimesh
    for file in os.listdir("data/morobot/meshes"):
        if file.endswith(".ply"):
            trimesh.load(os.path.join("data/morobot/meshes", file))

def process_images():
    for i in range(10):
        rgb = f"data/rgb/{i}.png"
        depth = f"data/depth/{i}.png"
        det_json = f"data/yolo_detections/{i}.json"
        dst = "megapose6d/local_data/examples/morobot"

        copy_file(rgb, f"{dst}/image_rgb.png")
        copy_file(depth, f"{dst}/image_depth.png")
        copy_file(det_json, f"{dst}/inputs/object_data.json")
        copy_file("data/camera_intrinsic.json", f"{dst}/camera_intrinsic.json")
        copy_and_replace_folder("data/morobot/meshes", f"{dst}/meshes")

        run_command(["python", "-m", "megapose.scripts.run_inference_on_example", "morobot", "--vis-detections"])
        run_command(["python", "-m", "megapose.scripts.run_inference_on_example", "morobot", "--run-inference"])
        run_command(["python", "-m", "megapose.scripts.run_inference_on_example", "morobot", "--vis-outputs"])
        visualize()

        ensure_dir_exists(f"{dst}/visualizations/pose")
        ensure_dir_exists(f"{dst}/visualizations/detections")

        shutil.copy2(f"{dst}/visualizations/poses.png", f"{dst}/visualizations/pose/{i}_poses.png")
        shutil.copy2(f"{dst}/visualizations/all_results.png", f"{dst}/visualizations/pose/{i}_all_results.png")
        shutil.copy2(f"{dst}/visualizations/detections.png", f"{dst}/visualizations/detections/{i}_detections.png")
        shutil.copy2(f"{dst}/outputs/object_data.json", f"{dst}/visualizations/pose/{i}_object_data.json")
        yolo_img = f"data/yolo_detections/{i}_detected.png"
        if os.path.exists(yolo_img):
            shutil.copy2(yolo_img, f"{dst}/visualizations/{i}_yolo_detected.png")

if __name__ == "__main__":
    print("[PIPELINE] Starte YOLO → MegaPose Pipeline")
    detect_and_save()
    validate_meshes()
    process_images()
    print("[DONE] Verarbeitung abgeschlossen.")
