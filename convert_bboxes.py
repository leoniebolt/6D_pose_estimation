import json
from pathlib import Path
import cv2
from ultralytics import YOLO

def detect_and_save():
    # YOLO-Modell laden (Pfad anpassen falls nötig)
    model_path = "yolo/my_model/my_model.pt"
    detector = YOLO(model_path)

    input_dir = Path("data/rgb")
    output_dir = Path("data/yolo_detections")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Klassen-Labels aus Modell laden (Index -> Label)
    class_labels = detector.names

    for idx in range(10):
        img_file = input_dir / f"{idx}.png"
        if not img_file.is_file():
            print(f"{img_file} nicht gefunden, überspringe Bild.")
            continue

        # Bild laden
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"Bild {img_file} konnte nicht geladen werden.")
            continue

        # Detektion durchführen
        results = detector(str(img_file))

        # Annotiertes Bild speichern
        annotated_image = results[0].plot()
        annotated_path = output_dir / f"{idx}_annotated.png"
        cv2.imwrite(str(annotated_path), annotated_image)

        detections_list = []

        # Bounding Boxes, Klassen und Confidences auslesen
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

        # JSON speichern
        json_path = output_dir / f"{idx}.json"
        with open(json_path, "w") as json_file:
            json.dump(detections_list, json_file, indent=2)

    print("Fertig: Alle Bilder verarbeitet und Ergebnisse gespeichert.")

if __name__ == "__main__":
    detect_and_save()
