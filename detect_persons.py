import argparse
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def detect_persons(model_path, input_dir, output_dir, conf=0.25):
    model = YOLO(model_path)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        print(f"No images found in {input_dir}")
        return

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not read {img_path}, skipping")
            continue

        results = model(img, conf=conf, classes=[0], verbose=False)  # class 0 = person

        boxes = results[0].boxes
        n = len(boxes)
        print(f"{img_path.name}: {n} person(s) detected")

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            score = float(box.conf[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"person {score:.2f}",
                (x1, max(y1 - 6, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), img)
        print(f"  -> saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect persons with YOLO and draw bounding boxes")
    parser.add_argument("--model", default="yolo26m.pt", help="Path to YOLO model")
    parser.add_argument("--input", default="input_images", help="Folder of input images")
    parser.add_argument("--output", default="output/detections", help="Folder for output images")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    detect_persons(args.model, args.input, args.output, args.conf)
