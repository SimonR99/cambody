import cv2
import numpy as np
from ultralytics import YOLO

# COCO 17-keypoint skeleton connections
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head
    (5, 6),                                   # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),        # arms
    (5, 11), (6, 12), (11, 12),              # torso
    (11, 13), (13, 15), (12, 14), (14, 16), # legs
]

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def draw_poses(image: np.ndarray, results) -> np.ndarray:
    out = image.copy()
    for result in results:
        boxes = result.boxes
        keypoints = result.keypoints

        if boxes is None or keypoints is None:
            continue

        kp_xy = keypoints.xy.cpu().numpy()   # (N, 17, 2)
        kp_conf = keypoints.conf.cpu().numpy() if keypoints.conf is not None else None

        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, f"person {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            kps = kp_xy[i]  # (17, 2)
            confs = kp_conf[i] if kp_conf is not None else np.ones(17)

            # Skeleton edges
            for a, b in SKELETON:
                if confs[a] > 0.3 and confs[b] > 0.3:
                    pt_a = tuple(kps[a].astype(int))
                    pt_b = tuple(kps[b].astype(int))
                    if pt_a != (0, 0) and pt_b != (0, 0):
                        cv2.line(out, pt_a, pt_b, (255, 165, 0), 2)

            # Keypoints
            for j, (x, y) in enumerate(kps.astype(int)):
                if confs[j] > 0.3 and (x, y) != (0, 0):
                    cv2.circle(out, (x, y), 4, (0, 0, 255), -1)

    return out


def main():
    image_path = "image_3.png"
    output_path = "image_3_poses.jpg"

    model = YOLO("yolo26m-pose.pt")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load {image_path}")

    results = model(image, classes=[0], verbose=False, conf=0.02)  # class 0 = person

    n_persons = sum(len(r.boxes) for r in results if r.boxes is not None)
    print(f"Detected {n_persons} person(s) in {image_path}")

    for r_idx, result in enumerate(results):
        if result.keypoints is None:
            continue
        kp_conf = result.keypoints.conf
        for i in range(len(result.boxes)):
            visible = int((kp_conf[i] > 0.3).sum()) if kp_conf is not None else 0
            print(f"  Person {i + 1}: {visible}/17 visible keypoints")

    annotated = draw_poses(image, results)
    cv2.imwrite(output_path, annotated)
    print(f"Saved annotated image to {output_path}")


if __name__ == "__main__":
    main()
