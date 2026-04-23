"""
compare_pose.py — Detect persons with YOLO, then compare:
  - YOLO-pose 2D skeleton (on person crops)
  - SAM3D body 3D mesh + keypoints (full image + bboxes)

Output: side-by-side panel per image:
  [Detection] | [YOLO-Pose] | [SAM3D Keypoints] | [SAM3D Mesh]

Usage:
    conda activate sam3
    python compare_pose.py \
        --checkpoint ./checkpoints/sam-3d-body-dinov3/model.ckpt \
        --mhr_path   ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import pyrootutils

pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

from ultralytics import YOLO
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from sam_3d_body.visualization.renderer import Renderer
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

_visualizer = SkeletonVisualizer(line_width=2, radius=5)
_visualizer.set_pose_meta(mhr70_pose_info)


def expand_box(box, margin, h, w):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * margin, (y2 - y1) * margin
    return (
        max(int(cx - bw / 2), 0),
        max(int(cy - bh / 2), 0),
        min(int(cx + bw / 2), w),
        min(int(cy + bh / 2), h),
    )


def draw_yolo_pose(base_img, crop_results, offset_xy):
    out = base_img.copy()
    ox, oy = offset_xy
    for result in crop_results:
        if result.keypoints is None or result.boxes is None:
            continue
        kp_xy = result.keypoints.xy.cpu().numpy()
        kp_conf = (
            result.keypoints.conf.cpu().numpy()
            if result.keypoints.conf is not None
            else None
        )
        for i in range(len(result.boxes)):
            kps = kp_xy[i] + np.array([ox, oy])
            confs = kp_conf[i] if kp_conf is not None else np.ones(17)
            for a, b in COCO_SKELETON:
                if confs[a] > 0.3 and confs[b] > 0.3:
                    pa = tuple(kps[a].astype(int))
                    pb = tuple(kps[b].astype(int))
                    cv2.line(out, pa, pb, (255, 165, 0), 2)
            for j, (x, y) in enumerate(kps.astype(int)):
                if confs[j] > 0.3:
                    cv2.circle(out, (x, y), 4, (0, 0, 255), -1)
    return out


def label_panel(img, text):
    h, w = img.shape[:2]
    header = np.full((40, w, 3), 40, dtype=np.uint8)
    cv2.putText(header, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    return np.vstack([header, img])


def main(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading YOLO detector ...")
    detector = YOLO(args.detector)

    print("Loading YOLO pose model ...")
    pose_model = YOLO(args.pose_model)

    print("Loading SAM3D body model ...")
    model, model_cfg = load_sam_3d_body(args.checkpoint, device=device, mhr_path=args.mhr_path)
    estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=model_cfg)
    faces = model.head_pose.faces.cpu().numpy()

    images = sorted(
        p for p in Path(args.input).iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        print(f"No images found in {args.input}")
        return

    for img_path in images:
        print(f"\n--- {img_path.name} ---")
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print("  Cannot read, skipping")
            continue
        h, w = img_bgr.shape[:2]

        # ── Step 1: detect persons ──────────────────────────────────────────
        det_results = detector(img_bgr, conf=args.conf, classes=[0], verbose=False)
        raw_boxes = [
            box.xyxy[0].cpu().numpy()
            for r in det_results
            if r.boxes is not None
            for box in r.boxes
        ]
        if not raw_boxes:
            print("  No persons detected, skipping")
            continue
        print(f"  {len(raw_boxes)} person(s) detected")

        # Expand boxes with margin
        crop_boxes = [expand_box(b, args.margin, h, w) for b in raw_boxes]

        # ── Panel A: detection boxes ────────────────────────────────────────
        panel_det = img_bgr.copy()
        for x1, y1, x2, y2 in crop_boxes:
            cv2.rectangle(panel_det, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ── Panel B: YOLO pose on crops, translated to full image ───────────
        panel_yolo = img_bgr.copy()
        for x1, y1, x2, y2 in crop_boxes:
            crop = img_bgr[y1:y2, x1:x2]
            pose_results = pose_model(crop, classes=[0], verbose=False, conf=0.1)
            panel_yolo = draw_yolo_pose(panel_yolo, pose_results, (x1, y1))
            cv2.rectangle(panel_yolo, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ── Panels C & D: SAM3D ────────────────────────────────────────────
        sam_bboxes = np.array(crop_boxes, dtype=np.float32)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        panel_sam_kp = img_bgr.copy()
        panel_sam_mesh = img_bgr.copy()

        try:
            outputs = estimator.process_one_image(img_rgb, bboxes=sam_bboxes)
        except Exception as e:
            print(f"  SAM3D failed: {e}")
            outputs = []

        if outputs:
            depths = np.array([o["pred_cam_t"][2] for o in outputs])
            outputs_sorted = [outputs[i] for i in np.argsort(-depths)]

            for o in outputs_sorted:
                kp2d = o["pred_keypoints_2d"]
                kp2d_vis = np.concatenate([kp2d, np.ones((kp2d.shape[0], 1))], axis=-1)
                panel_sam_kp = _visualizer.draw_skeleton(panel_sam_kp, kp2d_vis)

            all_verts, all_faces = [], []
            for pid, o in enumerate(outputs_sorted):
                all_verts.append(o["pred_vertices"] + o["pred_cam_t"])
                all_faces.append(faces + len(o["pred_vertices"]) * pid)
            all_verts = np.concatenate(all_verts, axis=0)
            all_faces = np.concatenate(all_faces, axis=0)
            fake_cam_t = (all_verts.max(0) + all_verts.min(0)) / 2
            all_verts -= fake_cam_t

            renderer = Renderer(focal_length=outputs_sorted[0]["focal_length"], faces=all_faces)
            panel_sam_mesh = (
                renderer(
                    all_verts, fake_cam_t, panel_sam_mesh,
                    mesh_base_color=LIGHT_BLUE, scene_bg_color=(1, 1, 1),
                ) * 255
            ).astype(np.uint8)
        else:
            print("  SAM3D produced no outputs")

        # ── Compose final panel ─────────────────────────────────────────────
        composite = np.hstack([
            label_panel(panel_det,      "Detection"),
            label_panel(panel_yolo,     "YOLO-Pose"),
            label_panel(panel_sam_kp,   "SAM3D Keypoints"),
            label_panel(panel_sam_mesh, "SAM3D Mesh"),
        ])
        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), composite)
        print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare YOLO-pose vs SAM3D on person crops")
    parser.add_argument("--detector",   default="yolo26m.pt")
    parser.add_argument("--pose_model", default="yolo26m-pose.pt")
    parser.add_argument("--checkpoint", default="./checkpoints/sam-3d-body-dinov3/model.ckpt")
    parser.add_argument("--mhr_path",   default="./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")
    parser.add_argument("--input",      default="input_images")
    parser.add_argument("--output",     default="output/compare")
    parser.add_argument("--conf",       type=float, default=0.25)
    parser.add_argument("--margin",     type=float, default=1.2)
    args = parser.parse_args()
    main(args)
