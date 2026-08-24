import os
import csv
import argparse
import pickle
from glob import glob

import numpy as np
import torch
from tqdm import tqdm

from spatiallm.layout.layout import Layout
from spatiallm.layout.entity import Bbox

# ScanNet gt20 label mapping (benchmark 20 classes).
# wall/floor/unlabeled are excluded from the object-detection GT;
# picture/bookshelf are renamed to SpatialLM vocabulary for better transfer.
SCANNET_GT20_CLASSES = {
    0: "wall",
    1: "floor",
    2: "cabinet",
    3: "bed",
    4: "chair",
    5: "sofa",
    6: "table",
    7: "door",
    8: "window",
    9: "bookcase",  # bookshelf
    10: "painting",  # picture
    11: "counter",
    12: "desk",
    13: "curtain",
    14: "refrigerator",
    15: "shower_curtain",
    16: "toilet",
    17: "sink",
    18: "bathtub",
    19: "otherfurniture",
}
EXCLUDED_LABELS = {-1, 0, 1}  # unlabeled, wall, floor

MIN_INSTANCE_POINTS = 30  # skip tiny/noisy instances


def compute_obb(points: np.ndarray):
    """Compute a yaw-only oriented bounding box of an (N, 3) point set.

    Returns (angle_z, center, scale) where the box frame is Rz(angle_z).
    """
    xy = points[:, :2]
    mu = xy.mean(axis=0)
    _, _, v = np.linalg.svd(xy - mu, full_matrices=False)
    v0 = v[0]
    if v0[0] < 0:
        v0 = -v0
    angle = np.arctan2(v0[1], v0[0])
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])

    local = (xy - mu) @ rot  # world -> box frame
    extent = np.ptp(local, axis=0)
    center_local = local.min(axis=0) + extent / 2
    center_xy = mu + center_local @ rot.T  # box frame -> world

    zmin, zmax = points[:, 2].min(), points[:, 2].max()
    center = np.array([center_xy[0], center_xy[1], (zmin + zmax) / 2])
    scale = np.array([extent[0], extent[1], zmax - zmin])
    return angle, center, scale


def write_ply(path: str, points: np.ndarray, colors: np.ndarray):
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    vertex = np.zeros(
        n, dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)]
    )
    vertex["xyz"] = points.astype(np.float32)
    vertex["rgb"] = np.clip(colors, 0, 255).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex.tobytes())


def convert_scene(pth_path: str, axis_align: np.ndarray, pcd_path: str, layout_path: str):
    data = torch.load(pth_path, map_location="cpu", weights_only=False)
    coords = data["coord"].astype(np.float64)
    colors = data["color"]
    semantic_gt20 = data["semantic_gt20"]
    instance_gt = data["instance_gt"]

    # apply the yaw part of the axis-align matrix (the .pth coords are only
    # gravity-aligned); a global translation is irrelevant since we re-anchor
    # the scene to the origin below and generate the GT in the same frame
    coords = coords @ axis_align[:3, :3].T
    coords -= coords.min(axis=0)

    write_ply(pcd_path, coords, colors)

    # one oriented bbox per object instance (majority-voted gt20 class)
    layout = Layout()
    for inst_id in np.unique(instance_gt):
        if inst_id < 1:
            continue  # 0 / -1 are not instances
        mask = instance_gt == inst_id
        if mask.sum() < MIN_INSTANCE_POINTS:
            continue
        labels = semantic_gt20[mask]
        labels = labels[labels >= 0]
        if len(labels) == 0:
            continue
        class_id = np.bincount(labels, minlength=20).argmax()
        if class_id in EXCLUDED_LABELS:
            continue

        angle, center, scale = compute_obb(coords[mask])
        layout.bboxes.append(
            Bbox(
                id=len(layout.bboxes),
                class_name=SCANNET_GT20_CLASSES[class_id],
                position_x=center[0],
                position_y=center[1],
                position_z=center[2],
                angle_z=angle,
                scale_x=scale[0],
                scale_y=scale[1],
                scale_z=scale[2],
            )
        )

    # canonical order + drop sub-0.15m boxes and empty class names
    layout.reorder_entities()
    with open(layout_path, "w") as f:
        f.write(layout.to_language_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ScanNet (Pointcept .pth) -> SpatialLM dataset")
    parser.add_argument("-s", "--src", type=str, default="data/scannet")
    parser.add_argument("-d", "--dst", type=str, default="data/scannet_spatiallm")
    parser.add_argument(
        "--scene", type=str, default=None, help="convert a single scene id for testing"
    )
    args = parser.parse_args()

    with open(os.path.join(args.src, "scannet_axis_align_matrix_trainval.pkl"), "rb") as f:
        matrices = pickle.load(f)

    os.makedirs(os.path.join(args.dst, "pcd"), exist_ok=True)
    os.makedirs(os.path.join(args.dst, "layout"), exist_ok=True)

    split_rows = []
    for split in ["train", "val"]:
        pth_files = sorted(glob(os.path.join(args.src, split, "*.pth")))
        if args.scene:
            pth_files = [p for p in pth_files if os.path.basename(p)[: -len(".pth")] == args.scene]
            split_rows = [(args.scene, "train")]
            split = "train"

        for pth_path in tqdm(pth_files, desc=split):
            scene_id = os.path.basename(pth_path)[: -len(".pth")]
            if scene_id not in matrices:
                print(f"Skipping {scene_id}: no axis-align matrix")
                continue
            convert_scene(
                pth_path,
                np.array(matrices[scene_id]),
                os.path.join(args.dst, "pcd", f"{scene_id}.ply"),
                os.path.join(args.dst, "layout", f"{scene_id}.txt"),
            )
            split_rows.append((scene_id, split))

    with open(os.path.join(args.dst, "split.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "split"])
        writer.writerows(split_rows)
    print(f"Done. {len(split_rows)} scenes -> {args.dst}")
