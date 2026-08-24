import sys

import numpy as np
import torch
import pickle

sys.path.insert(0, ".")
from spatiallm.layout.layout import Layout
from prepare_scannet import compute_obb, SCANNET_GT20_CLASSES, EXCLUDED_LABELS

scene = sys.argv[1] if len(sys.argv) > 1 else "scene0000_00"
src = sys.argv[2] if len(sys.argv) > 2 else "data/scannet"
d = torch.load(f"{src}/train/{scene}.pth", map_location="cpu", weights_only=False)
with open(f"{src}/scannet_axis_align_matrix_trainval.pkl", "rb") as f:
    M = np.array(pickle.load(f)[scene])
c = d["coord"] @ M[:3, :3].T
c -= c.min(0)
sem, inst = d["semantic_gt20"], d["instance_gt"]

with open(f"data/scannet_spatiallm/layout/{scene}.txt") as f:
    layout = Layout(f.read())

ratios = []
for iid in np.unique(inst):
    if iid < 1:
        continue
    mask = inst == iid
    if mask.sum() < 30:
        continue
    labels = sem[mask]
    labels = labels[labels >= 0]
    if len(labels) == 0:
        continue
    cid = np.bincount(labels, minlength=20).argmax()
    if cid in EXCLUDED_LABELS:
        continue
    pts = c[mask]
    angle, center, scale = compute_obb(pts)
    centers = np.array(
        [[b.position_x, b.position_y, b.position_z] for b in layout.bboxes]
    )
    best_idx = np.linalg.norm(centers - center, axis=1).argmin()
    best = layout.bboxes[best_idx]
    ca, sa = np.cos(best.angle_z), np.sin(best.angle_z)
    rot = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    local = (pts - centers[best_idx]) @ rot
    half = np.array([best.scale_x, best.scale_y, best.scale_z]) / 2 + 1e-6
    inside = np.all(np.abs(local) <= half, axis=1).mean()
    ratios.append((SCANNET_GT20_CLASSES[cid], best.class_name, inside, mask.sum()))

print(f"{scene}: source-class | box-class | own-points-inside | n_pts")
for r in ratios:
    print(f"  {r[0]:>15} | {r[1]:>15} | {r[2]:.3f} | {r[3]}")
arr = np.array([r[2] for r in ratios])
print(f"mean containment: {arr.mean():.3f}, >99%: {(arr > 0.99).sum()}/{len(arr)}")
print(f"class match: {sum(r[0]==r[1] for r in ratios)}/{len(ratios)}")
