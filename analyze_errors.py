"""Per-scene error analysis for ScanNet val predictions.

Matching protocol identical to eval.py (per-class Hungarian at IoU>=thr, scale
floored to 0.1, identity label mapping). On top of the TP count it classifies
every error and records matched-pair geometry:

Error taxonomy:
  unmatched GT box -> best IoU against any unmatched pred (any class):
      >= 0.25          -> class confusion
      [conf_lo, 0.25)  -> poor localization miss   (conf_lo = 0.1)
      <  conf_lo       -> undetected
  unmatched pred box -> best IoU against any unmatched GT (any class):
      >= 0.25          -> class confusion counterpart
      [conf_lo, 0.25)  -> poor localization FP
      <  conf_lo       -> duplicate if IoU>=0.5 vs a *matched* same-class pred, else spurious

Usage:
  python analyze_errors.py --pred_dir data/scannet_spatiallm/pred_val \
      --tag final     # writes error_analysis/<tag>_*.csv / .json
"""

import os
import os.path as osp
import json
import math
import argparse
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from eval import (
    read_label_mapping,
    assign_class_map,
    assign_minimum_scale,
    get_entity_class,
    get_BBox3D,
)
from bbox.metrics import iou_3d
from spatiallm import Layout

CONF_LO = 0.1
DUP_IOU = 0.5
LARGE = 1e6
CLASSES = [
    name.replace("_", " ")
    for name in [
        "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookcase",
        "painting", "counter", "desk", "curtain", "refrigerator", "shower_curtain",
        "toilet", "sink", "bathtub", "otherfurniture",
    ]
]


def pair_iou(a, b):
    return iou_3d(get_BBox3D(a), get_BBox3D(b))


def ang_diff(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def load_boxes(txt_path, class_map):
    with open(txt_path, "r") as f:
        layout = Layout(f.read())
    layout.bboxes = assign_class_map(layout.bboxes, class_map)
    assign_minimum_scale(layout.bboxes, minimum_scale=0.1)
    return layout.bboxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", default="data/scannet_spatiallm/layout")
    ap.add_argument("--metadata", default="data/scannet_spatiallm/val.csv")
    ap.add_argument("--label_mapping", default="data/scannet_spatiallm/benchmark_categories.tsv")
    ap.add_argument("--tag", default="final")
    args = ap.parse_args()

    class_map = read_label_mapping(args.label_mapping, "scannet18", "scannet18")
    scene_ids = pd.read_csv(args.metadata)["id"].tolist()

    out_dir = "data/scannet_spatiallm/error_analysis"
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    matched_geo = []           # (cls, iou, center_dist, sx_ratio, sy_ratio, sz_ratio, ang_diff_deg)
    confusion = []             # (scene, gt_cls, pred_cls, iou)
    gt_miss = defaultdict(lambda: defaultdict(int))  # cls -> {class_confusion, loc_miss, undetected}
    pred_err = defaultdict(lambda: defaultdict(int))  # cls -> {class_confusion, loc_fp, duplicate, spurious}

    for sid in scene_ids:
        gt = load_boxes(osp.join(args.gt_dir, f"{sid}.txt"), class_map)
        pr = load_boxes(osp.join(args.pred_dir, f"{sid}.txt"), class_map)
        ply_mb = osp.getsize(f"data/scannet_spatiallm/pcd/{sid}.ply") / 1e6

        tp25 = tp50 = 0
        matched_pairs = []      # (gt_idx, pr_idx, cls)
        unmatched_pr = set(range(len(pr)))
        unmatched_gt = set(range(len(gt)))

        for cls in CLASSES:
            g_idx = [i for i in unmatched_gt if get_entity_class(gt[i]) == cls]
            p_idx = [i for i in unmatched_pr if get_entity_class(pr[i]) == cls]
            if not g_idx or not p_idx:
                continue
            iou_m = np.array([[pair_iou(pr[p], gt[g]) for g in g_idx] for p in p_idx])
            cost = np.full(iou_m.shape, LARGE)
            cost[iou_m > 0.25] = -1
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                if iou_m[r, c] >= 0.25:
                    gi, pi = g_idx[c], p_idx[r]
                    matched_pairs.append((gi, pi, cls, float(iou_m[r, c])))
                    unmatched_gt.discard(gi)
                    unmatched_pr.discard(pi)
                    tp25 += 1
                    if iou_m[r, c] >= 0.5:
                        tp50 += 1

        for gi, pi, cls, iou in matched_pairs:
            g, p = gt[gi], pr[pi]
            cd = math.dist(
                (g.position_x, g.position_y, g.position_z),
                (p.position_x, p.position_y, p.position_z),
            )
            matched_geo.append(
                (cls, iou, cd, p.scale_x / g.scale_x, p.scale_y / g.scale_y,
                 p.scale_z / g.scale_z, math.degrees(ang_diff(p.angle_z, g.angle_z)))
            )

        # unmatched GT taxonomy
        for gi in list(unmatched_gt):
            g = gt[gi]
            best_iou, best_pr = 0.0, None
            for pi in unmatched_pr:
                v = pair_iou(pr[pi], g)
                if v > best_iou:
                    best_iou, best_pr = v, pi
            gc = get_entity_class(g)
            if best_iou >= 0.25:
                gt_miss[gc]["class_confusion"] += 1
                confusion.append((sid, gc, get_entity_class(pr[best_pr]), round(best_iou, 3)))
                unmatched_pr.discard(best_pr)  # consumed by the confusion pair
            elif best_iou >= CONF_LO:
                gt_miss[gc]["loc_miss"] += 1
            else:
                gt_miss[gc]["undetected"] += 1

        # unmatched pred taxonomy
        for pi in list(unmatched_pr):
            p = pr[pi]
            best_iou = max((pair_iou(p, gt[gi]) for gi in unmatched_gt), default=0.0)
            pc = get_entity_class(p)
            if best_iou >= 0.25:
                pred_err[pc]["class_confusion"] += 1
            elif best_iou >= CONF_LO:
                pred_err[pc]["loc_fp"] += 1
            else:
                dup = any(
                    get_entity_class(pr[mp[1]]) == pc and pair_iou(p, pr[mp[1]]) >= DUP_IOU
                    for mp in matched_pairs
                )
                pred_err[pc]["duplicate" if dup else "spurious"] += 1

        n_gt, n_pr = len(gt), len(pr)
        f1_25 = 2 * tp25 / (n_gt + n_pr) if (n_gt + n_pr) else 1.0
        rows.append(
            dict(scene=sid, ply_mb=round(ply_mb, 2), n_gt=n_gt, n_pred=n_pr,
                 tp25=tp25, tp50=tp50, f1_25=round(f1_25, 4),
                 n_missed=len(unmatched_gt), n_extra=len(unmatched_pr),
                 n_conf=sum(1 for c in confusion if c[0] == sid))
        )

    df = pd.DataFrame(rows)
    df.to_csv(osp.join(out_dir, f"{args.tag}_per_scene.csv"), index=False)

    geo = pd.DataFrame(
        matched_geo, columns=["cls", "iou", "center_dist", "sx", "sy", "sz", "ang_deg"]
    )
    geo.to_csv(osp.join(out_dir, f"{args.tag}_matched_geo.csv"), index=False)

    pd.DataFrame(confusion, columns=["scene", "gt_cls", "pred_cls", "iou"]).to_csv(
        osp.join(out_dir, f"{args.tag}_confusion.csv"), index=False
    )

    summary = {
        "tag": args.tag,
        "scenes": len(df),
        "micro_f1_25": round(float(df.tp25.sum() * 2 / (df.n_gt.sum() + df.n_pred.sum())), 4),
        "micro_f1_50": round(float(df.tp50.sum() * 2 / (df.n_gt.sum() + df.n_pred.sum())), 4),
        "gt_total": int(df.n_gt.sum()),
        "pred_total": int(df.n_pred.sum()),
        "gt_error_taxonomy": {k: dict(v) for k, v in gt_miss.items()},
        "pred_error_taxonomy": {k: dict(v) for k, v in pred_err.items()},
        "gt_errors_total": int(df.n_missed.sum()),
        "pred_errors_total": int(df.n_extra.sum()),
        "confusion_pairs": int(len(confusion)),
        "worst10_scenes": df.nsmallest(10, "f1_25")[["scene", "f1_25", "n_gt", "n_pred", "ply_mb"]].to_dict("records"),
        "scene_f1_25_quartiles": {
            q: round(float(v), 3) for q, v in df.f1_25.quantile([0, 0.25, 0.5, 0.75, 1]).items()
        },
    }
    with open(osp.join(out_dir, f"{args.tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("\n=== matched-pair geometry by class (median) ===")
    print(geo.groupby("cls").agg(
        n=("iou", "size"), iou_med=("iou", "median"), cdist_med=("center_dist", "median"),
        sx_med=("sx", "median"), sz_med=("sz", "median"),
        ang_med=("ang_deg", "median"), ang_p90=("ang_deg", lambda x: x.quantile(0.9)),
    ).round(3).to_string())


if __name__ == "__main__":
    main()
