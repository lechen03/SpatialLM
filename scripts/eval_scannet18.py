"""Supplementary 18-class ScanNet F1 evaluation.

Identical protocol to eval.py (functions imported from it, unchanged), but
scores ALL 18 ScanNet GT classes instead of the hardcoded SpatialLM vocabulary
OBJECTS list (which silently drops 11 of our 18 classes, incl. table/desk/toilet).

Also reports micro-averaged F1 and counts unmapped (hallucinated) predictions.

Usage: same args as eval.py, e.g.
  python eval_scannet18.py --metadata data/scannet_spatiallm/val.csv \
      --gt_dir data/scannet_spatiallm/layout --pred_dir data/scannet_spatiallm/pred_val \
      --label_mapping data/scannet_spatiallm/benchmark_categories.tsv \
      --label_from scannet18 --label_to scannet18
"""

import os
import argparse
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from terminaltables import AsciiTable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import (
    read_label_mapping,
    assign_class_map,
    assign_minimum_scale,
    get_entity_class,
    calc_bbox_tp,
)
from spatiallm import Layout

SCANNET18 = [
    name.replace("_", " ")
    for name in [
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookcase",
        "painting",
        "counter",
        "desk",
        "curtain",
        "refrigerator",
        "shower_curtain",
        "toilet",
        "sink",
        "bathtub",
        "otherfurniture",
    ]
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ScanNet-18 class F1 evaluation")
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--label_mapping", type=str, required=True)
    parser.add_argument("--label_from", type=str, default="scannet18")
    parser.add_argument("--label_to", type=str, default="scannet18")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    scene_id_list = df["id"].tolist()
    class_map = read_label_mapping(args.label_mapping, args.label_from, args.label_to)

    classwise_25: dict = defaultdict(list)
    classwise_50: dict = defaultdict(list)
    unmapped_counts: dict = defaultdict(int)
    unmapped_total = 0
    micro = {"tp25": 0, "tp50": 0, "pred": 0, "gt": 0}

    for scene_id in scene_id_list:
        with open(os.path.join(args.pred_dir, f"{scene_id}.txt"), "r") as f:
            pred_layout = Layout(f.read())
        with open(os.path.join(args.gt_dir, f"{scene_id}.txt"), "r") as f:
            gt_layout = Layout(f.read())

        for b in pred_layout.bboxes:
            if class_map.get(b.class_name.replace("_", " ")) is None:
                unmapped_counts[b.class_name] += 1
                unmapped_total += 1

        pred_layout.bboxes = assign_class_map(pred_layout.bboxes, class_map)
        gt_layout.bboxes = assign_class_map(gt_layout.bboxes, class_map)
        assign_minimum_scale(pred_layout.bboxes, minimum_scale=0.1)
        assign_minimum_scale(gt_layout.bboxes, minimum_scale=0.1)

        for class_name in SCANNET18:
            pred_entities = [
                b for b in pred_layout.bboxes if get_entity_class(b) == class_name
            ]
            gt_entities = [
                b for b in gt_layout.bboxes if get_entity_class(b) == class_name
            ]
            t25 = calc_bbox_tp(pred_entities, gt_entities, iou_threshold=0.25)
            t50 = calc_bbox_tp(pred_entities, gt_entities, iou_threshold=0.50)
            classwise_25[class_name].append(t25)
            classwise_50[class_name].append(t50)
            micro["tp25"] += t25.tp
            micro["tp50"] += t50.tp
            micro["pred"] += t25.num_pred
            micro["gt"] += t25.num_gt

    headers = ["Objects (scannet18)", "F1 @.25 IoU", "F1 @.50 IoU", "scenes"]
    table_data = [headers]
    f1s_25, f1s_50 = [], []
    for class_name in SCANNET18:
        f1_25 = np.ma.masked_where(
            [t.masked for t in classwise_25[class_name]],
            [t.f1 for t in classwise_25[class_name]],
        ).mean()
        f1_50 = np.ma.masked_where(
            [t.masked for t in classwise_50[class_name]],
            [t.f1 for t in classwise_50[class_name]],
        ).mean()
        n_active = sum(not t.masked for t in classwise_25[class_name])
        f1s_25.append(f1_25)
        f1s_50.append(f1_50)
        table_data.append(
            [class_name, f"{f1_25:.4f}", f"{f1_50:.4f}", f"{n_active}/{len(scene_id_list)}"]
        )
    print("\n" + AsciiTable(table_data).table)

    macro_25 = float(np.mean(f1s_25))
    macro_50 = float(np.mean(f1s_50))
    micro_25 = 2 * micro["tp25"] / (micro["pred"] + micro["gt"])
    micro_50 = 2 * micro["tp50"] / (micro["pred"] + micro["gt"])
    print(f"macro F1 (masked per-class mean, official style): @{0.25}: {macro_25:.4f}  @{0.50}: {macro_50:.4f}")
    print(f"micro F1 (all boxes pooled):                     @0.25: {micro_25:.4f}  @0.50: {micro_50:.4f}")
    print(f"gt boxes: {micro['gt']}, mapped pred boxes: {micro['pred']}, unmapped (hallucinated) pred boxes: {unmapped_total}")
    if unmapped_counts:
        top = sorted(unmapped_counts.items(), key=lambda kv: -kv[1])[:10]
        print("top unmapped pred classes: " + ", ".join(f"{c}:{n}" for c, n in top))
