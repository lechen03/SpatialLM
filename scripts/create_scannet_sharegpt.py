"""Generate ShareGPT-format training data for the converted ScanNet dataset.

Standalone version of spatiallm/tuner/create_dataset.py that avoids importing
the training stack (datasets/transformers), which is only available in the
training environment.
"""

import os
import json
import sys
from glob import glob

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatiallm.layout.layout import Layout

LAYOUT_S_PLACEHOLDER = "<|layout_s|>"
LAYOUT_E_PLACEHOLDER = "<|layout_e|>"
POINT_CLOUD_PLACEHOLDER = "<point_cloud>"

if __name__ == "__main__":
    dataset_dir = "data/scannet_spatiallm"
    split_csv = os.path.join(dataset_dir, "split.csv")
    code_template_file = "code_template.txt"
    dataset_name = "scannet"

    pcd_dir = os.path.join(dataset_dir, "pcd")
    layout_dir = os.path.join(dataset_dir, "layout")

    pcd_scene_ids = {
        os.path.basename(p).split(".")[0] for p in glob(os.path.join(pcd_dir, "*.ply"))
    }
    layout_scene_ids = {
        os.path.basename(p).split(".")[0] for p in glob(os.path.join(layout_dir, "*.txt"))
    }
    scene_ids = sorted(pcd_scene_ids & layout_scene_ids)

    df = pd.read_csv(split_csv, dtype=str)
    df.set_index("id", inplace=True)

    with open(code_template_file, "r") as f:
        code_template = f.read()

    dataset = {"train": [], "val": []}
    for scene_id in tqdm(scene_ids, desc="converting"):
        try:
            split = df.loc[scene_id, "split"]
            with open(os.path.join(layout_dir, f"{scene_id}.txt"), "r") as f:
                layout = Layout(f.read())
            language_string = layout.to_language_string()

            dataset[split].append(
                {
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"{POINT_CLOUD_PLACEHOLDER}Detect boxes. The reference code is as followed: {code_template}",
                        },
                        {
                            "from": "gpt",
                            "value": f"{LAYOUT_S_PLACEHOLDER}{language_string}{LAYOUT_E_PLACEHOLDER}",
                        },
                    ],
                    "point_clouds": [f"pcd/{scene_id}.ply"],
                }
            )
        except Exception as e:
            print(f"Error processing scene {scene_id}: {e}")

    for split in ["train", "val"]:
        out = os.path.join(dataset_dir, f"{dataset_name}_{split}.json")
        with open(out, "w") as f:
            json.dump(dataset[split], f, indent=2)
        print(f"{out}: {len(dataset[split])} samples")

    dataset_info = {
        f"{dataset_name}_{split}": {
            "file_name": f"{dataset_name}_{split}.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "point_clouds": "point_clouds",
            },
        }
        for split in ["train", "val"]
    }
    info_path = os.path.join(dataset_dir, "dataset_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            dataset_info = {**json.load(f), **dataset_info}
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"{info_path} updated")
