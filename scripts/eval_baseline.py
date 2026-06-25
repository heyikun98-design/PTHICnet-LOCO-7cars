"""Minimal evaluation script for E0/E1 baseline (PointNet++) checkpoints."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "feather"))
sys.path.insert(0, str(PROJECT_ROOT / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "feather" / "data_utils"))
sys.path.insert(0, str(PROJECT_ROOT / "feather" / "model"))


def parse_args():
    parser = argparse.ArgumentParser("eval_baseline")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ablation_mode", type=str, required=True,
                        choices=["baseline", "early_fusion_clean"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default="experiments/eval_baseline.json")
    return parser.parse_args()


def _detect_ablation_mode(state_dict):
    """Infer ablation mode from checkpoint state_dict layer names."""
    for key in state_dict.keys():
        if "cross_attn" in key.lower() or "crossattention" in key.lower():
            return "baseline"
    return "early_fusion_clean"


def collect_data_files(root_dir):
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"data dir not found: {root_dir}")
    files = []
    for parent, _, names in os.walk(root_dir):
        for name in names:
            if name.endswith((".json", ".feather")):
                files.append(os.path.join(parent, name))
    return sorted(files)


def accuracy_ratio(pred, target):
    p = np.abs(pred)
    t = np.abs(target)
    denom = np.maximum(p, t)
    return np.where(denom > 0, np.minimum(p, t) / denom, 0.0)


def vehicle_key(fp):
    from data_utils.HICLoader_feather import CAR_TO_VEHICLE
    parent = os.path.basename(os.path.dirname(fp))
    if parent.lower() in CAR_TO_VEHICLE:
        return CAR_TO_VEHICLE[parent.lower()]
    return parent


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    ablation_mode = args.ablation_mode
    if ablation_mode == "auto":
        ablation_mode = _detect_ablation_mode(checkpoint["model_state_dict"])

    config_path = args.config
    if config_path is None:
        ckpt_dir = Path(args.checkpoint).resolve().parents[1]
        for cand in ["config_used.yaml", "config.yaml"]:
            if (ckpt_dir / cand).exists():
                config_path = str(ckpt_dir / cand)
                break
    if config_path is None or not Path(config_path).exists():
        config_path = str(PROJECT_ROOT / "configs" / "default.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]

    for env_key, yaml_key in [
        ("PT_HICNET_MATERIAL_LOOKUP_PATH", "material_lookup_path"),
        ("PT_HICNET_NORMALIZATION_PARAMS_PATH", "normalization_params_path"),
    ]:
        rel = data_cfg.get(yaml_key)
        if rel:
            os.environ[env_key] = str(PROJECT_ROOT / rel)

    from data_utils.HICLoader_feather import HICDataLoader, CAR_TO_VEHICLE

    use_early_fusion = (ablation_mode == "early_fusion_clean")

    if ablation_mode == "early_fusion_clean":
        from pointnet2_reg_ablation import get_model
        model = get_model(normal_channel=False)
    else:
        from pointnet2_reg_att_props import get_model
        model = get_model(normal_channel=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    runtime_args = argparse.Namespace(
        num_point=int(data_cfg["num_point"]),
        use_uniform_sample=bool(data_cfg.get("use_uniform_sample", False)),
        use_normals=bool(data_cfg.get("use_normals", False)),
    )

    data_root = data_cfg.get("data_root", "")
    search_dir = os.path.join(data_root, data_cfg["test_data_dir"]) if data_cfg.get("test_data_dir") else data_root
    all_files = collect_data_files(search_dir)
    test_vehicles = data_cfg.get("test_vehicles", ["JX65"])
    vehicle_to_car = {v: k for k, v in CAR_TO_VEHICLE.items()}
    test_car_dirs = {vehicle_to_car.get(v, v.lower()) for v in test_vehicles}
    files = [fp for fp in all_files if os.path.basename(os.path.dirname(fp)).lower() in test_car_dirs]

    if not files:
        raise FileNotFoundError(f"no test files found for vehicles {test_vehicles} in {search_dir}")
    print(f"Eval on {len(files)} test files (vehicles: {test_vehicles})")

    eval_batch_size = int(cfg.get("training", {}).get("batch_size", 15))

    grouped_pred = {}
    grouped_true = {}
    with torch.no_grad():
        for fp in files:
            ds = HICDataLoader(
                root=fp, args=runtime_args,
                early_fusion=use_early_fusion,
                normalize_thickness=bool(data_cfg.get("normalize_thickness", True)),
                eval_deterministic=True,
            )
            loader = DataLoader(ds, batch_size=eval_batch_size, shuffle=False, num_workers=0)
            key = vehicle_key(fp)
            grouped_pred.setdefault(key, [])
            grouped_true.setdefault(key, [])
            for batch in loader:
                if use_early_fusion:
                    fused_input, hic_point, category, age_group, target = batch
                    fused_input = fused_input.to(device).transpose(2, 1)
                    hic_point = hic_point.to(device)
                    category = category.to(device)
                    age_group = age_group.to(device)
                    pred, _ = model(fused_input, hic_point, category, age_group)
                else:
                    points, hic_point, category, thickness, material_props, age_group, target = batch
                    points = points.to(device).transpose(2, 1)
                    hic_point = hic_point.to(device)
                    category = category.to(device)
                    thickness = thickness.to(device)
                    material_props = material_props.to(device)
                    age_group = age_group.to(device)
                    pred, _ = model(points, hic_point, category, thickness, material_props, age_group)
                grouped_pred[key].append(pred.detach().cpu().numpy())
                grouped_true[key].append(target.detach().cpu().numpy())

    metrics = {}
    all_pred = []
    all_true = []
    for key in sorted(grouped_pred.keys()):
        pred = np.concatenate(grouped_pred[key], axis=0)
        true = np.concatenate(grouped_true[key], axis=0)
        all_pred.append(pred)
        all_true.append(true)
        metrics[key] = {
            "mse": float(np.mean((pred - true) ** 2)),
            "accuracy": float(np.mean(accuracy_ratio(pred, true))),
            "count": int(pred.shape[0]),
        }
    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    result = {
        "overall": {
            "mse": float(np.mean((all_pred - all_true) ** 2)),
            "accuracy": float(np.mean(accuracy_ratio(all_pred, all_true))),
            "count": int(all_pred.shape[0]),
        },
        "by_vehicle": metrics,
    }

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved eval: {output_path}")


if __name__ == "__main__":
    main()
