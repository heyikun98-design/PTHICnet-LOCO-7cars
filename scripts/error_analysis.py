#!/usr/bin/env python
"""Per-sample error analysis for PT-HICnet.

Loads a checkpoint, runs inference on the test set, and produces:
- Accuracy binned by HIC value and age_group
- Worst-N samples (lowest accuracy_ratio)
- Distribution plots (histogram, boxplot by bucket)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "feather"))
sys.path.insert(0, str(PROJECT_ROOT / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "feather" / "data_utils"))


def parse_args():
    parser = argparse.ArgumentParser("error_analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_acc_model.pth or best_model.pth")
    parser.add_argument("--config", type=str, default=None, help="Fallback config yaml")
    parser.add_argument("--output_dir", type=str, default="experiments/error_analysis")
    parser.add_argument("--worst_n", type=int, default=20, help="Number of worst-case samples to report")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def accuracy_ratio(pred, target):
    p = np.abs(pred)
    t = np.abs(target)
    denom = np.maximum(p, t)
    return np.where(denom > 0, np.minimum(p, t) / denom, 0.0)


def collect_data_files(root_dir):
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"data dir not found: {root_dir}")
    files = []
    for parent, _, names in os.walk(root_dir):
        for name in names:
            if name.endswith((".json", ".feather")):
                files.append(os.path.join(parent, name))
    return sorted(files)


def split_train_test(all_files, test_vehicles):
    from data_utils.HICLoader_feather import CAR_TO_VEHICLE

    vehicle_to_car = {v: k for k, v in CAR_TO_VEHICLE.items()}
    test_car_dirs = {vehicle_to_car.get(v, v.lower()) for v in test_vehicles}

    train_files, test_files = [], []
    for fp in all_files:
        parent = os.path.basename(os.path.dirname(fp))
        if parent.lower() in test_car_dirs:
            test_files.append(fp)
        else:
            train_files.append(fp)
    return train_files, test_files


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Resolve config
    config_path = args.config
    if config_path is None:
        ckpt_config = Path(args.checkpoint).resolve().parents[1] / "config_used.yaml"
        if ckpt_config.exists():
            config_path = str(ckpt_config)
        else:
            config_path = "configs/default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    # Env setup
    for env_key, yaml_key in [
        ("PT_HICNET_MATERIAL_LOOKUP_PATH", "material_lookup_path"),
        ("PT_HICNET_NORMALIZATION_PARAMS_PATH", "normalization_params_path"),
    ]:
        rel = data_cfg.get(yaml_key)
        if rel:
            os.environ[env_key] = str(PROJECT_ROOT / rel)

    from data_utils.HICLoader_feather import HICDataLoader
    from pt_hicnet import PT_HICnet

    # Build model
    model_hparams = ckpt.get("model_hparams", {})
    in_channels = model_hparams.get("in_channels", 19)
    film_mode = model_hparams.get("film_mode", "global")
    pt_radius = model_hparams.get("pt_radius", [60, 150, 400, 1500])
    pt_nsample = model_hparams.get("pt_nsample", [32, 32, 32, 32])

    model = PT_HICnet(
        in_channels=in_channels,
        film_mode=film_mode,
        pt_radius=tuple(pt_radius),
        pt_nsample=tuple(pt_nsample),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    ckpt_epoch = ckpt.get("epoch", "unknown")
    print(f"[Model] Loaded from epoch {ckpt_epoch}, film_mode={film_mode}")

    # Build test loader
    data_root = data_cfg.get("data_root", "")
    search_dir = os.path.join(data_root, data_cfg["train_data_dir"]) if data_cfg["train_data_dir"] else data_root
    all_files = collect_data_files(search_dir)
    test_vehicles = data_cfg.get("test_vehicles", ["JX65"])
    _, test_files = split_train_test(all_files, test_vehicles)
    print(f"Test files: {len(test_files)} (vehicles: {test_vehicles})")

    runtime_args = argparse.Namespace(
        num_point=int(data_cfg["num_point"]),
        use_uniform_sample=bool(data_cfg["use_uniform_sample"]),
        use_normals=bool(data_cfg["use_normals"]),
    )
    datasets = [
        HICDataLoader(root=fp, args=runtime_args, early_fusion=True, normalize_thickness=True)
        for fp in test_files
    ]
    loader = DataLoader(ConcatDataset(datasets), batch_size=int(train_cfg["batch_size"]), shuffle=False)

    # Inference
    all_preds, all_targets, all_age_groups = [], [], []
    with torch.no_grad():
        for fused_input, hic_point, _, age_group, target in loader:
            fused_input = fused_input.to(device).transpose(2, 1)
            hic_point = hic_point.to(device)
            age_group = age_group.to(device)
            pred, _ = model(fused_input, hic_point, age_group)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_age_groups.append(age_group.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()
    age_groups = np.concatenate(all_age_groups, axis=0).flatten()

    # Per-sample accuracy
    sample_acc = accuracy_ratio(preds, targets)
    sample_mse = (preds - targets) ** 2

    # HIC bins
    bins = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, float("inf"))]
    bin_labels = ["<500", "500-1k", "1k-1.5k", "1.5k-2k", ">2k"]

    # Age groups
    age_labels = {0: "Child", 1: "Adult"}

    # --- Print report ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_lines = []
    report_lines.append("# Error Analysis Report")
    report_lines.append(f"\nCheckpoint: `{args.checkpoint}` (epoch {ckpt_epoch})")
    report_lines.append(f"Samples: {len(preds)}")
    report_lines.append("")

    # Overall
    overall_acc = accuracy_ratio(preds, targets)
    overall_mse = np.mean(sample_mse)
    report_lines.append("## Overall")
    report_lines.append(f"- Mean Accuracy: {np.mean(overall_acc)*100:.2f}%")
    report_lines.append(f"- Mean MSE: {overall_mse:.0f}")
    report_lines.append(f"- Neg Rate: {float(np.mean(preds < 0)):.4f}")
    report_lines.append("")

    # By HIC bucket
    report_lines.append("## By HIC Bucket")
    report_lines.append("| Bucket | Count | Mean Acc | Mean MSE | Mean HIC |")
    report_lines.append("|---|---:|---:|---:|---:|")
    bucket_stats = {}
    for (lo, hi), label in zip(bins, bin_labels):
        mask = (targets >= lo) & (targets < hi)
        n = mask.sum()
        if n == 0:
            report_lines.append(f"| {label} | 0 | — | — | — |")
            continue
        acc = np.mean(sample_acc[mask]) * 100
        mse = np.mean(sample_mse[mask])
        mean_hic = np.mean(targets[mask])
        report_lines.append(f"| {label} | {n} | {acc:.2f}% | {mse:.0f} | {mean_hic:.0f} |")
        bucket_stats[label] = {"n": int(n), "acc": acc, "mse": float(mse), "mean_hic": float(mean_hic)}
    report_lines.append("")

    # By age_group
    report_lines.append("## By Age Group")
    report_lines.append("| Age | Count | Mean Acc | Mean MSE |")
    report_lines.append("|---|---:|---:|---:|")
    for ag_val, ag_name in age_labels.items():
        mask = age_groups == ag_val
        n = mask.sum()
        if n == 0:
            continue
        acc = np.mean(sample_acc[mask]) * 100
        mse = np.mean(sample_mse[mask])
        report_lines.append(f"| {ag_name} | {n} | {acc:.2f}% | {mse:.0f} |")
    report_lines.append("")

    # By HIC bucket × Age
    report_lines.append("## By HIC × Age")
    report_lines.append("| Bucket | Age | Count | Mean Acc | Mean MSE |")
    report_lines.append("|---|---:|---:|---:|")
    for (lo, hi), label in zip(bins, bin_labels):
        for ag_val, ag_name in age_labels.items():
            mask = (targets >= lo) & (targets < hi) & (age_groups == ag_val)
            n = mask.sum()
            if n == 0:
                continue
            acc = np.mean(sample_acc[mask]) * 100
            mse = np.mean(sample_mse[mask])
            report_lines.append(f"| {label} | {ag_name} | {n} | {acc:.2f}% | {mse:.0f} |")
    report_lines.append("")

    # Worst-N samples
    report_lines.append(f"## Worst {args.worst_n} Samples (lowest accuracy_ratio)")
    report_lines.append("| Rank | Accuracy | HIC True | HIC Pred | Age |")
    report_lines.append("|---|---:|---:|---:|---:|")
    worst_idx = np.argsort(sample_acc)[: args.worst_n]
    for rank, idx in enumerate(worst_idx, 1):
        report_lines.append(
            f"| {rank} | {sample_acc[idx]*100:.2f}% | {targets[idx]:.0f} | {preds[idx]:.0f} | "
            f"{age_labels.get(int(age_groups[idx]), '?')} |"
        )
    report_lines.append("")

    # Best-N samples
    report_lines.append(f"## Best {args.worst_n} Samples (highest accuracy_ratio)")
    report_lines.append("| Rank | Accuracy | HIC True | HIC Pred | Age |")
    report_lines.append("|---|---:|---:|---:|---:|")
    best_idx = np.argsort(sample_acc)[-args.worst_n:][::-1]
    for rank, idx in enumerate(best_idx, 1):
        report_lines.append(
            f"| {rank} | {sample_acc[idx]*100:.2f}% | {targets[idx]:.0f} | {preds[idx]:.0f} | "
            f"{age_labels.get(int(age_groups[idx]), '?')} |"
        )
    report_lines.append("")

    # Write report
    report_path = out_dir / "error_analysis.md"
    report_path.write_text("\n".join(report_lines))

    # Write JSON for programmatic use
    json_data = {
        "checkpoint": str(args.checkpoint),
        "ckpt_epoch": ckpt_epoch,
        "n_samples": len(preds),
        "overall": {
            "mean_acc": float(np.mean(overall_acc)),
            "mean_mse": float(overall_mse),
            "neg_rate": float(np.mean(preds < 0)),
        },
        "by_bucket": bucket_stats,
        "worst_n": [
            {
                "rank": i + 1,
                "accuracy": float(sample_acc[idx]),
                "hic_true": float(targets[idx]),
                "hic_pred": float(preds[idx]),
                "age_group": int(age_groups[idx]),
            }
            for i, idx in enumerate(worst_idx)
        ],
    }
    json_path = out_dir / "error_analysis.json"
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False))

    print(f"Report saved: {report_path}")
    print(f"JSON saved: {json_path}")
    print(f"\nQuick stats:")
    print(f"  Overall Acc: {np.mean(overall_acc)*100:.2f}%")
    print(f"  Overall MSE: {overall_mse:.0f}")
    for (lo, hi), label in zip(bins, bin_labels):
        mask = (targets >= lo) & (targets < hi)
        if mask.sum() > 0:
            print(f"  {label}: n={mask.sum()}, acc={np.mean(sample_acc[mask])*100:.2f}%")


if __name__ == "__main__":
    main()
