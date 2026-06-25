import argparse
import hashlib
import importlib
import json
import os
import sys
import warnings
from pathlib import Path


def check_dependencies():
    required_modules = ["torch", "pandas", "pyarrow", "scipy", "yaml"]
    missing = []
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - fail-fast path
            missing.append(f"{module_name} ({exc})")
    if missing:
        raise RuntimeError(
            "Dependency preflight failed. Missing/invalid modules: " + "; ".join(missing)
        )


check_dependencies()

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "feather"))
sys.path.insert(0, str(PROJECT_ROOT / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "feather" / "data_utils"))


def parse_args():
    parser = argparse.ArgumentParser("eval_pt_hicnet")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="experiments/eval_pt_hicnet.json")
    return parser.parse_args()


def _to_hashable(obj):
    if isinstance(obj, dict):
        return {k: _to_hashable(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, tuple):
        return [_to_hashable(v) for v in obj]
    if isinstance(obj, list):
        return [_to_hashable(v) for v in obj]
    return obj


def build_hash_payload(cfg, resolved_delta):
    return {
        "model": _to_hashable(cfg.get("model", {})),
        "training": _to_hashable(cfg.get("training", {})),
        "loss": {
            "type": cfg.get("loss", {}).get("type", "kde_weighted_huber"),
            "delta": float(resolved_delta),
        },
    }


def compute_config_hash(cfg, resolved_delta):
    payload = build_hash_payload(cfg, resolved_delta)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_value(v):
    if isinstance(v, tuple):
        return list(v)
    return v


def resolve_delta(loss_cfg, checkpoint=None, default_delta=5.0):
    source = "default"
    delta = float(default_delta)

    if isinstance(loss_cfg, dict) and loss_cfg.get("delta") is not None:
        delta = float(loss_cfg["delta"])
        source = "yaml"

    if checkpoint is not None:
        ckpt_loss = checkpoint.get("resolved_loss", {})
        ckpt_delta = ckpt_loss.get("delta")
        if ckpt_delta is not None:
            ckpt_delta = float(ckpt_delta)
            if abs(ckpt_delta - delta) > 1e-12:
                warnings.warn(
                    f"[LossConfig] delta override: {delta:.6g} ({source}) -> {ckpt_delta:.6g} (checkpoint)",
                    UserWarning,
                )
            delta = ckpt_delta
            source = "checkpoint"

    print(f"[LossConfig] resolved delta={delta:.6g} source={source}")
    return delta, source


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


def neg_rate(pred):
    return float(np.mean(pred < 0))


def vehicle_key(fp):
    """从文件路径提取车型标识（用于按车分组报告）。"""
    from data_utils.HICLoader_feather import CAR_TO_VEHICLE

    parent = os.path.basename(os.path.dirname(fp))
    if parent.lower() in CAR_TO_VEHICLE:
        return CAR_TO_VEHICLE[parent.lower()]
    return parent


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    config_path = args.config
    if config_path is None:
        ckpt_config = Path(args.checkpoint).resolve().parents[1] / "config_used.yaml"
        if ckpt_config.exists():
            config_path = str(ckpt_config)
        else:
            config_path = "configs/default.yaml"
            warnings.warn(
                f"[Config] config_used.yaml not found near checkpoint, fallback to {config_path}",
                UserWarning,
            )

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

    from data_utils.HICLoader_feather import HICDataLoader  # noqa: E402
    from pt_hicnet import PT_HICnet  # noqa: E402

    model_hparams = checkpoint.get("model_hparams")
    if not isinstance(model_hparams, dict):
        raise ValueError("Checkpoint missing model_hparams; cannot ensure eval consistency.")

    yaml_model_cfg = cfg.get("model", {})
    mismatch = []
    for key, ckpt_val in model_hparams.items():
        if key not in yaml_model_cfg:
            mismatch.append((key, "<missing>", ckpt_val))
            continue
        yaml_val = yaml_model_cfg.get(key)
        if _normalize_value(yaml_val) != _normalize_value(ckpt_val):
            mismatch.append((key, yaml_val, ckpt_val))
    if mismatch:
        details = "; ".join([f"{k}: yaml={y} ckpt={c}" for k, y, c in mismatch])
        raise ValueError(
            "Model hyperparameters mismatch between YAML and checkpoint. "
            f"Checkpoint is the single source of truth. Details: {details}"
        )

    resolved_delta, _ = resolve_delta(cfg.get("loss", {}), checkpoint=checkpoint)
    current_hash = compute_config_hash(cfg, resolved_delta)
    expected_hash = checkpoint.get("config_hash")
    if expected_hash is None:
        hash_file = Path(args.checkpoint).resolve().parents[1] / "config_hash.txt"
        if hash_file.exists():
            expected_hash = hash_file.read_text(encoding="utf-8").strip()
    if expected_hash is None:
        raise ValueError("Checkpoint missing config hash metadata (config_hash/config_hash.txt).")
    if current_hash != expected_hash:
        raise ValueError(
            f"Config hash mismatch. current={current_hash}, checkpoint={expected_hash}. Abort eval."
        )
    print(f"[Config] hash verified: {current_hash}")

    runtime_args = argparse.Namespace(
        num_point=int(data_cfg["num_point"]),
        use_uniform_sample=bool(data_cfg["use_uniform_sample"]),
        use_normals=bool(data_cfg["use_normals"]),
    )

    # B1: 拼接 data_root，收集测试文件
    data_root = data_cfg.get("data_root", "")
    search_dir = os.path.join(data_root, data_cfg["test_data_dir"]) if data_cfg["test_data_dir"] else data_root
    all_files = collect_data_files(search_dir)
    test_vehicles = data_cfg.get("test_vehicles", ["JX65"])

    # 筛选测试车辆文件
    from data_utils.HICLoader_feather import CAR_TO_VEHICLE
    vehicle_to_car = {v: k for k, v in CAR_TO_VEHICLE.items()}
    test_car_dirs = {vehicle_to_car.get(v, v.lower()) for v in test_vehicles}
    files = [fp for fp in all_files if os.path.basename(os.path.dirname(fp)).lower() in test_car_dirs]

    if not files:
        raise FileNotFoundError(f"no test files found for vehicles {test_vehicles} in {search_dir}")
    print(f"Eval on {len(files)} test files (vehicles: {test_vehicles})")

    model = PT_HICnet(
        in_channels=int(model_hparams["in_channels"]),
        use_normals=bool(model_hparams["use_normals"]),
        film_mode=str(model_hparams["film_mode"]),
        pt_radius=tuple(model_hparams["pt_radius"]),
        pt_nsample=tuple(model_hparams["pt_nsample"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    eval_batch_size = int(checkpoint.get("training_hparams", {}).get("batch_size", 1))

    grouped_pred = {}
    grouped_true = {}
    with torch.no_grad():
        for fp in files:
            ds = HICDataLoader(
                root=fp,
                args=runtime_args,
                early_fusion=True,
                normalize_thickness=bool(data_cfg.get("normalize_thickness", True)),
                eval_deterministic=True,
            )
            loader = DataLoader(ds, batch_size=eval_batch_size, shuffle=False, num_workers=0)
            key = vehicle_key(fp)
            grouped_pred.setdefault(key, [])
            grouped_true.setdefault(key, [])
            for fused_input, hic_point, _, age_group, target in loader:
                fused_input = fused_input.to(device).transpose(2, 1)
                hic_point = hic_point.to(device)
                age_group = age_group.to(device)
                pred, _ = model(fused_input, hic_point, age_group)
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
            "neg_rate": neg_rate(all_pred),
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
