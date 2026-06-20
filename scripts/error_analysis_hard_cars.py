#!/usr/bin/env python
"""Hard-car failure analysis for LOCO-CV folds.

Default mode is lightweight: it reads `history.json` files and writes a hard-car
summary. Add `--run_inference` to generate per-sample predictions and bucketed
error tables for CY02C/M6.
"""

import argparse
import csv
import json
import os
import pickle
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "feather"))
sys.path.insert(0, str(PROJECT_ROOT / "feather" / "data_utils"))
sys.path.insert(0, str(PROJECT_ROOT / "feather" / "model"))
sys.path.insert(0, str(PROJECT_ROOT / "models"))

ARCH_PATTERNS = {
    "E0": "pt_hicnet_loco_e0_fold*",
    "E1": "pt_hicnet_loco_e1_fold*",
    "E2": "pt_hicnet_loco_e2_fold*",
    "E3": "pt_hicnet_loco_fold*",
    "E4": "pt_hicnet_loco_e4_fold*",
}

ALL_VEHICLES = ["C201", "EP32", "JX65", "CY02C", "M6", "S50EVK", "FX11"]
NORMAL_CARS = ["C201", "EP32", "JX65", "S50EVK", "FX11"]
HARD_CARS = ["CY02C", "M6"]
PT_FILM = {"E2": "none", "E3": "global", "E4": "deep"}
MATERIAL_FEATURES = [f"mat_{i:02d}" for i in range(15)]
MATERIAL_FEATURE_LABELS = {
    "mat_00": "density",
    "mat_01": "young_modulus",
    "mat_02": "poisson_ratio",
    "mat_03": "stress_curve_0.001_0",
    "mat_04": "stress_curve_0.001_0.05",
    "mat_05": "stress_curve_0.001_0.1",
    "mat_06": "stress_curve_0.001_0.15",
    "mat_07": "stress_curve_0.001_0.2",
    "mat_08": "stress_curve_0.001_0.5",
    "mat_09": "stress_curve_1_0",
    "mat_10": "stress_curve_1_0.05",
    "mat_11": "stress_curve_1_0.1",
    "mat_12": "stress_curve_1_0.15",
    "mat_13": "stress_curve_1_0.2",
    "mat_14": "stress_curve_1_0.5",
}
HIC_BINS = [
    (0.0, 500.0, "<500"),
    (500.0, 1000.0, "500-1k"),
    (1000.0, 1500.0, "1k-1.5k"),
    (1500.0, 2000.0, "1.5k-2k"),
    (2000.0, float("inf"), ">2k"),
]
AGE_LABELS = {0: "Child", 1: "Adult"}


def parse_args():
    parser = argparse.ArgumentParser("error_analysis_hard_cars")
    parser.add_argument("--results_root", type=str, default="experiments")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output_dir", type=str, default="experiments/error_analysis/hard_cars")
    parser.add_argument("--architectures", type=str, nargs="+", default=["E0", "E2", "E3", "E4"])
    parser.add_argument("--vehicles", type=str, nargs="+", default=["CY02C", "M6"])
    parser.add_argument("--run_inference", action="store_true")
    parser.add_argument("--reuse_inference_csv", action="store_true")
    parser.add_argument("--run_data_diagnostics", action="store_true")
    parser.add_argument("--diagnostic_vehicles", type=str, nargs="+", default=ALL_VEHICLES)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--worst_n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def hic_bucket(value):
    value = float(value)
    for lo, hi, label in HIC_BINS:
        if lo <= value < hi:
            return label
    return "unknown"


def accuracy_ratio_np(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    p = np.abs(pred)
    t = np.abs(target)
    denom = np.maximum(p, t)
    return np.where(denom > 0, np.minimum(p, t) / denom, 0.0)


def find_run_dir(results_root, arch, vehicle):
    pattern = ARCH_PATTERNS[arch]
    matches = []
    for candidate in sorted(Path(results_root).glob(pattern)):
        parts = candidate.name.split("_")
        for i, part in enumerate(parts):
            if part.startswith("fold") and i + 1 < len(parts) and parts[i + 1] == vehicle:
                matches.append(candidate)
                break
    if not matches:
        raise FileNotFoundError(f"No run directory for arch={arch} vehicle={vehicle}")
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous run directories for arch={arch} vehicle={vehicle}: {matches}")
    return matches[0]


def best_history_metrics(run_dir):
    history_path = run_dir / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = [entry for entry in history if isinstance(entry.get("epoch"), int)]
    if not epochs:
        raise ValueError(f"No integer epochs in {history_path}")
    select_key = "val_accuracy" if "val_accuracy" in epochs[0] else "test_accuracy"
    best = max(epochs, key=lambda entry: entry.get(select_key, 0.0))
    val = best.get("val_accuracy", best.get("test_accuracy", 0.0))
    test = best.get("test_accuracy", best.get("val_accuracy", 0.0))
    return {
        "run_dir": str(run_dir),
        "best_epoch": best["epoch"],
        "val_accuracy": float(val),
        "test_accuracy": float(test),
        "gap": float(val - test),
        "test_mse": float(best.get("test_mse", best.get("val_mse", float("nan")))),
        "n_epochs": len(epochs),
    }


def collect_history_rows(results_root, architectures, vehicles):
    rows = []
    for arch in architectures:
        if arch not in ARCH_PATTERNS:
            raise ValueError(f"Unknown architecture: {arch}")
        for vehicle in vehicles:
            run_dir = find_run_dir(results_root, arch, vehicle)
            metrics = best_history_metrics(run_dir)
            rows.append({"arch": arch, "vehicle": vehicle, **metrics})
    return rows


def set_data_env(data_cfg):
    for env_key, yaml_key in [
        ("PT_HICNET_MATERIAL_LOOKUP_PATH", "material_lookup_path"),
        ("PT_HICNET_NORMALIZATION_PARAMS_PATH", "normalization_params_path"),
    ]:
        rel_path = data_cfg.get(yaml_key)
        if rel_path:
            os.environ[env_key] = str(PROJECT_ROOT / rel_path)


def resolve_project_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_vehicle_key(name, keys):
    candidates = {
        name,
        name.replace(" ", "_"),
        name.replace(" ", "-"),
        name.replace("_", "-"),
        name.replace("-", "_"),
    }
    match = re.match(r"^(\d+)([A-Za-z]\w*)$", name)
    if match:
        num_part, letter_part = match.groups()
        candidates.add(f"{num_part}_{letter_part}")
        candidates.add(f"{num_part}-{letter_part}")
    for candidate in list(candidates):
        candidates.add(candidate + ".xlsx")
    for key in candidates:
        if key in keys:
            return key
    return None


def load_material_lookup(data_cfg):
    path = resolve_project_path(data_cfg.get("material_lookup_path", "feather/material_lookup_by_vehicle.pkl"))
    with open(path, "rb") as f:
        return pickle.load(f), path


def load_material_normalization(data_cfg):
    path = resolve_project_path(data_cfg.get("normalization_params_path", "feather/normalization_params.pkl"))
    with open(path, "rb") as f:
        params = pickle.load(f)
    return params, path


def vector_signature(values, decimals=6):
    return tuple(round(float(value), decimals) for value in values)


def ids_to_str(values, limit=80):
    values = sorted(values)
    shown = values[:limit]
    suffix = "" if len(values) <= limit else f";...(+{len(values) - limit})"
    return ";".join(str(value) for value in shown) + suffix


def collect_material_lookup_diagnostics(cfg, vehicles):
    lookup, lookup_path = load_material_lookup(cfg["data"])
    rows = []
    vehicle_id_sets = {}
    vehicle_vec_sets = {}
    vehicle_key_map = {}

    for vehicle in vehicles:
        key = resolve_vehicle_key(vehicle, lookup.keys())
        vehicle_key_map[vehicle] = key or ""
        materials = lookup.get(key, {}) if key else {}
        id_set = set(materials.keys())
        vec_set = {vector_signature(vec) for vec in materials.values()}
        vehicle_id_sets[vehicle] = id_set
        vehicle_vec_sets[vehicle] = vec_set

    normal_vehicles = [v for v in vehicles if v in NORMAL_CARS]
    hard_vehicles = [v for v in vehicles if v in HARD_CARS]
    normal_id_union = set().union(*(vehicle_id_sets[v] for v in normal_vehicles)) if normal_vehicles else set()
    hard_id_union = set().union(*(vehicle_id_sets[v] for v in hard_vehicles)) if hard_vehicles else set()
    normal_vec_union = set().union(*(vehicle_vec_sets[v] for v in normal_vehicles)) if normal_vehicles else set()
    hard_vec_union = set().union(*(vehicle_vec_sets[v] for v in hard_vehicles)) if hard_vehicles else set()

    for vehicle in vehicles:
        id_set = vehicle_id_sets[vehicle]
        vec_set = vehicle_vec_sets[vehicle]
        id_only_vs_normal = id_set - normal_id_union if vehicle in HARD_CARS else set()
        vec_only_vs_normal = vec_set - normal_vec_union if vehicle in HARD_CARS else set()
        rows.append(
            {
                "vehicle": vehicle,
                "group": vehicle_group(vehicle),
                "lookup_key": vehicle_key_map[vehicle],
                "n_material_ids": len(id_set),
                "n_unique_vectors": len(vec_set),
                "n_id_overlap_with_normal_union": len(id_set & normal_id_union),
                "n_id_only_vs_normal_union": len(id_only_vs_normal),
                "id_only_vs_normal_union": ids_to_str(id_only_vs_normal),
                "n_vector_overlap_with_normal_union": len(vec_set & normal_vec_union),
                "n_vector_only_vs_normal_union": len(vec_only_vs_normal),
                "material_ids": ids_to_str(id_set),
            }
        )

    summary = [
        {
            "lookup_path": str(lookup_path),
            "normal_vehicle_count": len(normal_vehicles),
            "hard_vehicle_count": len(hard_vehicles),
            "normal_unique_ids": len(normal_id_union),
            "hard_unique_ids": len(hard_id_union),
            "id_overlap": len(normal_id_union & hard_id_union),
            "hard_only_ids": len(hard_id_union - normal_id_union),
            "hard_only_id_list": ids_to_str(hard_id_union - normal_id_union),
            "normal_unique_vectors": len(normal_vec_union),
            "hard_unique_vectors": len(hard_vec_union),
            "vector_overlap": len(normal_vec_union & hard_vec_union),
            "hard_only_vectors": len(hard_vec_union - normal_vec_union),
        }
    ]
    return {"summary": summary, "rows": rows}


def build_hard_only_material_reference(cfg, vehicles):
    lookup, _ = load_material_lookup(cfg["data"])
    try:
        norm_params, _ = load_material_normalization(cfg["data"])
    except Exception:
        norm_params = {}

    vehicle_vec_to_ids = {}
    vehicle_vec_to_values = {}
    for vehicle in vehicles:
        key = resolve_vehicle_key(vehicle, lookup.keys())
        materials = lookup.get(key, {}) if key else {}
        vec_to_ids = defaultdict(list)
        vec_to_values = {}
        for mid, vec in materials.items():
            sig = vector_signature(vec)
            vec_to_ids[sig].append(mid)
            vec_to_values[sig] = [float(value) for value in vec]
        vehicle_vec_to_ids[vehicle] = vec_to_ids
        vehicle_vec_to_values[vehicle] = vec_to_values

    normal_vec_union = set()
    for vehicle in vehicles:
        if vehicle in NORMAL_CARS:
            normal_vec_union.update(vehicle_vec_to_ids.get(vehicle, {}).keys())

    hard_only_by_vehicle = {}
    detail_rows = []
    for vehicle in vehicles:
        if vehicle not in HARD_CARS:
            hard_only_by_vehicle[vehicle] = set()
            continue
        hard_only = set(vehicle_vec_to_ids.get(vehicle, {}).keys()) - normal_vec_union
        hard_only_by_vehicle[vehicle] = hard_only
        for sig in sorted(hard_only):
            z_vec = vehicle_vec_to_values[vehicle][sig]
            row = {
                "vehicle": vehicle,
                "material_ids": ids_to_str(vehicle_vec_to_ids[vehicle][sig]),
                "signature": ";".join(f"{value:.6g}" for value in sig),
            }
            for i, name in enumerate(MATERIAL_FEATURES):
                row[f"{name}_z"] = z_vec[i]
                row[f"{name}_raw"] = inverse_material_mean(z_vec[i], i, norm_params)
            detail_rows.append(row)

    return {
        "hard_only_by_vehicle": hard_only_by_vehicle,
        "vehicle_vec_to_ids": vehicle_vec_to_ids,
        "hard_only_vector_rows": detail_rows,
    }


def inverse_material_mean(value, feature_idx, norm_params):
    if not norm_params or norm_params.get("method") != "z-score":
        return float("nan")
    mean_vals = norm_params.get("mean_vals")
    std_vals = norm_params.get("std_vals")
    if mean_vals is None or std_vals is None or feature_idx >= len(mean_vals):
        return float("nan")
    return float(value * std_vals[feature_idx] + mean_vals[feature_idx])


def material_feature_shift_raw_units(feature_group_comparison, cfg):
    try:
        norm_params, norm_path = load_material_normalization(cfg["data"])
    except Exception:
        return []
    rows = []
    for row in feature_group_comparison:
        match = re.match(r"^mat_(\d{2})_mean$", row["feature"])
        if not match:
            continue
        feature_idx = int(match.group(1))
        feature_name = f"mat_{feature_idx:02d}"
        normal_raw = inverse_material_mean(row["normal_mean"], feature_idx, norm_params)
        hard_raw = inverse_material_mean(row["hard_mean"], feature_idx, norm_params)
        rows.append(
            {
                "feature": row["feature"],
                "label": MATERIAL_FEATURE_LABELS.get(feature_name, feature_name),
                "normal_mean_z": row["normal_mean"],
                "hard_mean_z": row["hard_mean"],
                "diff_z": row["diff_hard_minus_normal"],
                "normal_mean_raw": normal_raw,
                "hard_mean_raw": hard_raw,
                "diff_raw": hard_raw - normal_raw if np.isfinite(normal_raw) and np.isfinite(hard_raw) else float("nan"),
                "cohen_d": row["cohen_d"],
                "normalization_method": norm_params.get("method", ""),
                "normalization_path": str(norm_path),
            }
        )
    return rows


def collect_data_files(root_dir):
    files = []
    for parent, _, names in os.walk(root_dir):
        for name in names:
            if name.endswith((".json", ".feather")):
                files.append(os.path.join(parent, name))
    return sorted(files)


def resolve_data_root(data_root):
    data_root = Path(data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    return data_root


def files_for_vehicle(data_root, vehicle):
    from data_utils.HICLoader_feather import CAR_TO_VEHICLE

    data_root = resolve_data_root(data_root)
    vehicle_to_car = {v: k for k, v in CAR_TO_VEHICLE.items()}
    car_dir = vehicle_to_car.get(vehicle, vehicle.lower())
    all_files = collect_data_files(str(data_root))
    files = [
        fp
        for fp in all_files
        if os.path.basename(os.path.dirname(fp)).lower() == car_dir.lower()
    ]
    if not files:
        raise FileNotFoundError(f"No data files for vehicle={vehicle} under {data_root}")
    return files


def vehicle_group(vehicle):
    return "hard" if vehicle in HARD_CARS else "normal"


def load_vehicle_datapoints(vehicle, cfg):
    from data_utils.HICLoader_feather import HICDataLoader

    runtime_args = argparse.Namespace(
        num_point=int(cfg["data"]["num_point"]),
        use_uniform_sample=bool(cfg["data"].get("use_uniform_sample", False)),
        use_normals=bool(cfg["data"].get("use_normals", False)),
    )
    data_root = cfg["data"].get("data_root", "")
    datapoints = []
    for fp in files_for_vehicle(data_root, vehicle):
        dataset = HICDataLoader(
            root=fp,
            args=runtime_args,
            early_fusion=True,
            normalize_thickness=bool(cfg["data"].get("normalize_thickness", True)),
        )
        if dataset.datapoints is None:
            len(dataset)
        for sample_idx, item in enumerate(dataset.datapoints):
            datapoints.append((fp, sample_idx, item))
    return datapoints


def safe_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def material_exposure_row(vehicle, fp, sample_idx, item, points, material, hic_point, material_ref):
    hard_only_set = material_ref["hard_only_by_vehicle"].get(vehicle, set())
    if vehicle not in HARD_CARS or not hard_only_set:
        return {
            "vehicle": vehicle,
            "file": os.path.basename(fp),
            "sample_index": sample_idx,
            "sample_info": str(item.get("sample_info", sample_idx)),
            "n_nodes": int(points.shape[0]),
            "n_hard_only_material_nodes": 0,
            "hard_only_material_frac": 0.0,
            "n_hard_only_vectors_in_sample": 0,
            "hard_only_material_ids_in_sample": "",
            "hard_only_min_dist_to_hic": float("nan"),
            "hard_only_mean_dist_to_hic": float("nan"),
            "hard_only_median_dist_to_hic": float("nan"),
            "hard_only_top32_nearest_frac": 0.0,
            "hard_only_nodes_within_60": 0,
            "hard_only_nodes_within_150": 0,
            "hard_only_nodes_within_400": 0,
            "hard_only_frac_within_60": 0.0,
            "hard_only_frac_within_150": 0.0,
            "hard_only_frac_within_400": 0.0,
        }

    if material.ndim != 2 or material.shape[1] != 15:
        return {
            "vehicle": vehicle,
            "file": os.path.basename(fp),
            "sample_index": sample_idx,
            "sample_info": str(item.get("sample_info", sample_idx)),
            "n_nodes": int(points.shape[0]),
            "n_hard_only_material_nodes": 0,
            "hard_only_material_frac": 0.0,
            "n_hard_only_vectors_in_sample": 0,
            "hard_only_material_ids_in_sample": "",
            "hard_only_min_dist_to_hic": float("nan"),
            "hard_only_mean_dist_to_hic": float("nan"),
            "hard_only_median_dist_to_hic": float("nan"),
            "hard_only_top32_nearest_frac": 0.0,
            "hard_only_nodes_within_60": 0,
            "hard_only_nodes_within_150": 0,
            "hard_only_nodes_within_400": 0,
            "hard_only_frac_within_60": 0.0,
            "hard_only_frac_within_150": 0.0,
            "hard_only_frac_within_400": 0.0,
        }

    signatures = [vector_signature(row) for row in material]
    mask = np.array([sig in hard_only_set for sig in signatures], dtype=bool)
    n_nodes = int(points.shape[0])
    n_hard = int(np.sum(mask))
    distances = np.linalg.norm(points - hic_point.reshape(1, 3), axis=1)
    k = min(32, n_nodes)
    nearest_idx = np.argpartition(distances, k - 1)[:k] if k > 0 else np.array([], dtype=np.int64)

    hard_sigs = sorted(set(sig for sig, is_hard in zip(signatures, mask) if is_hard))
    ids = []
    vehicle_vec_to_ids = material_ref["vehicle_vec_to_ids"].get(vehicle, {})
    for sig in hard_sigs:
        ids.extend(vehicle_vec_to_ids.get(sig, []))

    hard_distances = distances[mask]
    within_60 = int(np.sum(mask & (distances <= 60.0)))
    within_150 = int(np.sum(mask & (distances <= 150.0)))
    within_400 = int(np.sum(mask & (distances <= 400.0)))
    return {
        "vehicle": vehicle,
        "file": os.path.basename(fp),
        "sample_index": sample_idx,
        "sample_info": str(item.get("sample_info", sample_idx)),
        "n_nodes": n_nodes,
        "n_hard_only_material_nodes": n_hard,
        "hard_only_material_frac": n_hard / n_nodes if n_nodes else float("nan"),
        "n_hard_only_vectors_in_sample": len(hard_sigs),
        "hard_only_material_ids_in_sample": ids_to_str(set(ids)),
        "hard_only_min_dist_to_hic": float(np.min(hard_distances)) if n_hard else float("nan"),
        "hard_only_mean_dist_to_hic": float(np.mean(hard_distances)) if n_hard else float("nan"),
        "hard_only_median_dist_to_hic": float(np.median(hard_distances)) if n_hard else float("nan"),
        "hard_only_top32_nearest_frac": float(np.mean(mask[nearest_idx])) if k > 0 else float("nan"),
        "hard_only_nodes_within_60": within_60,
        "hard_only_nodes_within_150": within_150,
        "hard_only_nodes_within_400": within_400,
        "hard_only_frac_within_60": within_60 / n_hard if n_hard else 0.0,
        "hard_only_frac_within_150": within_150 / n_hard if n_hard else 0.0,
        "hard_only_frac_within_400": within_400 / n_hard if n_hard else 0.0,
    }


def collect_data_diagnostics(cfg, vehicles):
    scale_rows = []
    feature_rows = []
    exposure_rows = []
    vehicle_bounds = {}
    material_ref = build_hard_only_material_reference(cfg, vehicles)
    eps = 1e-8

    for vehicle in vehicles:
        print(f"[DataDiag] {vehicle}")
        vehicle_bounds[vehicle] = {
            "min": np.array([np.inf, np.inf, np.inf], dtype=np.float64),
            "max": np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64),
            "n_nodes_total": 0,
            "n_samples": 0,
        }
        for fp, sample_idx, item in load_vehicle_datapoints(vehicle, cfg):
            points = np.asarray(item["point_set"], dtype=np.float64)
            if points.ndim != 2 or points.shape[0] == 0:
                continue
            bbox_min = points.min(axis=0)
            bbox_max = points.max(axis=0)
            vehicle_bounds[vehicle]["min"] = np.minimum(vehicle_bounds[vehicle]["min"], bbox_min)
            vehicle_bounds[vehicle]["max"] = np.maximum(vehicle_bounds[vehicle]["max"], bbox_max)
            vehicle_bounds[vehicle]["n_nodes_total"] += int(points.shape[0])
            vehicle_bounds[vehicle]["n_samples"] += 1
            span = bbox_max - bbox_min
            diag = float(np.linalg.norm(span))
            center = (bbox_min + bbox_max) * 0.5
            hic_point = np.array(
                [
                    item["hic_point"]["x"],
                    item["hic_point"]["y"],
                    item["hic_point"]["z"],
                ],
                dtype=np.float64,
            )
            hic_rel = (hic_point - bbox_min) / (span + eps)
            hic_center_dist = float(np.linalg.norm(hic_point - center) / (diag + eps))
            age_raw = item.get("age_group", "Adult")
            age_group = 0 if str(age_raw).lower().startswith("child") else 1

            scale_rows.append(
                {
                    "vehicle": vehicle,
                    "group": vehicle_group(vehicle),
                    "file": os.path.basename(fp),
                    "sample_index": sample_idx,
                    "sample_info": str(item.get("sample_info", sample_idx)),
                    "n_nodes": int(points.shape[0]),
                    "bbox_x": float(span[0]),
                    "bbox_y": float(span[1]),
                    "bbox_z": float(span[2]),
                    "bbox_diag": diag,
                    "hic_rel_x": float(hic_rel[0]),
                    "hic_rel_y": float(hic_rel[1]),
                    "hic_rel_z": float(hic_rel[2]),
                    "hic_center_dist_norm": hic_center_dist,
                    "hic_value": float(item.get("hic_value", 0.0)),
                    "age_group": age_group,
                    "age_label": AGE_LABELS.get(age_group, "?"),
                }
            )

            thickness = np.asarray(item.get("thickness", []), dtype=np.float64).reshape(-1)
            material = np.asarray(item.get("material_props", []), dtype=np.float64)
            feature_row = {
                "vehicle": vehicle,
                "group": vehicle_group(vehicle),
                "file": os.path.basename(fp),
                "sample_index": sample_idx,
                "sample_info": str(item.get("sample_info", sample_idx)),
                "thickness_mean": float(np.mean(thickness)) if thickness.size else float("nan"),
                "thickness_std": float(np.std(thickness)) if thickness.size else float("nan"),
                "thickness_min": float(np.min(thickness)) if thickness.size else float("nan"),
                "thickness_max": float(np.max(thickness)) if thickness.size else float("nan"),
            }
            if material.ndim == 2 and material.shape[1] == 15:
                mat_mean = np.mean(material, axis=0)
                mat_std = np.std(material, axis=0)
                for i, name in enumerate(MATERIAL_FEATURES):
                    feature_row[f"{name}_mean"] = float(mat_mean[i])
                    feature_row[f"{name}_std"] = float(mat_std[i])
            else:
                for name in MATERIAL_FEATURES:
                    feature_row[f"{name}_mean"] = float("nan")
                    feature_row[f"{name}_std"] = float("nan")
            feature_rows.append(feature_row)
            exposure_rows.append(
                material_exposure_row(vehicle, fp, sample_idx, item, points, material, hic_point, material_ref)
            )

    feature_group_comparison = compare_feature_groups(feature_rows)
    material_lookup = collect_material_lookup_diagnostics(cfg, vehicles)

    return {
        "scale_rows": scale_rows,
        "feature_rows": feature_rows,
        "vehicle_union_summary": summarize_vehicle_union(vehicle_bounds),
        "vehicle_scale_summary": summarize_scale_by_vehicle(scale_rows),
        "feature_group_comparison": feature_group_comparison,
        "feature_group_comparison_raw_units": material_feature_shift_raw_units(feature_group_comparison, cfg),
        "material_lookup_summary": material_lookup["summary"],
        "material_lookup_rows": material_lookup["rows"],
        "hard_only_material_vector_rows": material_ref["hard_only_vector_rows"],
        "hard_only_material_exposure_rows": exposure_rows,
    }


def summarize_vehicle_union(vehicle_bounds):
    rows = []
    for vehicle, bounds in sorted(vehicle_bounds.items()):
        span = bounds["max"] - bounds["min"]
        diag = float(np.linalg.norm(span))
        rows.append(
            {
                "vehicle": vehicle,
                "group": vehicle_group(vehicle),
                "n_samples": int(bounds["n_samples"]),
                "n_nodes_total": int(bounds["n_nodes_total"]),
                "union_bbox_x": float(span[0]),
                "union_bbox_y": float(span[1]),
                "union_bbox_z": float(span[2]),
                "union_bbox_diag": diag,
            }
        )

    normal_diag = [
        row["union_bbox_diag"]
        for row in rows
        if row["vehicle"] in NORMAL_CARS and np.isfinite(row["union_bbox_diag"])
    ]
    normal_mean = float(np.mean(normal_diag)) if normal_diag else float("nan")
    for row in rows:
        row["union_diag_vs_normal_mean"] = (
            row["union_bbox_diag"] / normal_mean
            if normal_mean and np.isfinite(normal_mean)
            else float("nan")
        )
    return rows


def summarize_scale_by_vehicle(scale_rows):
    grouped = defaultdict(list)
    for row in scale_rows:
        grouped[row["vehicle"]].append(row)

    normal_diag_values = [
        row["bbox_diag"]
        for row in scale_rows
        if row["vehicle"] in NORMAL_CARS and np.isfinite(row["bbox_diag"])
    ]
    normal_diag_mean = float(np.mean(normal_diag_values)) if normal_diag_values else float("nan")

    out = []
    for vehicle, rows in sorted(grouped.items()):
        diag = safe_stats([row["bbox_diag"] for row in rows])
        out.append(
            {
                "vehicle": vehicle,
                "group": vehicle_group(vehicle),
                "count": len(rows),
                "bbox_diag_mean": diag["mean"],
                "bbox_diag_std": diag["std"],
                "bbox_diag_vs_normal_mean": (
                    diag["mean"] / normal_diag_mean if normal_diag_mean and np.isfinite(normal_diag_mean) else float("nan")
                ),
                "bbox_x_mean": safe_stats([row["bbox_x"] for row in rows])["mean"],
                "bbox_y_mean": safe_stats([row["bbox_y"] for row in rows])["mean"],
                "bbox_z_mean": safe_stats([row["bbox_z"] for row in rows])["mean"],
                "n_nodes_mean": safe_stats([row["n_nodes"] for row in rows])["mean"],
                "hic_rel_x_mean": safe_stats([row["hic_rel_x"] for row in rows])["mean"],
                "hic_rel_y_mean": safe_stats([row["hic_rel_y"] for row in rows])["mean"],
                "hic_rel_z_mean": safe_stats([row["hic_rel_z"] for row in rows])["mean"],
                "hic_center_dist_norm_mean": safe_stats([row["hic_center_dist_norm"] for row in rows])["mean"],
                "hic_value_mean": safe_stats([row["hic_value"] for row in rows])["mean"],
            }
        )
    return out


def compare_feature_groups(feature_rows):
    if not feature_rows:
        return []
    metric_keys = [
        key
        for key in feature_rows[0].keys()
        if key not in {"vehicle", "group", "file", "sample_index", "sample_info"}
    ]
    out = []
    for key in metric_keys:
        normal = np.array(
            [row[key] for row in feature_rows if row["group"] == "normal" and np.isfinite(row[key])],
            dtype=np.float64,
        )
        hard = np.array(
            [row[key] for row in feature_rows if row["group"] == "hard" and np.isfinite(row[key])],
            dtype=np.float64,
        )
        if normal.size == 0 or hard.size == 0:
            continue
        normal_std = float(np.std(normal))
        hard_std = float(np.std(hard))
        pooled = float(np.sqrt((normal_std ** 2 + hard_std ** 2) * 0.5))
        diff = float(np.mean(hard) - np.mean(normal))
        out.append(
            {
                "feature": key,
                "normal_mean": float(np.mean(normal)),
                "hard_mean": float(np.mean(hard)),
                "diff_hard_minus_normal": diff,
                "abs_diff": abs(diff),
                "cohen_d": diff / pooled if pooled > 1e-12 else float("nan"),
                "abs_cohen_d": abs(diff / pooled) if pooled > 1e-12 else float("nan"),
            }
        )
    return sorted(out, key=lambda row: row["abs_cohen_d"] if np.isfinite(row["abs_cohen_d"]) else -1, reverse=True)


def inplace_relu(module):
    if module.__class__.__name__.find("ReLU") != -1:
        module.inplace = True


def build_model(arch, checkpoint, data_cfg, device):
    use_normals = bool(data_cfg.get("use_normals", False))
    num_point = int(data_cfg.get("num_point", 8192))

    if arch == "E0":
        import pointnet2_reg_att_props as model_module

        model = model_module.get_model(normal_channel=use_normals, num_point=num_point)
    elif arch == "E1":
        import pointnet2_reg_ablation as model_module

        model = model_module.get_model(normal_channel=use_normals, num_point=num_point)
    elif arch in PT_FILM:
        from pt_hicnet import PT_HICnet

        hparams = checkpoint.get("model_hparams", {})
        model = PT_HICnet(
            in_channels=int(hparams.get("in_channels", 22 if use_normals else 19)),
            use_normals=bool(hparams.get("use_normals", use_normals)),
            film_mode=str(hparams.get("film_mode", PT_FILM[arch])),
            pt_radius=tuple(hparams.get("pt_radius", [60, 150, 400, 1500])),
            pt_nsample=tuple(hparams.get("pt_nsample", [32, 32, 32, 32])),
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    model.apply(inplace_relu)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def sample_meta(dataset, file_path):
    if dataset.datapoints is None:
        len(dataset)
    meta = []
    for idx, item in enumerate(dataset.datapoints):
        age_raw = item.get("age_group", "Adult")
        age_group = 0 if str(age_raw).lower().startswith("child") else 1
        meta.append(
            {
                "file": os.path.basename(file_path),
                "sample_index": idx,
                "sample_info": str(item.get("sample_info", idx)),
                "age_group": age_group,
                "age_label": AGE_LABELS.get(age_group, "?"),
                "hic_true_meta": float(item.get("hic_value", 0.0)),
            }
        )
    return meta


def infer_one_arch_vehicle(arch, vehicle, run_dir, cfg, args, device):
    from data_utils.HICLoader_feather import HICDataLoader

    set_seed(args.seed)
    checkpoint_path = run_dir / "checkpoints" / "best_acc_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(arch, checkpoint, cfg["data"], device)

    runtime_args = argparse.Namespace(
        num_point=int(cfg["data"]["num_point"]),
        use_uniform_sample=bool(cfg["data"].get("use_uniform_sample", False)),
        use_normals=bool(cfg["data"].get("use_normals", False)),
    )
    batch_size = args.batch_size or int(cfg["training"].get("batch_size", 15))
    data_root = cfg["data"].get("data_root", "")
    files = files_for_vehicle(data_root, vehicle)
    rows = []
    seen = 0

    for fp in files:
        early_fusion = arch != "E0"
        dataset = HICDataLoader(
            root=fp,
            args=runtime_args,
            early_fusion=early_fusion,
            normalize_thickness=bool(cfg["data"].get("normalize_thickness", True)),
        )
        meta = sample_meta(dataset, fp)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )
        offset = 0
        with torch.no_grad():
            for batch in loader:
                if early_fusion:
                    points, hic_point, category, age_group, target = batch
                    points = points.to(device).transpose(2, 1)
                    hic_point = hic_point.to(device)
                    category = category.to(device)
                    age_group = age_group.to(device)
                    target = target.to(device)
                    if arch == "E1":
                        pred, _ = model(points, hic_point, category, age_group)
                    else:
                        pred, _ = model(points, hic_point, age_group)
                else:
                    points, hic_point, category, thickness, material_props, age_group, target = batch
                    points = points.to(device).transpose(2, 1)
                    hic_point = hic_point.to(device)
                    category = category.to(device)
                    thickness = thickness.to(device)
                    material_props = material_props.to(device)
                    age_group = age_group.to(device)
                    target = target.to(device)
                    pred, _ = model(points, hic_point, category, thickness, material_props, age_group)

                pred_np = pred.detach().cpu().numpy().reshape(-1)
                target_np = target.detach().cpu().numpy().reshape(-1)
                age_np = age_group.detach().cpu().numpy().reshape(-1)
                acc_np = accuracy_ratio_np(pred_np, target_np)
                batch_meta = meta[offset : offset + len(pred_np)]
                offset += len(pred_np)

                for i, (pred_val, target_val, acc_val) in enumerate(zip(pred_np, target_np, acc_np)):
                    info = batch_meta[i]
                    signed_error = float(pred_val - target_val)
                    rows.append(
                        {
                            "arch": arch,
                            "vehicle": vehicle,
                            "run_dir": str(run_dir),
                            "checkpoint_epoch": checkpoint.get("epoch", ""),
                            "file": info["file"],
                            "sample_index": info["sample_index"],
                            "sample_info": info["sample_info"],
                            "age_group": int(age_np[i]),
                            "age_label": AGE_LABELS.get(int(age_np[i]), "?"),
                            "hic_true": float(target_val),
                            "hic_pred": float(pred_val),
                            "signed_error": signed_error,
                            "abs_error": abs(signed_error),
                            "squared_error": signed_error ** 2,
                            "accuracy": float(acc_val),
                            "hic_bucket": hic_bucket(target_val),
                        }
                    )
                    seen += 1
                    if args.max_samples is not None and seen >= args.max_samples:
                        return rows
    return rows


def group_stats(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    stats = []
    for key_vals, items in sorted(groups.items()):
        acc = np.array([item["accuracy"] for item in items], dtype=np.float64)
        se = np.array([item["squared_error"] for item in items], dtype=np.float64)
        out = {key: value for key, value in zip(keys, key_vals)}
        out.update(
            {
                "count": len(items),
                "mean_accuracy": float(acc.mean()) if len(acc) else float("nan"),
                "mean_mse": float(se.mean()) if len(se) else float("nan"),
                "neg_rate": float(np.mean([item["hic_pred"] < 0 for item in items])),
            }
        )
        stats.append(out)
    return stats


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def coerce_inference_rows(rows):
    numeric_fields = {
        "checkpoint_epoch": int,
        "sample_index": int,
        "age_group": int,
        "hic_true": float,
        "hic_pred": float,
        "signed_error": float,
        "abs_error": float,
        "squared_error": float,
        "accuracy": float,
    }
    out = []
    for row in rows:
        converted = dict(row)
        for key, caster in numeric_fields.items():
            if key not in converted or converted[key] == "":
                continue
            converted[key] = caster(float(converted[key])) if caster is int else caster(converted[key])
        out.append(converted)
    return out


def paired_delta_rows(inference_rows, base_arch="E0", compare_arch="E3"):
    by_key = {}
    for row in inference_rows:
        key = (row["vehicle"], row["file"], row["sample_info"])
        by_key.setdefault(key, {})[row["arch"]] = row

    pairs = []
    for key, arch_rows in sorted(by_key.items()):
        if base_arch not in arch_rows or compare_arch not in arch_rows:
            continue
        base = arch_rows[base_arch]
        comp = arch_rows[compare_arch]
        pairs.append(
            {
                "vehicle": key[0],
                "file": key[1],
                "sample_info": key[2],
                "age_label": comp["age_label"],
                "hic_bucket": comp["hic_bucket"],
                "hic_true": comp["hic_true"],
                f"{base_arch}_pred": base["hic_pred"],
                f"{compare_arch}_pred": comp["hic_pred"],
                f"{base_arch}_accuracy": base["accuracy"],
                f"{compare_arch}_accuracy": comp["accuracy"],
                f"{compare_arch}_minus_{base_arch}_accuracy": comp["accuracy"] - base["accuracy"],
                f"{base_arch}_abs_error": base["abs_error"],
                f"{compare_arch}_abs_error": comp["abs_error"],
                f"{compare_arch}_minus_{base_arch}_abs_error": comp["abs_error"] - base["abs_error"],
            }
        )
    return sorted(pairs, key=lambda row: row[f"{compare_arch}_minus_{base_arch}_accuracy"])


def pearson_corr(xs, ys):
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def join_material_exposure_with_delta(exposure_rows, pairwise_rows):
    exposure_by_key = {
        (row["vehicle"], row["file"], row["sample_info"]): row
        for row in exposure_rows
    }
    joined = []
    for row in pairwise_rows:
        key = (row["vehicle"], row["file"], row["sample_info"])
        exposure = exposure_by_key.get(key)
        if not exposure:
            continue
        out = dict(row)
        for key_name, value in exposure.items():
            if key_name in {"vehicle", "file", "sample_info"}:
                continue
            out[key_name] = value
        joined.append(out)
    return joined


def summarize_material_exposure_delta(joined_rows):
    if not joined_rows:
        return []
    delta_key = "E3_minus_E0_accuracy"
    summary = []
    by_vehicle = defaultdict(list)
    for row in joined_rows:
        by_vehicle[row["vehicle"]].append(row)

    for vehicle, rows in sorted(by_vehicle.items()):
        deltas = np.array([row[delta_key] for row in rows], dtype=np.float64)
        frac = np.array([row["hard_only_material_frac"] for row in rows], dtype=np.float64)
        top32 = np.array([row["hard_only_top32_nearest_frac"] for row in rows], dtype=np.float64)
        min_dist = np.array([row["hard_only_min_dist_to_hic"] for row in rows], dtype=np.float64)
        exposed = frac > 0
        exposed_rows = deltas[exposed]
        unexposed_rows = deltas[~exposed]
        summary.append(
            {
                "vehicle": vehicle,
                "count": len(rows),
                "exposed_count": int(np.sum(exposed)),
                "exposed_frac": float(np.mean(exposed)) if len(rows) else float("nan"),
                "mean_delta_pp": float(np.mean(deltas) * 100.0) if deltas.size else float("nan"),
                "mean_delta_exposed_pp": float(np.mean(exposed_rows) * 100.0) if exposed_rows.size else float("nan"),
                "mean_delta_unexposed_pp": float(np.mean(unexposed_rows) * 100.0) if unexposed_rows.size else float("nan"),
                "median_delta_exposed_pp": float(np.median(exposed_rows) * 100.0) if exposed_rows.size else float("nan"),
                "median_delta_unexposed_pp": float(np.median(unexposed_rows) * 100.0) if unexposed_rows.size else float("nan"),
                "pearson_frac_vs_delta": pearson_corr(frac, deltas),
                "pearson_top32_vs_delta": pearson_corr(top32, deltas),
                "pearson_min_dist_vs_delta": pearson_corr(min_dist, deltas),
                "mean_hard_only_frac": float(np.mean(frac)) if frac.size else float("nan"),
                "mean_hard_only_top32_frac": float(np.mean(top32)) if top32.size else float("nan"),
            }
        )
    return summary


def pct(value):
    return f"{value * 100:.2f}%"


def make_report(history_rows, inference_rows, stats, data_diag, pairwise_rows, args):
    lines = ["# Hard-Car Failure Analysis", ""]
    lines.append("## History Summary")
    lines.append("| Arch | Vehicle | Val | Test | Gap | Best Epoch |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in history_rows:
        lines.append(
            f"| {row['arch']} | {row['vehicle']} | {pct(row['val_accuracy'])} | "
            f"{pct(row['test_accuracy'])} | {row['gap'] * 100:.1f}pp | {row['best_epoch']} |"
        )

    lines.append("")
    if data_diag:
        lines.append("## Vehicle Union Scale Diagnostics")
        lines.append("| Vehicle | Group | Samples | Union Diag | vs Normal Mean | Union X | Union Y | Union Z |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for row in data_diag["vehicle_union_summary"]:
            lines.append(
                f"| {row['vehicle']} | {row['group']} | {row['n_samples']} | "
                f"{row['union_bbox_diag']:.0f} | {row['union_diag_vs_normal_mean']:.2f}x | "
                f"{row['union_bbox_x']:.0f} | {row['union_bbox_y']:.0f} | {row['union_bbox_z']:.0f} |"
            )

        lines.append("")
        lines.append("## Local Sample Scale Diagnostics")
        lines.append("| Vehicle | Group | Count | BBox Diag | vs Normal Mean | BBox X | BBox Y | BBox Z |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for row in data_diag["vehicle_scale_summary"]:
            lines.append(
                f"| {row['vehicle']} | {row['group']} | {row['count']} | "
                f"{row['bbox_diag_mean']:.0f} | {row['bbox_diag_vs_normal_mean']:.2f}x | "
                f"{row['bbox_x_mean']:.0f} | {row['bbox_y_mean']:.0f} | {row['bbox_z_mean']:.0f} |"
            )

        lines.append("")
        lines.append("## Largest Hard-vs-Normal Feature Shifts")
        lines.append("Values below are normalized feature means unless a raw-unit row is shown later.")
        lines.append("")
        lines.append("| Feature | Normal Mean (z) | Hard Mean (z) | Diff (z) | Cohen d |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in data_diag["feature_group_comparison"][:12]:
            lines.append(
                f"| {row['feature']} | {row['normal_mean']:.4g} | {row['hard_mean']:.4g} | "
                f"{row['diff_hard_minus_normal']:.4g} | {row['cohen_d']:.2f} |"
            )
        lines.append("")

        raw_rows = data_diag.get("feature_group_comparison_raw_units", [])
        if raw_rows:
            lines.append("## Material Feature Raw-Unit Check")
            lines.append("Material channels are Z-Score normalized at runtime. Physical interpretation should use these raw-unit estimates, not the z-score means directly.")
            lines.append("")
            lines.append("| Feature | Label | Normal Raw | Hard Raw | Diff Raw | Cohen d |")
            lines.append("|---|---|---:|---:|---:|---:|")
            for row in raw_rows[:12]:
                lines.append(
                    f"| {row['feature']} | {row['label']} | {row['normal_mean_raw']:.4g} | "
                    f"{row['hard_mean_raw']:.4g} | {row['diff_raw']:.4g} | {row['cohen_d']:.2f} |"
                )
            lines.append("")

        material_summary = data_diag.get("material_lookup_summary", [])
        material_rows = data_diag.get("material_lookup_rows", [])
        if material_summary:
            summary = material_summary[0]
            lines.append("## Material Lookup Coverage")
            lines.append(
                f"ID-level overlap is {summary['id_overlap']}/{summary['hard_unique_ids']} hard IDs, "
                f"while rounded vector-level overlap is {summary['vector_overlap']}/{summary['hard_unique_vectors']} hard material vectors."
            )
            lines.append("")
            lines.append("| Vehicle | IDs | Unique Vectors | ID Overlap vs Normal | ID Only vs Normal | Vector Overlap vs Normal | Vector Only vs Normal |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for row in material_rows:
                lines.append(
                    f"| {row['vehicle']} | {row['n_material_ids']} | {row['n_unique_vectors']} | "
                    f"{row['n_id_overlap_with_normal_union']} | {row['n_id_only_vs_normal_union']} | "
                    f"{row['n_vector_overlap_with_normal_union']} | {row['n_vector_only_vs_normal_union']} |"
                )
            lines.append("")

    if not inference_rows:
        lines.append("Inference was not run. Re-run with `--run_inference` for per-sample tables.")
        return "\n".join(lines) + "\n"

    lines.append("## Inference Summary")
    lines.append("| Arch | Vehicle | Count | Mean Acc | Mean MSE | Neg Rate |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in stats["by_arch_vehicle"]:
        lines.append(
            f"| {row['arch']} | {row['vehicle']} | {row['count']} | "
            f"{pct(row['mean_accuracy'])} | {row['mean_mse']:.0f} | {row['neg_rate']:.4f} |"
        )

    lines.append("")
    lines.append("## Worst Samples")
    lines.append("| Rank | Arch | Vehicle | Acc | HIC True | HIC Pred | Age | Bucket | Sample |")
    lines.append("|---:|---|---|---:|---:|---:|---|---|---|")
    worst = sorted(inference_rows, key=lambda row: row["accuracy"])[: args.worst_n]
    for rank, row in enumerate(worst, 1):
        lines.append(
            f"| {rank} | {row['arch']} | {row['vehicle']} | {pct(row['accuracy'])} | "
            f"{row['hic_true']:.0f} | {row['hic_pred']:.0f} | {row['age_label']} | "
            f"{row['hic_bucket']} | {row['sample_info']} |"
        )

    if pairwise_rows:
        lines.append("")
        lines.append("## E0 vs E3 Per-Sample Delta")
        lines.append("| Rank | Vehicle | E0 Acc | E3 Acc | E3-E0 | HIC True | Age | Bucket | Sample |")
        lines.append("|---:|---|---:|---:|---:|---:|---|---|---|")
        for rank, row in enumerate(pairwise_rows[: args.worst_n], 1):
            delta = row["E3_minus_E0_accuracy"]
            lines.append(
                f"| {rank} | {row['vehicle']} | {pct(row['E0_accuracy'])} | {pct(row['E3_accuracy'])} | "
                f"{delta * 100:.1f}pp | {row['hic_true']:.0f} | {row['age_label']} | "
                f"{row['hic_bucket']} | {row['sample_info']} |"
            )

    exposure_delta_summary = data_diag.get("hard_only_material_delta_summary", []) if data_diag else []
    if exposure_delta_summary:
        lines.append("")
        lines.append("## Hard-Only Material Exposure vs E0-E3 Delta")
        lines.append("| Vehicle | Count | Exposed | Mean Delta | Exposed Delta | Unexposed Delta | r(frac, delta) | r(top32, delta) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in exposure_delta_summary:
            lines.append(
                f"| {row['vehicle']} | {row['count']} | {row['exposed_count']} "
                f"({row['exposed_frac'] * 100:.1f}%) | {row['mean_delta_pp']:.2f}pp | "
                f"{row['mean_delta_exposed_pp']:.2f}pp | {row['mean_delta_unexposed_pp']:.2f}pp | "
                f"{row['pearson_frac_vs_delta']:.3f} | {row['pearson_top32_vs_delta']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    set_seed(args.seed)

    results_root = PROJECT_ROOT / args.results_root
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(PROJECT_ROOT / args.config)
    set_data_env(cfg["data"])

    history_rows = collect_history_rows(results_root, args.architectures, args.vehicles)
    inference_rows = []
    stats = {}
    data_diag = {}
    pairwise_rows = []
    exposure_delta_rows = []
    exposure_delta_summary = []

    if args.run_data_diagnostics:
        data_diag = collect_data_diagnostics(cfg, args.diagnostic_vehicles)
        write_csv(output_dir / "hard_car_scale_diagnostics.csv", data_diag["scale_rows"])
        write_csv(output_dir / "hard_vs_normal_feature_rows.csv", data_diag["feature_rows"])
        write_csv(output_dir / "hard_vs_normal_feature_shift.csv", data_diag["feature_group_comparison"])
        write_csv(
            output_dir / "hard_vs_normal_feature_shift_raw_units.csv",
            data_diag["feature_group_comparison_raw_units"],
        )
        write_csv(output_dir / "material_lookup_summary.csv", data_diag["material_lookup_summary"])
        write_csv(output_dir / "material_lookup_by_vehicle_summary.csv", data_diag["material_lookup_rows"])
        write_csv(output_dir / "hard_only_material_vectors.csv", data_diag["hard_only_material_vector_rows"])
        write_csv(output_dir / "hard_only_material_exposure.csv", data_diag["hard_only_material_exposure_rows"])
        write_csv(output_dir / "vehicle_union_scale_summary.csv", data_diag["vehicle_union_summary"])
        write_csv(output_dir / "vehicle_scale_summary.csv", data_diag["vehicle_scale_summary"])

    if args.reuse_inference_csv:
        inference_csv = output_dir / "hard_car_per_sample.csv"
        if not inference_csv.exists():
            raise FileNotFoundError(f"--reuse_inference_csv requested but missing {inference_csv}")
        inference_rows = coerce_inference_rows(read_csv_rows(inference_csv))
        stats = {
            "by_arch_vehicle": group_stats(inference_rows, ["arch", "vehicle"]),
            "by_bucket": group_stats(inference_rows, ["arch", "vehicle", "hic_bucket"]),
            "by_age": group_stats(inference_rows, ["arch", "vehicle", "age_label"]),
        }
        pairwise_rows = paired_delta_rows(inference_rows, base_arch="E0", compare_arch="E3")
        write_csv(output_dir / "hard_car_pairwise_e0_e3_delta.csv", pairwise_rows)
    elif args.run_inference:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        for row in history_rows:
            print(f"[Infer] {row['arch']} {row['vehicle']} from {row['run_dir']}")
            inference_rows.extend(
                infer_one_arch_vehicle(
                    row["arch"],
                    row["vehicle"],
                    Path(row["run_dir"]),
                    cfg,
                    args,
                    device,
                )
            )
        write_csv(output_dir / "hard_car_per_sample.csv", inference_rows)
        stats = {
            "by_arch_vehicle": group_stats(inference_rows, ["arch", "vehicle"]),
            "by_bucket": group_stats(inference_rows, ["arch", "vehicle", "hic_bucket"]),
            "by_age": group_stats(inference_rows, ["arch", "vehicle", "age_label"]),
        }
        pairwise_rows = paired_delta_rows(inference_rows, base_arch="E0", compare_arch="E3")
        write_csv(output_dir / "hard_car_pairwise_e0_e3_delta.csv", pairwise_rows)

    if data_diag and pairwise_rows:
        exposure_delta_rows = join_material_exposure_with_delta(
            data_diag.get("hard_only_material_exposure_rows", []),
            pairwise_rows,
        )
        exposure_delta_summary = summarize_material_exposure_delta(exposure_delta_rows)
        data_diag["hard_only_material_exposure_with_delta"] = exposure_delta_rows
        data_diag["hard_only_material_delta_summary"] = exposure_delta_summary
        write_csv(output_dir / "hard_only_material_exposure_with_delta.csv", exposure_delta_rows)
        write_csv(output_dir / "hard_only_material_delta_summary.csv", exposure_delta_summary)

    payload = {
        "architectures": args.architectures,
        "vehicles": args.vehicles,
        "history": history_rows,
        "inference_ran": bool(args.run_inference),
        "data_diagnostics_ran": bool(args.run_data_diagnostics),
        "stats": stats,
        "data_diagnostics": {
            "vehicle_scale_summary": data_diag.get("vehicle_scale_summary", []),
            "vehicle_union_summary": data_diag.get("vehicle_union_summary", []),
            "feature_group_comparison": data_diag.get("feature_group_comparison", [])[:30],
            "feature_group_comparison_raw_units": data_diag.get("feature_group_comparison_raw_units", [])[:30],
            "material_lookup_summary": data_diag.get("material_lookup_summary", []),
            "material_lookup_rows": data_diag.get("material_lookup_rows", []),
            "hard_only_material_vector_rows": data_diag.get("hard_only_material_vector_rows", []),
            "hard_only_material_delta_summary": exposure_delta_summary,
        },
        "pairwise_e0_e3_delta_count": len(pairwise_rows),
        "hard_only_material_exposure_delta_count": len(exposure_delta_rows),
    }
    (output_dir / "hard_car_analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "hard_car_analysis.md").write_text(
        make_report(history_rows, inference_rows, stats, data_diag, pairwise_rows, args),
        encoding="utf-8",
    )
    print(f"[OK] wrote {output_dir / 'hard_car_analysis.md'}")
    print(f"[OK] wrote {output_dir / 'hard_car_analysis.json'}")
    if args.run_inference or args.reuse_inference_csv:
        print(f"[OK] wrote {output_dir / 'hard_car_per_sample.csv'}")
        print(f"[OK] wrote {output_dir / 'hard_car_pairwise_e0_e3_delta.csv'}")
    if args.run_data_diagnostics:
        print(f"[OK] wrote {output_dir / 'vehicle_union_scale_summary.csv'}")
        print(f"[OK] wrote {output_dir / 'vehicle_scale_summary.csv'}")
        print(f"[OK] wrote {output_dir / 'hard_vs_normal_feature_shift_raw_units.csv'}")
        print(f"[OK] wrote {output_dir / 'material_lookup_summary.csv'}")
        print(f"[OK] wrote {output_dir / 'material_lookup_by_vehicle_summary.csv'}")
        print(f"[OK] wrote {output_dir / 'hard_only_material_vectors.csv'}")
        print(f"[OK] wrote {output_dir / 'hard_only_material_exposure.csv'}")
    if exposure_delta_rows:
        print(f"[OK] wrote {output_dir / 'hard_only_material_exposure_with_delta.csv'}")
        print(f"[OK] wrote {output_dir / 'hard_only_material_delta_summary.csv'}")


if __name__ == "__main__":
    main()
