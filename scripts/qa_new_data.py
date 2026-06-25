#!/usr/bin/env python
"""Phase 5 新数据 QA 脚本 — 训练前必跑。

检查项：
  1. 车型数量、样本量
  2. Bbox 尺度（union + per-sample）
  3. HIC 分布（mean, max, >2k, invalid）
  4. 材料向量覆盖（vs old 5-normal union）
  5. Hard-only material vector 报告

输出：experiments/phase5_qa/qa_report.json + qa_summary.csv

用法：
  python scripts/qa_new_data.py \
    --data_dir 车模型数据 \
    --old_normal_vehicles C201,EP32,JX65,S50EVK,FX11 \
    --output_dir experiments/phase5_qa
"""

import argparse
import csv
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "feather"))
sys.path.insert(0, str(PROJECT_ROOT / "feather" / "data_utils"))

# ---- material feature labels ----
MAT_LABELS = {
    "mat_00": "density", "mat_01": "young_modulus", "mat_02": "poisson_ratio",
    "mat_03": "stress_0.001_0", "mat_04": "stress_0.001_0.05",
    "mat_05": "stress_0.001_0.1", "mat_06": "stress_0.001_0.15",
    "mat_07": "stress_0.001_0.2", "mat_08": "stress_0.001_0.5",
    "mat_09": "stress_1_0", "mat_10": "stress_1_0.05",
    "mat_11": "stress_1_0.1", "mat_12": "stress_1_0.15",
    "mat_13": "stress_1_0.2", "mat_14": "stress_1_0.5",
}
MAT_KEYS = [f"mat_{i:02d}" for i in range(15)]


def parse_args():
    p = argparse.ArgumentParser("qa_new_data")
    p.add_argument("--data_dir", type=str, default="车模型数据")
    p.add_argument("--old_normal_vehicles", type=str, default="C201,EP32,JX65,S50EVK,FX11")
    p.add_argument("--output_dir", type=str, default="experiments/phase5_qa")
    p.add_argument("--material_lookup_path", type=str,
                   default="feather/material_lookup_by_vehicle.pkl")
    p.add_argument("--normalization_params_path", type=str,
                   default="feather/normalization_params.pkl")
    return p.parse_args()


# ---- vehicle extraction (mirrors HICLoader_feather) ----
try:
    from data_utils.HICLoader_feather import CAR_TO_VEHICLE
except ImportError:
    CAR_TO_VEHICLE = {}


def extract_vehicle_identifier(file_path):
    """从文件路径提取车辆标识符。"""
    parent_folder = os.path.basename(os.path.dirname(file_path))
    if parent_folder.lower() in CAR_TO_VEHICLE:
        return CAR_TO_VEHICLE[parent_folder.lower()]
    return parent_folder


def collect_feather_files(data_dir):
    """递归收集所有 .feather 文件，按 car 目录分组。"""
    car_files = defaultdict(list)
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"[ERROR] data_dir not found: {data_dir}")
        return car_files
    for fp in data_path.rglob("*.feather"):
        car_dir = fp.parent.name if fp.parent != data_path else "unknown"
        car_files[car_dir].append(str(fp))
    return dict(sorted(car_files.items()))


# ---- data loading (matches real nested feather format) ----
def _safe_float(val, default=0.0):
    """安全转换为浮点数（兼容字符串和 None）。"""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_vehicle_key(name, keys):
    candidates = {
        name,
        name.replace(" ", "_"),
        name.replace(" ", "-"),
        name.replace("_", "-"),
        name.replace("-", "_"),
    }
    candidates |= {f"{candidate}.xlsx" for candidate in list(candidates)}
    for candidate in candidates:
        if candidate in keys:
            return candidate
    return None


def load_material_assets(material_lookup_path, normalization_params_path):
    lookup_path = _resolve_path(material_lookup_path)
    norm_path = _resolve_path(normalization_params_path)
    if not lookup_path.exists():
        raise FileNotFoundError(f"material lookup not found: {lookup_path}")
    if not norm_path.exists():
        raise FileNotFoundError(f"normalization params not found: {norm_path}")
    with open(lookup_path, "rb") as f:
        lookup = pickle.load(f)
    with open(norm_path, "rb") as f:
        normalization = pickle.load(f)
    print(f"Material lookup: {lookup_path}")
    print(f"Normalization params: {norm_path}")
    return lookup, normalization


def normalize_material_vector(values, normalization):
    values = np.asarray(values, dtype=np.float64)
    method = normalization.get("method", "min-max")
    if method == "z-score":
        mean_vals = np.asarray(normalization["mean_vals"], dtype=np.float64)
        std_vals = np.asarray(normalization["std_vals"], dtype=np.float64)
        std_vals = np.where(np.abs(std_vals) > 1e-12, std_vals, 1.0)
        return (values - mean_vals) / std_vals
    min_vals = np.asarray(normalization["min_vals"], dtype=np.float64)
    max_vals = np.asarray(normalization["max_vals"], dtype=np.float64)
    denom = np.where(np.abs(max_vals - min_vals) > 1e-12, max_vals - min_vals, 1.0)
    return (values - min_vals) / denom


def material_lookup_for_vehicle(vehicle, material_lookup):
    key = _resolve_vehicle_key(vehicle, material_lookup.keys())
    if key is None:
        return None
    return {
        int(mat_id): np.asarray(mat_vec, dtype=np.float64)
        for mat_id, mat_vec in material_lookup[key].items()
    }


def load_feather_samples(fp, vehicle_materials, normalization):
    """加载一个 feather 文件，返回样本列表。

    真实格式: file_id (str) + data (struct{hic_point, nearby_nodes})
    """
    samples = []
    stats = {
        "raw_samples": 0,
        "hic_zero_count": 0,
        "hic_missing_count": 0,
        "material_nodes": 0,
        "unresolved_material_nodes": 0,
        "unresolved_material_ids": set(),
    }
    try:
        df = pd.read_feather(fp)
    except Exception as e:
        print(f"  [WARN] Failed to read {fp}: {e}")
        return samples, stats

    for _, row in df.iterrows():
        stats["raw_samples"] += 1
        file_id = str(row["file_id"])
        point_data = row["data"]

        # hic_point
        hic_point = point_data.get("hic_point", {})
        hic_raw = hic_point.get("HIC", hic_point.get("hic_value", None))
        if hic_raw in (None, ""):
            stats["hic_missing_count"] += 1
            continue
        hic_value = _safe_float(hic_raw, None)
        if hic_value is None:
            stats["hic_missing_count"] += 1
            continue
        if hic_value == 0:
            stats["hic_zero_count"] += 1
            continue

        age_group = hic_point.get("age_group", "Adult")
        age_code = 1 if str(age_group).lower().startswith("adult") else 0

        # nearby_nodes → bbox + material vectors
        nearby = point_data.get("nearby_nodes", [])
        xyz_list = []
        mat_sigs = set()

        has_direct_mat = False
        if len(nearby) > 0:
            first = nearby[0]
            has_direct_mat = all(k in first for k in ["密度RO", "杨氏模量E", "泊松比PR"])

        for node in nearby:
            x = _safe_float(node.get("X", 0))
            y = _safe_float(node.get("Y", 0))
            z = _safe_float(node.get("Z", 0))
            xyz_list.append([x, y, z])

            if has_direct_mat:
                raw_mat_vec = [
                    _safe_float(node.get("密度RO", 0) or node.get("密度R0", 0)),
                    _safe_float(node.get("杨氏模量E", 0)),
                    _safe_float(node.get("泊松比PR", 0)),
                    _safe_float(node.get("0.001应力应变曲线0", 0) or node.get("0.001应力应变曲线_0", 0)),
                    _safe_float(node.get("0.001应力应变曲线0.05", 0) or node.get("0.001应力应变曲线_0.05", 0)),
                    _safe_float(node.get("0.001应力应变曲线0.1", 0) or node.get("0.001应力应变曲线_0.1", 0)),
                    _safe_float(node.get("0.001应力应变曲线0.15", 0) or node.get("0.001应力应变曲线_0.15", 0)),
                    _safe_float(node.get("0.001应力应变曲线0.2", 0) or node.get("0.001应力应变曲线_0.2", 0)),
                    _safe_float(node.get("0.001应力应变曲线0.5", 0) or node.get("0.001应力应变曲线_0.5", 0)),
                    _safe_float(node.get("1应力应变曲线0", 0) or node.get("1应力应变曲线_0", 0)),
                    _safe_float(node.get("1应力应变曲线0.05", 0) or node.get("1应力应变曲线_0.05", 0)),
                    _safe_float(node.get("1应力应变曲线0.1", 0) or node.get("1应力应变曲线_0.1", 0)),
                    _safe_float(node.get("1应力应变曲线0.15", 0) or node.get("1应力应变曲线_0.15", 0)),
                    _safe_float(node.get("1应力应变曲线0.2", 0) or node.get("1应力应变曲线_0.2", 0)),
                    _safe_float(node.get("1应力应变曲线0.5", 0) or node.get("1应力应变曲线_0.5", 0)),
                ]
                mat_vec = normalize_material_vector(raw_mat_vec, normalization)
            else:
                stats["material_nodes"] += 1
                mid_raw = node.get("MID", node.get("MatName", None))
                try:
                    mid = int(float(mid_raw))
                except (TypeError, ValueError):
                    mid = None
                mat_vec = vehicle_materials.get(mid) if vehicle_materials is not None else None
                if mat_vec is None:
                    stats["unresolved_material_nodes"] += 1
                    if mid is not None:
                        stats["unresolved_material_ids"].add(mid)

            if mat_vec is not None:
                mat_sigs.add(tuple(round(float(v), 6) for v in mat_vec))

        if len(xyz_list) == 0:
            continue

        xyz = np.array(xyz_list, dtype=np.float32)
        pmin = xyz.min(axis=0)
        pmax = xyz.max(axis=0)
        span = pmax - pmin
        bbox = {
            "diag": float(np.sqrt(np.sum(span ** 2))),
            "x": float(span[0]),
            "y": float(span[1]),
            "z": float(span[2]),
        }

        samples.append({
            "file_id": file_id,
            "hic": hic_value,
            "age_code": age_code,
            "bbox": bbox,
            "xyz_min": pmin,
            "xyz_max": pmax,
            "mat_sigs": mat_sigs,
        })

    return samples, stats


def bbox_stats(points_xyz):
    """points_xyz: [N, 3] — 返回 bbox 对角线。"""
    if points_xyz is None or len(points_xyz) == 0:
        return {"diag": 0, "x": 0, "y": 0, "z": 0}
    pmin = points_xyz.min(axis=0)
    pmax = points_xyz.max(axis=0)
    span = pmax - pmin
    return {
        "diag": float(np.sqrt(np.sum(span ** 2))),
        "x": float(span[0]), "y": float(span[1]), "z": float(span[2]),
    }


# ---- main diagnostics ----
def run_diagnostics(car_files, old_normal_set, material_lookup, normalization):
    rows = []
    all_material_vecs = defaultdict(set)

    for car_dir, files in sorted(car_files.items()):
        vehicle = extract_vehicle_identifier(files[0]) if files else car_dir
        print(f"\n--- {vehicle} ({car_dir}): {len(files)} files ---")

        all_hics = []
        adult_count = 0
        child_count = 0
        hic_zero_count = 0
        all_bboxes_sample = []
        union_min = None
        union_max = None
        total_samples = 0
        raw_samples = 0
        hic_missing_count = 0
        material_nodes = 0
        unresolved_material_nodes = 0
        unresolved_material_ids = set()
        vehicle_materials = material_lookup_for_vehicle(vehicle, material_lookup)

        for fp in files:
            samples, file_stats = load_feather_samples(fp, vehicle_materials, normalization)
            raw_samples += file_stats["raw_samples"]
            hic_zero_count += file_stats["hic_zero_count"]
            hic_missing_count += file_stats["hic_missing_count"]
            material_nodes += file_stats["material_nodes"]
            unresolved_material_nodes += file_stats["unresolved_material_nodes"]
            unresolved_material_ids |= file_stats["unresolved_material_ids"]
            for s in samples:
                total_samples += 1
                all_hics.append(s["hic"])
                if s["age_code"] == 1:
                    adult_count += 1
                else:
                    child_count += 1
                all_bboxes_sample.append(s["bbox"])
                union_min = s["xyz_min"] if union_min is None else np.minimum(union_min, s["xyz_min"])
                union_max = s["xyz_max"] if union_max is None else np.maximum(union_max, s["xyz_max"])
                all_material_vecs[vehicle] |= s["mat_sigs"]

        hics = np.array(all_hics) if all_hics else np.array([0])
        if union_min is not None:
            union_span = union_max - union_min
            union_bbox = {
                "diag": float(np.sqrt(np.sum(union_span ** 2))),
                "x": float(union_span[0]),
                "y": float(union_span[1]),
                "z": float(union_span[2]),
            }
            mean_sample_bbox_diag = float(np.mean([b["diag"] for b in all_bboxes_sample]))
        else:
            union_bbox = {"diag": 0, "x": 0, "y": 0, "z": 0}
            mean_sample_bbox_diag = 0

        row = {
            "vehicle": vehicle,
            "car_dir": car_dir,
            "raw_samples": raw_samples,
            "total_samples": total_samples,
            "hic_zero_count": hic_zero_count,
            "hic_missing_count": hic_missing_count,
            "hic_mean": float(np.mean(hics)),
            "hic_max": float(np.max(hics)),
            "hic_min": float(np.min(hics)),
            "hic_gt2k_count": int(np.sum(hics > 2000)),
            "adult_count": adult_count,
            "child_count": child_count,
            "union_bbox_diag": union_bbox["diag"],
            "mean_sample_bbox_diag": mean_sample_bbox_diag,
            "unique_material_vectors": len(all_material_vecs[vehicle]),
            "material_nodes": material_nodes,
            "unresolved_material_nodes": unresolved_material_nodes,
            "unresolved_material_ids": sorted(unresolved_material_ids),
            "unresolved_material_id_count": len(unresolved_material_ids),
            "material_lookup_found": vehicle_materials is not None,
        }
        rows.append(row)
        print(f"  samples={total_samples}/{raw_samples}  hic_zero={hic_zero_count}  "
              f"hic_missing={hic_missing_count}  hic_mean={row['hic_mean']:.0f}  "
              f"hic_max={row['hic_max']:.0f}  hic>2k={row['hic_gt2k_count']}  "
              f"union_diag={union_bbox['diag']:.0f}mm  "
              f"mat_vecs={row['unique_material_vectors']}  "
              f"unresolved_mid={unresolved_material_nodes} nodes/"
              f"{len(unresolved_material_ids)} ids")

    # ---- cross-vehicle analysis ----
    # bbox scale vs old normal median
    normal_diags = [r["union_bbox_diag"] for r in rows
                    if r["vehicle"] in old_normal_set and r["union_bbox_diag"] > 0]
    if len(normal_diags) >= 3:
        old_normal_median_diag = float(np.median(normal_diags))
    else:
        all_diags = [r["union_bbox_diag"] for r in rows if r["union_bbox_diag"] > 0]
        old_normal_median_diag = float(np.median(all_diags)) if all_diags else 2800

    for r in rows:
        if r["union_bbox_diag"] > 0:
            r["bbox_ratio_vs_normal"] = round(r["union_bbox_diag"] / old_normal_median_diag, 3)
        else:
            r["bbox_ratio_vs_normal"] = float("nan")

    # material vector overlap
    old_normal_vec_union = set()
    for v in old_normal_set:
        old_normal_vec_union |= all_material_vecs.get(v, set())

    for r in rows:
        v = r["vehicle"]
        vecs = all_material_vecs.get(v, set())
        if len(vecs) == 0:
            r["mat_vec_overlap_vs_old"] = 1.0 if v in old_normal_set else 0.0
            r["mat_vec_only_count"] = 0
        else:
            overlap = len(vecs & old_normal_vec_union)
            r["mat_vec_overlap_vs_old"] = round(overlap / len(vecs), 3) if len(old_normal_vec_union) > 0 else 1.0
            r["mat_vec_only_count"] = len(vecs - old_normal_vec_union)
        r["is_old_normal"] = v in old_normal_set

    # Hard-only means absent from the frozen old-normal material union.
    hard_car_candidates = []
    for r2 in rows:
        if r2["vehicle"] in ("CY02C", "M6"):
            hard_car_candidates.append(r2["vehicle"])
        elif r2["mat_vec_overlap_vs_old"] < 0.80 and not r2["is_old_normal"]:
            hard_car_candidates.append(r2["vehicle"])

    hard_only_summary = {}
    for v in hard_car_candidates:
        vecs = all_material_vecs.get(v, set())
        only = vecs - old_normal_vec_union
        hard_only_summary[v] = {"count": len(only)}
        print(f"\n  Hard-only vectors ({v}): {len(only)} vectors not in old-normal union")

    return rows, hard_only_summary, old_normal_median_diag


def write_output(rows, hard_only, old_normal_median_diag, output_dir, old_normal_set):
    os.makedirs(output_dir, exist_ok=True)

    report = {
        "old_normal_vehicles": list(old_normal_set),
        "old_normal_median_union_diag_mm": old_normal_median_diag,
        "total_vehicles": len(rows),
        "total_samples": sum(r["total_samples"] for r in rows),
        "per_vehicle": rows,
        "hard_only_material_vectors": hard_only,
    }
    json_path = os.path.join(output_dir, "qa_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[JSON] {json_path}")

    csv_path = os.path.join(output_dir, "qa_summary.csv")
    csv_cols = [
        "vehicle", "car_dir", "raw_samples", "total_samples", "hic_zero_count",
        "hic_missing_count", "hic_mean",
        "hic_max", "hic_gt2k_count", "union_bbox_diag", "bbox_ratio_vs_normal",
        "unique_material_vectors", "mat_vec_overlap_vs_old", "mat_vec_only_count",
        "material_lookup_found", "material_nodes", "unresolved_material_nodes",
        "unresolved_material_id_count",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[CSV]  {csv_path}")

    # summary table
    print("\n" + "=" * 95)
    print("QA Summary")
    print("=" * 95)
    hdr = f"{'Vehicle':<14} {'Valid/Raw':>11} {'HIC0':>5} {'Miss':>5} {'HIC Mean':>9} {'HIC Max':>9} {'Diag':>7} {'MatVec':>7} {'Overlap':>8} {'UnresMID':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        valid_raw = f"{r['total_samples']}/{r['raw_samples']}"
        print(f"{r['vehicle']:<14} {valid_raw:>11} {r['hic_zero_count']:>5} "
              f"{r['hic_missing_count']:>5} {r['hic_mean']:>9.0f} "
              f"{r['hic_max']:>9.0f} {r['union_bbox_diag']:>7.0f} "
              f"{r['unique_material_vectors']:>7} {r['mat_vec_overlap_vs_old']:>8.2f} "
              f"{r['unresolved_material_nodes']:>9}")

    # flags
    print("\n--- FLAGS ---")
    warnings_count = 0
    critical_count = 0
    for r in rows:
        if r["hic_zero_count"] > 0:
            print(f"  [WARN] {r['vehicle']}: {r['hic_zero_count']} HIC=0 samples — exclude before training")
            warnings_count += 1
        if r["hic_missing_count"] > 0:
            print(f"  [WARN] {r['vehicle']}: {r['hic_missing_count']} samples have missing/invalid HIC")
            warnings_count += 1
        if r["total_samples"] < 30:
            print(f"  [WARN] {r['vehicle']}: only {r['total_samples']} samples (<30) — low-statistics fold")
            warnings_count += 1
        if not np.isnan(r["bbox_ratio_vs_normal"]) and (r["bbox_ratio_vs_normal"] < 0.8 or r["bbox_ratio_vs_normal"] > 1.2):
            print(f"  [INFO] {r['vehicle']}: bbox ratio {r['bbox_ratio_vs_normal']:.2f} outside [0.8, 1.2]")
            warnings_count += 1
        if not r["material_lookup_found"]:
            print(f"  [CRITICAL] {r['vehicle']}: vehicle missing from material lookup")
            critical_count += 1
        if r["unique_material_vectors"] == 0:
            print(f"  [CRITICAL] {r['vehicle']}: no material vectors resolved")
            critical_count += 1
        if r["unresolved_material_nodes"] > 0:
            preview = r["unresolved_material_ids"][:12]
            print(
                f"  [CRITICAL] {r['vehicle']}: {r['unresolved_material_nodes']} nodes across "
                f"{r['unresolved_material_id_count']} unresolved MIDs; examples={preview}"
            )
            critical_count += 1
        if r["mat_vec_overlap_vs_old"] < 0.80 and not r["is_old_normal"]:
            print(f"  [WARN] {r['vehicle']}: material vector overlap {r['mat_vec_overlap_vs_old']:.0%} < 80% "
                  f"— recommend material_dropout_prob=0.15")
            warnings_count += 1
    if warnings_count == 0 and critical_count == 0:
        print("  All checks passed.")
    print(f"\n  Warnings: {warnings_count}  Critical: {critical_count}")
    return critical_count


def main():
    args = parse_args()
    old_normal_set = set(v.strip() for v in args.old_normal_vehicles.split(","))
    material_lookup, normalization = load_material_assets(
        args.material_lookup_path,
        args.normalization_params_path,
    )

    print(f"Data dir: {args.data_dir}")
    print(f"Old normal vehicles: {old_normal_set}")

    car_files = collect_feather_files(args.data_dir)
    if not car_files:
        print("[ERROR] No .feather files found.")
        return 2

    vehicles = [extract_vehicle_identifier(files[0]) for files in car_files.values()]
    print(f"Found {len(car_files)} vehicles: {vehicles}")

    rows, hard_only, normal_median_diag = run_diagnostics(
        car_files,
        old_normal_set,
        material_lookup,
        normalization,
    )
    critical_count = write_output(
        rows,
        hard_only,
        normal_median_diag,
        args.output_dir,
        old_normal_set,
    )
    return 1 if critical_count else 0


if __name__ == "__main__":
    sys.exit(main())
