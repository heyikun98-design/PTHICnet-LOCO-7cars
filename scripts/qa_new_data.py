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
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

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
    return p.parse_args()


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


def extract_car_name(car_dir):
    """从 car 目录名解析车型代码。"""
    from data_utils.HICLoader_feather import CAR_TO_VEHICLE
    key = car_dir.lower()
    if key in CAR_TO_VEHICLE:
        return CAR_TO_VEHICLE[key]
    return car_dir


def load_feather_sample(fp):
    """加载单个 feather 文件，返回 (points, hic, age)。"""
    # 先尝试从路径获取 vehicle code，再导入 HICLoader
    try:
        from data_utils.HICLoader_feather import (
            HICDatasetFeather, HICLoaderFeather, extract_vehicle_identifier,
        )
    except ImportError:
        print("[WARN] Cannot import HICLoader — trying pyarrow directly")
        import pyarrow.feather as feather
        df = feather.read_feather(fp)
        pts_cols = ["x", "y", "z", "thickness"] + MAT_KEYS
        pts = df[pts_cols].values.astype(np.float32) if all(c in df.columns for c in pts_cols) else None
        hic_col = "hic" if "hic" in df.columns else ("HIC" if "HIC" in df.columns else None)
        hic = float(df[hic_col].iloc[0]) if hic_col else 0.0
        return pts, hic, None, df
    try:
        vehicle = extract_vehicle_identifier(fp)
        loader = HICLoaderFeather(
            [fp], vehicle, max_points=8192, train=True, use_normals=False,
            normalize_params_path=str(PROJECT_ROOT / "feather" / "normalization_params.pkl"),
            material_lookup_path=str(PROJECT_ROOT / "feather" / "material_lookup_by_vehicle.pkl"),
        )
        dataset = HICDatasetFeather([fp], vehicle, loader, is_train=False)
        if len(dataset) == 0:
            return None, 0.0, None, None
        item = dataset[0]
        pts = item.get("fused_input")  # [N,19] or [N,22]
        if pts is None:
            pts = item.get("points")
        hic = float(item.get("hic", item.get("target", 0)))
        age = item.get("age", item.get("age_group", None))
        return pts, hic, age, item
    except Exception as e:
        print(f"  [WARN] Failed to load {fp}: {e}")
        return None, 0.0, None, None


def bbox_stats(points_xyz):
    """points_xyz: [N, 3] — 返回 bbox 对角线和各轴跨度。"""
    if points_xyz is None or len(points_xyz) == 0:
        return {"diag": 0, "x": 0, "y": 0, "z": 0}
    pmin = points_xyz.min(axis=0)
    pmax = points_xyz.max(axis=0)
    span = pmax - pmin
    return {
        "diag": float(np.sqrt(np.sum(span ** 2))),
        "x": float(span[0]), "y": float(span[1]), "z": float(span[2]),
    }


def material_vec_signature(material_row, decimals=6):
    """将 15 维材料向量四舍五入为字符串签名。"""
    return tuple(round(float(v), decimals) for v in material_row)


# ---- 主诊断流程 ----

def run_diagnostics(car_files, old_normal_set):
    """对每辆车做 QA 诊断。"""
    rows = []
    old_normal_material_vecs = set()
    all_material_vecs = defaultdict(set)  # vehicle -> set of vec sigs

    for car_dir, files in sorted(car_files.items()):
        vehicle = extract_car_name(car_dir)
        print(f"\n--- {vehicle} ({car_dir}): {len(files)} files ---")

        hics = []
        ages = []
        bboxes_sample = []
        all_xyz = []
        total_samples = 0
        hic_zero_count = 0

        for fp in files:
            pts, hic, age, _ = load_feather_sample(fp)
            if pts is None:
                continue
            total_samples += 1
            if hic == 0:
                hic_zero_count += 1
            hics.append(hic)
            if age is not None:
                ages.append(age)

            # bbox
            xyz = pts[:, :3] if pts.shape[1] >= 3 else pts[:, :3]
            bbox = bbox_stats(xyz)
            bboxes_sample.append(bbox)
            all_xyz.append(xyz)

            # material vectors (unique per sample)
            if pts.shape[1] >= 19:
                mat_data = pts[:, 4:19]  # 15ch material after coords+thickness
                for i in range(len(mat_data)):
                    sig = material_vec_signature(mat_data[i])
                    all_material_vecs[vehicle].add(sig)

        # union bbox
        if all_xyz:
            union_xyz = np.concatenate(all_xyz, axis=0)
            union_bbox = bbox_stats(union_xyz)
            mean_sample_bbox = {
                "diag": float(np.mean([b["diag"] for b in bboxes_sample])),
                "x": float(np.mean([b["x"] for b in bboxes_sample])),
                "y": float(np.mean([b["y"] for b in bboxes_sample])),
                "z": float(np.mean([b["z"] for b in bboxes_sample])),
            }
        else:
            union_bbox = {"diag": 0, "x": 0, "y": 0, "z": 0}
            mean_sample_bbox = {"diag": 0, "x": 0, "y": 0, "z": 0}

        hics = np.array(hics)
        row = {
            "vehicle": vehicle,
            "car_dir": car_dir,
            "total_samples": total_samples,
            "hic_zero_count": hic_zero_count,
            "hic_mean": float(np.mean(hics)) if len(hics) else 0,
            "hic_max": float(np.max(hics)) if len(hics) else 0,
            "hic_min": float(np.min(hics)) if len(hics) else 0,
            "hic_gt2k_count": int(np.sum(hics > 2000)),
            "adult_count": sum(1 for a in ages if a == 1 or a == "Adult"),
            "child_count": sum(1 for a in ages if a == 0 or a == "Child"),
            "union_bbox_diag": union_bbox["diag"],
            "mean_sample_bbox_diag": mean_sample_bbox["diag"],
            "unique_material_vectors": len(all_material_vecs[vehicle]),
        }
        rows.append(row)
        print(f"  samples={total_samples}  hic_zero={hic_zero_count}  "
              f"hic_mean={row['hic_mean']:.0f}  hic_max={row['hic_max']:.0f}  "
              f"hic>2k={row['hic_gt2k_count']}  "
              f"union_diag={union_bbox['diag']:.0f}mm  "
              f"mat_vecs={row['unique_material_vectors']}")

    # ---- 跨车分析 ----
    # 1. bbox scale vs old normal mean
    normal_diags = [r["union_bbox_diag"] for r in rows if r["vehicle"] in old_normal_set and r["union_bbox_diag"] > 0]
    if len(normal_diags) >= 3:
        # 使用中位数（稳健）
        old_normal_median_diag = float(np.median(normal_diags))
    else:
        # fallback: 使用所有 normal 车（包括新 normal）的中位数
        all_normal_diags = [r["union_bbox_diag"] for r in rows if r["union_bbox_diag"] > 0]
        old_normal_median_diag = float(np.median(all_normal_diags)) if all_normal_diags else 2800

    for r in rows:
        if r["union_bbox_diag"] > 0:
            r["bbox_ratio_vs_normal"] = round(r["union_bbox_diag"] / old_normal_median_diag, 3)
        else:
            r["bbox_ratio_vs_normal"] = float("nan")

    # 2. material vector overlap vs old normal union
    old_normal_vec_union = set()
    for v in old_normal_set:
        if v in all_material_vecs:
            old_normal_vec_union |= all_material_vecs[v]

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

    # 3. hard-only material vectors (only in non-normal, not in any normal)
    all_normal_vec_union = set()
    for r2 in rows:
        all_normal_vec_union |= all_material_vecs.get(r2["vehicle"], set())
    # Identify hard cars: either old hard (CY02C, M6) or new cars with low overlap
    hard_car_candidates = []
    for r2 in rows:
        if r2["vehicle"] in ("CY02C", "M6"):
            hard_car_candidates.append(r2["vehicle"])
        elif r2["mat_vec_overlap_vs_old"] < 0.80 and not r2["is_old_normal"]:
            hard_car_candidates.append(r2["vehicle"])

    hard_only_summary = {}
    for v in hard_car_candidates:
        vecs = all_material_vecs.get(v, set())
        only = vecs - all_normal_vec_union
        hard_only_summary[v] = {"count": len(only)}
        print(f"\n  Hard-only vectors ({v}): {len(only)} vectors not in any normal car")

    return rows, hard_only_summary, old_normal_median_diag


def write_output(rows, hard_only, old_normal_median_diag, output_dir, old_normal_set):
    os.makedirs(output_dir, exist_ok=True)

    # JSON report
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

    # CSV summary
    csv_path = os.path.join(output_dir, "qa_summary.csv")
    csv_cols = [
        "vehicle", "car_dir", "total_samples", "hic_zero_count", "hic_mean",
        "hic_max", "hic_gt2k_count", "union_bbox_diag", "bbox_ratio_vs_normal",
        "unique_material_vectors", "mat_vec_overlap_vs_old", "mat_vec_only_count",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[CSV]  {csv_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("QA Summary")
    print("=" * 80)
    header = f"{'Vehicle':<14} {'Samples':>8} {'HIC=0':>6} {'HIC Mean':>9} {'HIC Max':>9} {'>2k':>5} {'Diag':>7} {'BboxRatio':>10} {'MatVec':>7} {'Overlap':>8} {'Only':>5}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['vehicle']:<14} {r['total_samples']:>8} {r['hic_zero_count']:>6} "
              f"{r['hic_mean']:>9.0f} {r['hic_max']:>9.0f} {r['hic_gt2k_count']:>5} "
              f"{r['union_bbox_diag']:>7.0f} {r['bbox_ratio_vs_normal']:>10.2f} "
              f"{r['unique_material_vectors']:>7} {r['mat_vec_overlap_vs_old']:>8.2f} "
              f"{r.get('mat_vec_only_count', 0):>5}")

    # Flag issues
    print("\n--- FLAGS ---")
    flags = 0
    for r in rows:
        if r["hic_zero_count"] > 0:
            print(f"  [WARN] {r['vehicle']}: {r['hic_zero_count']} HIC=0 samples — exclude before training")
            flags += 1
        if r["total_samples"] < 30:
            print(f"  [WARN] {r['vehicle']}: only {r['total_samples']} samples (<30) — low-statistics fold")
            flags += 1
        if r["bbox_ratio_vs_normal"] < 0.8 or r["bbox_ratio_vs_normal"] > 1.2:
            print(f"  [INFO] {r['vehicle']}: bbox ratio {r['bbox_ratio_vs_normal']:.2f} outside [0.8, 1.2]")
            flags += 1
        if r["mat_vec_overlap_vs_old"] < 0.80 and not r["is_old_normal"]:
            print(f"  [WARN] {r['vehicle']}: material vector overlap {r['mat_vec_overlap_vs_old']:.0%} < 80% "
                  f"— recommend material_dropout_prob=0.15")
            flags += 1
    if flags == 0:
        print("  All checks passed.")
    print(f"\n  Total flags: {flags}")


def main():
    args = parse_args()
    old_normal_set = set(v.strip() for v in args.old_normal_vehicles.split(","))

    print(f"Data dir: {args.data_dir}")
    print(f"Old normal vehicles: {old_normal_set}")

    # Collect
    car_files = collect_feather_files(args.data_dir)
    if not car_files:
        print("[ERROR] No .feather files found.")
        return

    vehicle_names = [extract_car_name(c) for c in car_files]
    print(f"Found {len(car_files)} vehicles: {vehicle_names}")

    # Diagnose
    rows, hard_only, normal_median_diag = run_diagnostics(car_files, old_normal_set)

    # Write
    write_output(rows, hard_only, normal_median_diag, args.output_dir, old_normal_set)


if __name__ == "__main__":
    main()
