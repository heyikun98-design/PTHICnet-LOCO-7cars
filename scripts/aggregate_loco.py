#!/usr/bin/env python
"""Aggregate LOCO-CV results across folds and architectures.

Reads history.json from each fold, extracts best-checkpoint metrics and
produces multi-perspective summaries.

Usage:
  python scripts/aggregate_loco.py                          # E3 only (backward compat)
  python scripts/aggregate_loco.py --architectures E0 E2 E3 # specified arches
  python scripts/aggregate_loco.py --architectures all      # all 5 arches
"""

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---- Vehicle grouping (update for Phase 5 expanded data) ----
# These lists are the Phase 3 frozen baseline. For Phase 5, edit below
# or pass via CLI (future work).
VEHICLES = ["C201", "EP32", "JX65", "CY02C", "M6", "S50EVK", "FX11"]
NORMAL_CARS = ["C201", "EP32", "JX65", "S50EVK", "FX11"]
HARD_CARS = ["CY02C", "M6"]

# Directory naming patterns per architecture.
# Phase 3 E3 naming: pt_hicnet_loco_fold3_CY02C_seed42_film-global
# Phase 5 naming:   pt_hicnet_loco_e3_C201_seed42_film-global_md0.15
# The E3 list catches both legacy Phase 3 and Phase 5 names. Ablation prefixes
# are kept out of main tables by using a separate --results_root.
ARCH_PATTERNS = {
    "E0": "pt_hicnet_loco_e0_*",
    "E1": "pt_hicnet_loco_e1_*",
    "E2": "pt_hicnet_loco_e2_*",
    "E3": ["pt_hicnet_loco_e3_*", "pt_hicnet_loco_fold*"],
    "E4": "pt_hicnet_loco_e4_*",
}

ALL_ARCHITECTURES = ["E0", "E1", "E2", "E3", "E4"]
EXCLUDED_MAIN_PREFIXES = (
    "pt_hicnet_loco_e3_matdrop",
)


def _extract_vehicle(dirname):
    """Extract vehicle code from LOCO directory name.

    Handles both:
      Phase 3:  pt_hicnet_loco_fold3_CY02C_seed42_film-global
                → vehicle after "fold{N}"
      Phase 5:  pt_hicnet_loco_e3_C201_seed42_film-global_md0.15
                → vehicle is token after arch prefix (e3/e2/e0)
    """
    phase3 = re.search(r"_fold\d+_(.+?)_seed\d+(?:_|$)", dirname)
    if phase3:
        return phase3.group(1)
    phase5 = re.search(r"_e[0-4]_(.+?)_seed\d+(?:_|$)", dirname)
    if phase5:
        return phase5.group(1)
    return "???"


def _extract_seed(dirname):
    """Extract training seed from directory name.

    Both Phase 3 and Phase 5 naming include seed{N}:
      pt_hicnet_loco_e3_fold3_CY02C_seed42_film-global
      pt_hicnet_loco_e3_C201_seed42_film-global_md0.15
    """
    match = re.search(r"(?:^|_)seed(\d+)(?:_|$)", dirname)
    if match:
        return int(match.group(1))
    return 0


def find_history(results_root, patterns):
    """Locate history.json files matching one or more glob patterns."""
    if isinstance(patterns, str):
        patterns = [patterns]
    candidates = []
    seen = set()
    for pattern in patterns:
        for candidate in sorted(Path(results_root).glob(pattern)):
            if any(candidate.name.startswith(prefix) for prefix in EXCLUDED_MAIN_PREFIXES):
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)
    results = []
    for candidate in candidates:
        hist = candidate / "history.json"
        if hist.exists():
            vehicle = _extract_vehicle(candidate.name)
            seed = _extract_seed(candidate.name)
            results.append((vehicle, seed, hist))
    return results


def extract_metrics(history):
    """Extract best-checkpoint metrics from history."""
    epochs = [e for e in history if isinstance(e.get("epoch"), int)]
    if not epochs:
        return None

    has_val = "val_accuracy" in epochs[0]
    select_key = "val_accuracy" if has_val else "test_accuracy"
    best = max(epochs, key=lambda e: e.get(select_key, 0))

    best_val = best.get("val_accuracy", best.get("test_accuracy", 0))
    best_test = best.get("test_accuracy", best.get("val_accuracy", 0))
    best_mse = best.get("test_mse", best.get("val_mse", float("inf")))

    return {
        "best_val": best_val,
        "best_test": best_test,
        "best_mse": best_mse,
        "gap": best_val - best_test,
        "best_epoch": best["epoch"],
        "n_epochs": len(epochs),
    }


def aggregate_architecture(results_root, arch, expected_seeds=None, expected_vehicles=None):
    """Aggregate all folds for one architecture, with multi-seed support.

    Returns a dict with:
      - per_vehicle: {vehicle: mean metrics across seeds}
      - per_vehicle_seed: {vehicle: {seed: metrics}}
      - seeds: set of seeds found
      - n_folds: unique (vehicle, seed) pairs
    """
    pattern = ARCH_PATTERNS.get(arch)
    if pattern is None:
        print(f"[WARN] Unknown architecture: {arch}")
        return None

    hist_files = find_history(results_root, pattern)
    if not hist_files:
        print(f"[WARN] {arch}: no history.json found")
        return None

    # per_vehicle_seed[vehicle][seed] = metrics
    per_vehicle_seed = {}
    all_seeds = set()
    duplicates = []
    for vehicle, seed, hist_path in hist_files:
        metrics = extract_metrics(json.loads(hist_path.read_text()))
        if metrics is None:
            print(f"[WARN] {arch} {vehicle} seed{seed}: empty history")
            continue
        all_seeds.add(seed)
        if vehicle not in per_vehicle_seed:
            per_vehicle_seed[vehicle] = {}
        if seed in per_vehicle_seed[vehicle]:
            duplicates.append((vehicle, seed, str(hist_path)))
            print(f"[ERROR] {arch}: duplicate result for {vehicle} seed{seed}: {hist_path}")
            continue
        per_vehicle_seed[vehicle][seed] = metrics

    if not per_vehicle_seed:
        return None

    # Compute per-vehicle mean ± std across seeds
    per_vehicle_mean = {}
    per_vehicle_std = {}
    for v, seed_dict in per_vehicle_seed.items():
        tests = [m["best_test"] for m in seed_dict.values()]
        vals = [m["best_val"] for m in seed_dict.values()]
        per_vehicle_mean[v] = {
            "best_test": mean(tests),
            "best_val": mean(vals),
            "best_mse": mean(m["best_mse"] for m in seed_dict.values()),
            "gap": mean(vals) - mean(tests),
            "best_epoch": int(mean(m["best_epoch"] for m in seed_dict.values())),
        }
        per_vehicle_std[v] = {
            "best_test": stdev(tests) if len(tests) > 1 else 0,
            "best_val": stdev(vals) if len(vals) > 1 else 0,
        }

    all_tests = [per_vehicle_mean[v]["best_test"] for v in per_vehicle_mean]
    all_vals = [per_vehicle_mean[v]["best_val"] for v in per_vehicle_mean]
    normal_tests = [per_vehicle_mean[v]["best_test"] for v in NORMAL_CARS if v in per_vehicle_mean]
    hard_tests = [per_vehicle_mean[v]["best_test"] for v in HARD_CARS if v in per_vehicle_mean]

    per_seed = {}
    for seed in sorted(all_seeds):
        seed_metrics = {
            vehicle: seed_dict[seed]
            for vehicle, seed_dict in per_vehicle_seed.items()
            if seed in seed_dict
        }
        tests = [metrics["best_test"] for metrics in seed_metrics.values()]
        vals = [metrics["best_val"] for metrics in seed_metrics.values()]
        normal = [seed_metrics[v]["best_test"] for v in NORMAL_CARS if v in seed_metrics]
        hard = [seed_metrics[v]["best_test"] for v in HARD_CARS if v in seed_metrics]
        per_seed[seed] = {
            "n_vehicles": len(seed_metrics),
            "mean_val": mean(vals),
            "mean_test": mean(tests),
            "normal_mean_test": mean(normal) if normal else 0,
            "hard_mean_test": mean(hard) if hard else 0,
        }

    required_seeds = set(expected_seeds or all_seeds)
    required_vehicles = list(expected_vehicles or per_vehicle_seed.keys())
    missing = []
    for vehicle in required_vehicles:
        present = set(per_vehicle_seed.get(vehicle, {}))
        for seed in sorted(required_seeds - present):
            missing.append((vehicle, seed))
            print(f"[ERROR] {arch}: missing {vehicle} seed{seed}")

    return {
        "arch": arch,
        "per_vehicle": per_vehicle_mean,          # backward-compat: vehicle → mean metrics
        "per_vehicle_seed": per_vehicle_seed,     # full: vehicle → {seed → metrics}
        "per_vehicle_std": per_vehicle_std,       # vehicle → std across seeds
        "seeds": sorted(all_seeds),
        "n_seeds": len(all_seeds),
        "per_seed": per_seed,
        "missing": missing,
        "duplicates": duplicates,
        "coverage_complete": not missing and not duplicates,
        "mean_val": mean(all_vals),
        "std_val": stdev(all_vals) if len(all_vals) > 1 else 0,
        "mean_test": mean(all_tests),
        "std_test": stdev(all_tests) if len(all_tests) > 1 else 0,
        "normal_mean_test": mean(normal_tests) if normal_tests else 0,
        "normal_std_test": stdev(normal_tests) if len(normal_tests) > 1 else 0,
        "hard_mean_test": mean(hard_tests) if hard_tests else 0,
        "n_folds": sum(len(sd) for sd in per_vehicle_seed.values()),
    }


def print_per_seed_summary(agg_results):
    """Architecture-level fleet metrics for each independent seed."""
    if not any(r.get("n_seeds", 1) > 1 for r in agg_results):
        return
    print("=" * 82)
    print("TABLE 4: Per-Seed Fleet Summary")
    print("=" * 82)
    print(f"{'Arch':<6} {'Seed':>6} {'Cars':>6} {'Val':>9} {'Test':>9} {'Normal':>9} {'Hard':>9}")
    print("-" * 82)
    for result in agg_results:
        for seed, metrics in sorted(result.get("per_seed", {}).items()):
            print(
                f"{result['arch']:<6} {seed:>6} {metrics['n_vehicles']:>6} "
                f"{metrics['mean_val']*100:>8.2f}% {metrics['mean_test']*100:>8.2f}% "
                f"{metrics['normal_mean_test']*100:>8.2f}% {metrics['hard_mean_test']*100:>8.2f}%"
            )
    print()


def print_seed_paired_deltas(agg_results):
    """Paired architecture deltas computed within each seed."""
    by_arch = {result["arch"]: result for result in agg_results}
    comparisons = [
        ("E2", "E0", "PT backbone"),
        ("E3", "E2", "FiLM global"),
        ("E4", "E2", "FiLM deep"),
    ]
    rows = []
    for lhs, rhs, label in comparisons:
        if lhs not in by_arch or rhs not in by_arch:
            continue
        common = sorted(set(by_arch[lhs]["per_seed"]) & set(by_arch[rhs]["per_seed"]))
        deltas = []
        for seed in common:
            delta = (
                by_arch[lhs]["per_seed"][seed]["normal_mean_test"]
                - by_arch[rhs]["per_seed"][seed]["normal_mean_test"]
            )
            deltas.append((seed, delta))
        if deltas:
            rows.append((lhs, rhs, label, deltas))
    if not rows:
        return
    print("=" * 82)
    print("TABLE 5: Seed-Paired Architecture Deltas (Normal Cars)")
    print("=" * 82)
    for lhs, rhs, label, deltas in rows:
        values = [delta for _, delta in deltas]
        spread = stdev(values) if len(values) > 1 else 0
        detail = ", ".join(f"seed{seed}={delta*100:+.2f}pp" for seed, delta in deltas)
        print(f"{lhs}-{rhs} ({label}): {mean(values)*100:+.2f} ± {spread*100:.2f}pp")
        print(f"  {detail}")
    print()


def print_per_arch_summary(agg_results):
    """Table 1: Per-Architecture LOCO Summary (multi-seed aware)."""
    max_seeds = max((r.get("n_seeds", 1) for r in agg_results), default=1)
    seed_str = f", {max_seeds} seeds" if max_seeds > 1 else ", seed=42"
    print("\n" + "=" * 100)
    print(f"TABLE 1: Per-Architecture LOCO-CV Summary (best-checkpoint{seed_str})")
    print("=" * 100)
    header = (f"{'Arch':<6} {'Folds':>5} {'Val Mean':>9} {'Val Std':>8} "
              f"{'Test Mean':>10} {'Test Std':>9} {'Normal':>9} {'Hard':>8}")
    sep = "-" * 100
    print(header)
    print(sep)
    for r in agg_results:
        print(f"{r['arch']:<6} {r['n_folds']:>5} {r['mean_val']*100:>8.2f}% {r['std_val']*100:>8.2f}pp "
              f"{r['mean_test']*100:>9.2f}% {r['std_test']*100:>8.2f}pp "
              f"{r['normal_mean_test']*100:>8.2f}% {r['hard_mean_test']*100:>7.2f}%")
    print(sep)

    # Delta row: E2-E0 (PT backbone), E3-E2 (FiLM global), E4-E2 (FiLM deep)
    by_arch = {r["arch"]: r for r in agg_results}
    if "E0" in by_arch and "E2" in by_arch:
        d = by_arch["E2"]["normal_mean_test"] - by_arch["E0"]["normal_mean_test"]
        print(f"{'E2-E0':<6} {'(PT backbone)':>21} {d*100:>+9.2f}pp (normal group)")
    if "E2" in by_arch and "E3" in by_arch:
        d = by_arch["E3"]["normal_mean_test"] - by_arch["E2"]["normal_mean_test"]
        print(f"{'E3-E2':<6} {'(FiLM global)':>21} {d*100:>+9.2f}pp (normal group)")
    if "E2" in by_arch and "E4" in by_arch:
        d = by_arch["E4"]["normal_mean_test"] - by_arch["E2"]["normal_mean_test"]
        print(f"{'E4-E2':<6} {'(FiLM deep)':>21} {d*100:>+9.2f}pp (normal group)")
    if "E0" in by_arch and "E1" in by_arch:
        d = by_arch["E1"]["normal_mean_test"] - by_arch["E0"]["normal_mean_test"]
        print(f"{'E1-E0':<6} {'(EF on PN++)':>21} {d*100:>+9.2f}pp (normal group)")
    print()


def print_per_vehicle_table(agg_results):
    """Table 2: Per-Vehicle Comparison across all architectures."""
    arches = [r["arch"] for r in agg_results]
    print("=" * (50 + len(arches) * 9))
    print("TABLE 2: Per-Vehicle Test Accuracy (best-checkpoint)")
    print("=" * (50 + len(arches) * 9))

    # Header
    line = f"{'Vehicle':<10}"
    for a in arches:
        line += f" {a:>8}"
    line += f" {'Winner':>8}"
    for a in arches[1:]:
        line += f" {a+'-E0':>8}"
    print(line)
    print("-" * (50 + len(arches) * 9))

    for v in VEHICLES:
        scores = {}
        for r in agg_results:
            if v in r["per_vehicle"]:
                scores[r["arch"]] = r["per_vehicle"][v]["best_test"]
        if not scores:
            continue

        line = f"{v:<10}"
        for a in arches:
            if a in scores:
                line += f" {scores[a]*100:>7.2f}%"

        # Winner
        winner = max(scores, key=scores.get)
        line += f" {winner:>7}"

        # Deltas vs E0
        if "E0" in scores:
            for a in arches:
                if a != "E0" and a in scores:
                    d = scores[a] - scores["E0"]
                    line += f" {d*100:>+7.1f}pp"
        print(line)
    print()


def print_multi_seed_per_vehicle(agg_results):
    """Table 3b: Per-Vehicle mean ± std across seeds (only when multi-seed)."""
    multi = [r for r in agg_results if r.get("n_seeds", 1) > 1]
    if not multi:
        return
    print("=" * 100)
    print("TABLE 3b: Per-Vehicle Test Accuracy (mean ± std across seeds)")
    print("=" * 100)
    header = f"{'Vehicle':<10}"
    for r in multi:
        header += f" {r['arch']+' Test':>16}"
    print(header)
    print("-" * 100)
    for v in VEHICLES:
        line = f"{v:<10}"
        has_any = False
        for r in multi:
            if v in r.get("per_vehicle_std", {}) and v in r.get("per_vehicle_seed", {}):
                mean_v = r["per_vehicle"][v]["best_test"] * 100
                std_v = r["per_vehicle_std"][v]["best_test"] * 100
                n_s = len(r["per_vehicle_seed"][v])
                line += f" {mean_v:>7.2f}% ±{std_v:>4.2f}pp ({n_s}s)"
                has_any = True
            elif v in r["per_vehicle"]:
                line += f" {r['per_vehicle'][v]['best_test']*100:>15.2f}%"
                has_any = True
            else:
                line += f" {'—':>16}"
        if has_any:
            print(line)
    print()


def print_paired_delta(agg_results, reference="E3"):
    """Table 3: Paired per-vehicle delta vs reference architecture."""
    ref_data = None
    for r in agg_results:
        if r["arch"] == reference:
            ref_data = r
            break
    if ref_data is None:
        return

    other_arches = [r for r in agg_results if r["arch"] != reference]
    if not other_arches:
        return

    print("=" * (50 + len(other_arches) * 10))
    print(f"TABLE 3: Paired Delta vs {reference} (per-vehicle Test Acc)")
    print("=" * (50 + len(other_arches) * 10))

    header = f"{'Vehicle':<10} {reference:>8}"
    for r in other_arches:
        header += f" {r['arch']:>8} {r['arch']+'-'+reference:>9}"
    print(header)
    print("-" * (50 + len(other_arches) * 10))

    for v in VEHICLES:
        if v not in ref_data["per_vehicle"]:
            continue
        ref_score = ref_data["per_vehicle"][v]["best_test"]
        line = f"{v:<10} {ref_score*100:>7.2f}%"
        for r in other_arches:
            if v in r["per_vehicle"]:
                other_score = r["per_vehicle"][v]["best_test"]
                delta = other_score - ref_score
                line += f" {other_score*100:>7.2f}% {delta*100:>+8.1f}pp"
            else:
                line += f" {'—':>7} {'—':>9}"
        print(line)
    print()


def main():
    global VEHICLES, NORMAL_CARS, HARD_CARS
    parser = argparse.ArgumentParser("aggregate_loco")
    parser.add_argument("--results_root", type=str, default="experiments",
                        help="Parent directory containing fold subdirectories")
    parser.add_argument("--architectures", type=str, nargs="*", default=None,
                        help="Architectures to aggregate (E0 E1 E2 E3 E4, or 'all'). "
                             "Omit for backward-compatible E3-only mode.")
    parser.add_argument("--vehicles", type=str, nargs="+", default=None,
                        help="Expected vehicle list; defaults to the frozen Phase 3 list.")
    parser.add_argument("--normal_cars", type=str, nargs="+", default=None)
    parser.add_argument("--hard_cars", type=str, nargs="+", default=None)
    parser.add_argument("--expected_seeds", type=int, nargs="+", default=None,
                        help="Required seed list, e.g. 42 3407 2026.")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero when any vehicle/seed result is missing or duplicated.")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if args.vehicles:
        VEHICLES = args.vehicles
    if args.normal_cars:
        NORMAL_CARS = args.normal_cars
    if args.hard_cars:
        HARD_CARS = args.hard_cars

    grouping_errors = []
    overlap = sorted(set(NORMAL_CARS) & set(HARD_CARS))
    ungrouped = sorted(set(VEHICLES) - set(NORMAL_CARS) - set(HARD_CARS))
    unknown_grouped = sorted((set(NORMAL_CARS) | set(HARD_CARS)) - set(VEHICLES))
    if overlap:
        grouping_errors.append(f"vehicles in both normal and hard groups: {overlap}")
    if ungrouped:
        grouping_errors.append(f"vehicles missing normal/hard assignment: {ungrouped}")
    if unknown_grouped:
        grouping_errors.append(f"grouped vehicles absent from --vehicles: {unknown_grouped}")
    for error in grouping_errors:
        print(f"[ERROR] Grouping: {error}")

    # Backward compatibility: no --architectures → E3 only
    if args.architectures is None:
        arches = ["E3"]
    elif args.architectures == ["all"]:
        arches = ALL_ARCHITECTURES
    else:
        arches = args.architectures

    # Aggregate each architecture
    agg_results = []
    missing_architectures = []
    for arch in arches:
        result = aggregate_architecture(
            results_root,
            arch,
            expected_seeds=args.expected_seeds,
            expected_vehicles=VEHICLES if args.expected_seeds else None,
        )
        if result:
            agg_results.append(result)
            print(f"[OK] {arch}: {result['n_folds']} folds, "
                  f"val={result['mean_val']*100:.2f}% test={result['mean_test']*100:.2f}%")
        else:
            missing_architectures.append(arch)

    if not agg_results:
        print("[ERROR] No results found. Check --results_root and --architectures.")
        sys.exit(1)

    incomplete = [result["arch"] for result in agg_results if not result["coverage_complete"]]
    if missing_architectures:
        print(f"[COVERAGE] Missing architectures: {', '.join(missing_architectures)}")
    if incomplete:
        print(f"[COVERAGE] Incomplete architectures: {', '.join(incomplete)}")
    if args.strict and (grouping_errors or missing_architectures or incomplete):
        print("[ERROR] Strict coverage check failed.")
        sys.exit(2)

    # Print per-arch details (backward-compatible format for single arch)
    if len(agg_results) == 1:
        r = agg_results[0]
        print(f"\n{'Vehicle':<10} {'Val':>8} {'Test':>8} {'Gap':>8} {'Best Ep':>8}")
        print("-" * 48)
        for v in VEHICLES:
            if v in r["per_vehicle"]:
                m = r["per_vehicle"][v]
                print(f"{v:<10} {m['best_val']*100:>7.2f}% {m['best_test']*100:>7.2f}% "
                      f"{m['gap']*100:>+7.1f}pp {m['best_epoch']:>7}")
        print("-" * 48)
        print(f"{'Mean':<10} {r['mean_val']*100:>7.2f}% {r['mean_test']*100:>7.2f}% "
              f"±{r['std_test']*100:.2f}pp")
        print(f"{'Normal':<10} {'':>8} {r['normal_mean_test']*100:>7.2f}% "
              f"±{r['normal_std_test']*100:.2f}pp")

    # Multi-architecture tables
    if len(agg_results) > 1:
        print_per_arch_summary(agg_results)
        print_per_vehicle_table(agg_results)
        print_paired_delta(agg_results, reference="E3")
        print_multi_seed_per_vehicle(agg_results)
        print_per_seed_summary(agg_results)
        print_seed_paired_deltas(agg_results)

    # Multi-seed stats for single-arch mode
    if len(agg_results) == 1 and agg_results[0].get("n_seeds", 1) > 1:
        r = agg_results[0]
        print(f"\n--- Multi-Seed Summary ({r['arch']}, {r['n_seeds']} seeds) ---")
        for v in VEHICLES:
            if v in r.get("per_vehicle_seed", {}):
                sd = r["per_vehicle_seed"][v]
                tests = [m["best_test"] * 100 for m in sd.values()]
                print(f"  {v:<10} seeds={sorted(sd.keys())}  test: {mean(tests):.2f}% ±{stdev(tests):.2f}pp" if len(tests) > 1 else f"  {v:<10} seed={sorted(sd.keys())[0]}  test: {tests[0]:.2f}%")
        print_per_seed_summary(agg_results)

    # JX65 consistency check (single arch E3 or when E3 is included)
    e3_result = next((r for r in agg_results if r["arch"] == "E3"), None)
    if e3_result and "JX65" in e3_result["per_vehicle"]:
        jx = e3_result["per_vehicle"]["JX65"]
        ref_best = 83.88
        diff = abs(ref_best - jx["best_test"] * 100)
        if diff <= 3.0:
            band = "OK"
        elif diff <= 5.0:
            band = "WARN"
        else:
            band = "INVESTIGATE"
        print(f"[JX65 Soft Check] LOCO: {jx['best_test']*100:.2f}% vs E3-ES ref: 83.88% "
              f"(Δ={diff:.2f}pp) → {band}")
        print(f"  Note: E3-ES = single-car upper bound. LOCO = cross-car generalization.")


if __name__ == "__main__":
    main()
