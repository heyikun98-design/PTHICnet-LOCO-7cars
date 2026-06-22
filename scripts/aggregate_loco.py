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
    parts = dirname.split("_")
    # Phase 3: look for "foldN"
    for i, p in enumerate(parts):
        if p.startswith("fold") and i + 1 < len(parts):
            return parts[i + 1]
    # Phase 5: after "e0"/"e1"/"e2"/"e3"/"e4" prefix
    for i, p in enumerate(parts):
        if p.lower() in ("e0", "e1", "e2", "e3", "e4") and i + 1 < len(parts):
            return parts[i + 1]
    return "???"


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
            results.append((vehicle, hist))
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


def aggregate_architecture(results_root, arch):
    """Aggregate all folds for one architecture."""
    pattern = ARCH_PATTERNS.get(arch)
    if pattern is None:
        print(f"[WARN] Unknown architecture: {arch}")
        return None

    hist_files = find_history(results_root, pattern)
    if not hist_files:
        print(f"[WARN] {arch}: no history.json found")
        return None

    per_vehicle = {}
    for vehicle, hist_path in hist_files:
        metrics = extract_metrics(json.loads(hist_path.read_text()))
        if metrics is None:
            print(f"[WARN] {arch} {vehicle}: empty history")
            continue
        per_vehicle[vehicle] = metrics

    if not per_vehicle:
        return None

    all_tests = [m["best_test"] for m in per_vehicle.values()]
    all_vals = [m["best_val"] for m in per_vehicle.values()]

    normal_tests = [per_vehicle[v]["best_test"] for v in NORMAL_CARS if v in per_vehicle]
    hard_tests = [per_vehicle[v]["best_test"] for v in HARD_CARS if v in per_vehicle]

    return {
        "arch": arch,
        "per_vehicle": per_vehicle,
        "mean_val": mean(all_vals),
        "std_val": stdev(all_vals) if len(all_vals) > 1 else 0,
        "mean_test": mean(all_tests),
        "std_test": stdev(all_tests) if len(all_tests) > 1 else 0,
        "normal_mean_test": mean(normal_tests) if normal_tests else 0,
        "normal_std_test": stdev(normal_tests) if len(normal_tests) > 1 else 0,
        "hard_mean_test": mean(hard_tests) if hard_tests else 0,
        "n_folds": len(per_vehicle),
    }


def print_per_arch_summary(agg_results):
    """Table 1: Per-Architecture LOCO Summary."""
    print("\n" + "=" * 95)
    print("TABLE 1: Per-Architecture LOCO-CV Summary (seed=42, best-checkpoint)")
    print("=" * 95)
    header = (f"{'Arch':<6} {'Folds':>5} {'Val Mean':>9} {'Val Std':>8} "
              f"{'Test Mean':>10} {'Test Std':>9} {'5-Normal':>9} {'2-Hard':>8}")
    sep = "-" * 95
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
        print(f"{'E2-E0':<6} {'(PT backbone)':>21} {d*100:>+9.2f}pp (5-normal)")
    if "E2" in by_arch and "E3" in by_arch:
        d = by_arch["E3"]["normal_mean_test"] - by_arch["E2"]["normal_mean_test"]
        print(f"{'E3-E2':<6} {'(FiLM global)':>21} {d*100:>+9.2f}pp (5-normal)")
    if "E2" in by_arch and "E4" in by_arch:
        d = by_arch["E4"]["normal_mean_test"] - by_arch["E2"]["normal_mean_test"]
        print(f"{'E4-E2':<6} {'(FiLM deep)':>21} {d*100:>+9.2f}pp (5-normal)")
    if "E0" in by_arch and "E1" in by_arch:
        d = by_arch["E1"]["normal_mean_test"] - by_arch["E0"]["normal_mean_test"]
        print(f"{'E1-E0':<6} {'(EF on PN++)':>21} {d*100:>+9.2f}pp (5-normal)")
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
    parser = argparse.ArgumentParser("aggregate_loco")
    parser.add_argument("--results_root", type=str, default="experiments",
                        help="Parent directory containing fold subdirectories")
    parser.add_argument("--architectures", type=str, nargs="*", default=None,
                        help="Architectures to aggregate (E0 E1 E2 E3 E4, or 'all'). "
                             "Omit for backward-compatible E3-only mode.")
    args = parser.parse_args()

    results_root = Path(args.results_root)

    # Backward compatibility: no --architectures → E3 only
    if args.architectures is None:
        arches = ["E3"]
    elif args.architectures == ["all"]:
        arches = ALL_ARCHITECTURES
    else:
        arches = args.architectures

    # Aggregate each architecture
    agg_results = []
    for arch in arches:
        result = aggregate_architecture(results_root, arch)
        if result:
            agg_results.append(result)
            print(f"[OK] {arch}: {result['n_folds']} folds, "
                  f"val={result['mean_val']*100:.2f}% test={result['mean_test']*100:.2f}%")

    if not agg_results:
        print("[ERROR] No results found. Check --results_root and --architectures.")
        sys.exit(1)

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
        print(f"{'5-normal':<10} {'':>8} {r['normal_mean_test']*100:>7.2f}% "
              f"±{r['normal_std_test']*100:.2f}pp")

    # Multi-architecture tables
    if len(agg_results) > 1:
        print_per_arch_summary(agg_results)
        print_per_vehicle_table(agg_results)
        print_paired_delta(agg_results, reference="E3")

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
