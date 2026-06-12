import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser("summarize_ablation")
    parser.add_argument("--results_dir", type=str, default="experiments")
    parser.add_argument("--output", type=str, default="experiments/ablation_report.md")
    return parser.parse_args()


def collect_eval_files(root):
    root_path = Path(root)
    return sorted(root_path.rglob("eval_*.json"))


def main():
    args = parse_args()
    eval_files = collect_eval_files(args.results_dir)
    grouped = {}
    for fp in eval_files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        name = fp.stem.replace("eval_", "")
        grouped.setdefault(name, {"mse": [], "acc": []})
        grouped[name]["mse"].append(data["overall"]["mse"])
        grouped[name]["acc"].append(data["overall"]["accuracy"])

    lines = [
        "# PT-HICnet Ablation Report",
        "",
        "## Overall",
        "",
        "| Experiment | MSE (mean±std) | Accuracy (mean±std) |",
        "|---|---:|---:|",
    ]
    for name in sorted(grouped.keys()):
        mse = np.array(grouped[name]["mse"], dtype=np.float64)
        acc = np.array(grouped[name]["acc"], dtype=np.float64)
        lines.append(
            f"| {name} | {mse.mean():.6f} ± {mse.std(ddof=0):.6f} | {acc.mean():.6f} ± {acc.std(ddof=0):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Conclusions",
            "",
            "- Compare E0->E1 to isolate early-fusion contribution.",
            "- Compare E1->E2 to isolate PT backbone contribution.",
            "- Compare E2->E3 to isolate FiLM contribution.",
            "- Mark gain as non-significant when relative improvement < 1% and std overlaps.",
        ]
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved report: {out}")


if __name__ == "__main__":
    main()

