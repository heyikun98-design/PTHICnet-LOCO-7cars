import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser("eval_ablation")
    parser.add_argument("--results_root", type=str, default="experiments")
    parser.add_argument("--output_json", type=str, default="experiments/ablation_summary.json")
    parser.add_argument("--output_md", type=str, default="experiments/ablation_summary.md")
    return parser.parse_args()


def _is_pt_checkpoint(ckpt_path):
    """Detect checkpoint type by checking for PT-specific metadata keys."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return "model_hparams" in ckpt


def _detect_baseline_mode(ckpt_path):
    """Infer baseline ablation mode from state_dict layer names."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", {})
    for key in state.keys():
        if "cross_attn" in key.lower() or "crossattention" in key.lower():
            return "baseline"
    return "early_fusion_clean"


def collect_all_checkpoints(root):
    return sorted(Path(root).glob("**/checkpoints/best_model.pth"))


def run_cmd(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    checkpoints = collect_all_checkpoints(args.results_root)

    if not checkpoints:
        print("No checkpoints found. Did training complete?")
        return

    summary = {}
    for ckpt in checkpoints:
        run_name = ckpt.parents[1].name
        out_json = PROJECT_ROOT / "experiments" / f"eval_{run_name}.json"

        if _is_pt_checkpoint(ckpt):
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "eval_pt_hicnet.py"),
                "--checkpoint", str(ckpt),
                "--output", str(out_json.relative_to(PROJECT_ROOT)),
            ]
        else:
            mode = _detect_baseline_mode(ckpt)
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "eval_baseline.py"),
                "--checkpoint", str(ckpt),
                "--ablation_mode", mode,
                "--output", str(out_json.relative_to(PROJECT_ROOT)),
            ]
        run_cmd(cmd)

        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        summary.setdefault(run_name, []).append(data["overall"])

    merged = {}
    for run_name, values in summary.items():
        mse = np.array([v["mse"] for v in values], dtype=np.float64)
        acc = np.array([v["accuracy"] for v in values], dtype=np.float64)
        merged[run_name] = {
            "mse_mean": float(mse.mean()),
            "mse_std": float(mse.std(ddof=0)),
            "acc_mean": float(acc.mean()),
            "acc_std": float(acc.std(ddof=0)),
            "runs": len(values),
        }

    out_json_path = PROJECT_ROOT / args.output_json
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    lines = [
        "# Ablation Summary",
        "",
        "| Run | MSE(mean±std) | Accuracy(mean±std) | N |",
        "|---|---:|---:|---:|",
    ]
    for run_name in sorted(merged.keys()):
        row = merged[run_name]
        lines.append(
            f"| {run_name} | {row['mse_mean']:.6f} ± {row['mse_std']:.6f} | "
            f"{row['acc_mean']:.6f} ± {row['acc_std']:.6f} | {row['runs']} |"
        )
    lines.extend(
        [
            "",
            "- E0->E1: isolate early fusion.",
            "- E1->E2: isolate PT backbone.",
            "- E2->E3: isolate FiLM global.",
            "- E3->E4: compare FiLM deep variant.",
        ]
    )
    out_md_path = PROJECT_ROOT / args.output_md
    out_md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved summary: {out_json_path}")
    print(f"saved report: {out_md_path}")


if __name__ == "__main__":
    main()
