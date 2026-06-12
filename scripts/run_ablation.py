import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser("run_ablation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--skip_train", action="store_true", default=False)
    return parser.parse_args()


def run_cmd(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seeds = cfg["experiment"]["seeds"]

    matrix = [
        {"name": "E0_baseline", "ablation_mode": "baseline", "use_early_fusion": False, "film_mode": "none"},
        {"name": "E1_early_fusion_clean", "ablation_mode": "early_fusion_clean", "use_early_fusion": True, "film_mode": "none"},
        {"name": "E2_pt_backbone", "ablation_mode": "pt_hicnet", "use_early_fusion": True, "film_mode": "none"},
        {"name": "E3_film_global", "ablation_mode": "pt_hicnet", "use_early_fusion": True, "film_mode": "global"},
        {"name": "E4_film_deep", "ablation_mode": "pt_hicnet", "use_early_fusion": True, "film_mode": "deep"},
    ]

    records = []
    for exp in matrix:
        for seed in seeds:
            run_id = f"{exp['name']}_seed{seed}"
            if not args.skip_train:
                if exp["ablation_mode"] == "pt_hicnet":
                    run_cmd(
                        [
                            sys.executable,
                            str(PROJECT_ROOT / "scripts" / "train_pt_hicnet.py"),
                            "--config",
                            args.config,
                            "--seed",
                            str(seed),
                            "--film_mode",
                            exp["film_mode"],
                        ]
                    )
                else:
                    cmd = [
                        sys.executable,
                        str(PROJECT_ROOT / "feather" / "train_reg_att_props_X70_feather.py"),
                        "--config",
                        args.config,
                        "--ablation_mode",
                        exp["ablation_mode"],
                        "--seed",
                        str(seed),
                    ]
                    if exp["use_early_fusion"]:
                        cmd.append("--use_early_fusion")
                    cmd.append("--normalize_thickness")
                    run_cmd(cmd)
            records.append({"run_id": run_id, **exp, "seed": seed})

    out = PROJECT_ROOT / "experiments" / "ablation_matrix.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"saved matrix: {out}")


if __name__ == "__main__":
    main()

