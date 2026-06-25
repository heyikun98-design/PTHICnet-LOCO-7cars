import argparse
import copy
import hashlib
import importlib
import json
import os
import random
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
from collections import defaultdict
from torch.utils.data import ConcatDataset, DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "feather"))
sys.path.insert(0, str(PROJECT_ROOT / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "feather" / "data_utils"))

import provider  # noqa: E402

try:
    import wandb
except ImportError:
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser("train_pt_hicnet")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--film_mode", type=str, default=None, choices=["none", "global", "deep"])
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience (0=disable)")
    parser.add_argument("--restore_best", action="store_true", default=None, help="Restore best acc weights on early stop")
    parser.add_argument("--adult_hic_weight", type=float, default=None, help="Loss weight multiplier for adult high-HIC samples (0=disabled)")
    parser.add_argument("--adult_hic_threshold", type=float, default=None, help="HIC threshold for adult weighting (default 1000)")
    parser.add_argument("--material_dropout_prob", type=float, default=None, help="Drop probability for material z-score channels 4:19 (0=disabled)")
    parser.add_argument("--material_jitter_std", type=float, default=None, help="Gaussian jitter std for material z-score channels 4:19 (0=disabled)")
    parser.add_argument("--test_vehicles", type=str, nargs="+", default=None, help="Override test vehicles (e.g., --test_vehicles C201 EP32)")
    parser.add_argument("--val_split", type=float, default=0.0, help="Fraction of training data to use for validation (0=skip)")
    parser.add_argument("--split_seed", type=int, default=2026, help="Seed for train/val split (independent of training seed)")
    parser.add_argument("--exp_name", type=str, default=None, help="Override experiment name")
    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument("--use_wandb", dest="use_wandb", action="store_true")
    wandb_group.add_argument("--no_wandb", dest="use_wandb", action="store_false")
    parser.set_defaults(use_wandb=None)
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_datasets_eval_deterministic(loader, flag):
    """Toggle eval_deterministic on all underlying HICDataLoader datasets."""
    def visit(dataset):
        if hasattr(dataset, "eval_deterministic"):
            dataset.eval_deterministic = flag
        if hasattr(dataset, "datasets"):
            for child in dataset.datasets:
                visit(child)
        if hasattr(dataset, "dataset"):
            visit(dataset.dataset)

    visit(loader.dataset)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    if not files:
        raise FileNotFoundError(f"no .json/.feather in {root_dir}")
    return sorted(files)


def split_train_test(all_files, test_vehicles):
    """根据车名映射将文件分为训练集和测试集。"""
    from data_utils.HICLoader_feather import CAR_TO_VEHICLE

    # 构建反向映射：vehicle_code → car_dir
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


def build_loader(files, args_ns, batch_size, num_workers, early_fusion, normalize_thickness, shuffle,
                 eval_deterministic=False):
    from data_utils.HICLoader_feather import HICDataLoader

    datasets = [
        HICDataLoader(
            root=fp,
            args=args_ns,
            early_fusion=early_fusion,
            normalize_thickness=normalize_thickness,
            eval_deterministic=eval_deterministic,
        )
        for fp in files
    ]
    return DataLoader(
        ConcatDataset(datasets),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=shuffle,
    )


def run_smoke_test(
    project_root,
    data_cfg,
    model_cfg,
    loss_cfg,
    film_mode,
    resolved_delta,
    delta_source,
    runtime_args,
    HICDataLoader,
    PT_HICnet,
    KDEWeightedHuberLoss,
):
    data_root = data_cfg.get("data_root", "")
    search_dir = os.path.join(data_root, data_cfg["train_data_dir"]) if data_cfg["train_data_dir"] else data_root
    all_files = collect_data_files(search_dir)
    feather_files = [fp for fp in all_files if fp.endswith(".feather")]
    if not feather_files:
        raise RuntimeError(f"[SmokeTest] no .feather file found under {search_dir}")

    ds = HICDataLoader(
        root=feather_files[0],
        args=runtime_args,
        early_fusion=True,
        normalize_thickness=bool(data_cfg.get("normalize_thickness", True)),
    )
    if len(ds) == 0:
        raise RuntimeError(f"[SmokeTest] empty dataset from {feather_files[0]}")

    fused_input, _, _, age_group, _ = ds[0]
    expected_channels = 22 if data_cfg.get("use_normals", False) else 19
    expected_shape = (int(data_cfg["num_point"]), expected_channels)
    if tuple(fused_input.shape) != expected_shape:
        raise RuntimeError(
            f"[SmokeTest] fused_input shape mismatch: got={tuple(fused_input.shape)} expected={expected_shape}"
        )
    if int(age_group) not in {0, 1}:
        raise RuntimeError(f"[SmokeTest] invalid age_group={age_group}, expected one of {{0, 1}}")

    kde_csv = loss_cfg.get("kde_reference_csv")
    if kde_csv:
        kde_csv = str(project_root / kde_csv)

    model = PT_HICnet(
        in_channels=expected_channels,
        use_normals=bool(data_cfg.get("use_normals", False)),
        film_mode=film_mode,
        pt_radius=tuple(model_cfg["pt_radius"]),
        pt_nsample=tuple(model_cfg["pt_nsample"]),
    ).to("cpu")
    criterion = KDEWeightedHuberLoss(
        kde_reference_csv=kde_csv,
        delta=resolved_delta,
        delta_source=delta_source,
    ).to("cpu")

    smoke_input = torch.randn(2, expected_channels, 1024, dtype=torch.float32)
    smoke_hic_point = torch.randn(2, 3, dtype=torch.float32)
    smoke_age_group = torch.randint(0, 2, (2,), dtype=torch.int64)
    smoke_target = torch.abs(torch.randn(2, 1, dtype=torch.float32)) + 1e-3

    pred, trans_feat = model(smoke_input, smoke_hic_point, smoke_age_group)
    loss = criterion(pred, smoke_target)

    if loss.ndim != 0:
        raise RuntimeError(f"[SmokeTest] loss must be scalar, got ndim={loss.ndim}")
    if (not torch.isfinite(loss)) or (loss.item() <= 0):
        raise RuntimeError(f"[SmokeTest] invalid loss={loss.item()}, expected finite positive scalar")

    print("[SmokeTest] passed: feather sample + forward/loss checks are valid.")


def accuracy_ratio(pred, target):
    p = torch.abs(pred)
    t = torch.abs(target)
    denom = torch.maximum(p, t)
    score = torch.where(denom > 0, torch.minimum(p, t) / denom, torch.zeros_like(denom))
    return torch.mean(score).item()


def neg_rate(pred):
    return (pred < 0).float().mean().item()


def apply_material_augmentation(fused_input_np, dropout_prob=0.0, jitter_std=0.0):
    """Regularize material z-score channels without changing xyz/thickness."""
    if fused_input_np.shape[-1] < 19:
        return fused_input_np
    material = fused_input_np[:, :, 4:19]
    if jitter_std > 0:
        noise = np.random.normal(0.0, jitter_std, size=material.shape).astype(material.dtype)
        material = material + noise
    if dropout_prob > 0:
        keep_mask = np.random.random(size=material.shape) >= dropout_prob
        material = np.where(keep_mask, material, 0.0).astype(fused_input_np.dtype)
    fused_input_np[:, :, 4:19] = material
    return fused_input_np


def main():
    args = parse_args()
    cfg = load_config(args.config)
    runtime_cfg = copy.deepcopy(cfg)
    data_cfg = runtime_cfg["data"]
    train_cfg = runtime_cfg["training"]
    model_cfg = runtime_cfg["model"]
    loss_cfg = runtime_cfg["loss"]
    exp_cfg = runtime_cfg["experiment"]

    resolved_film_mode = args.film_mode if args.film_mode is not None else model_cfg.get("film_mode", "none")
    if args.film_mode is not None and model_cfg.get("film_mode") != args.film_mode:
        warnings.warn(
            f"[ModelConfig] film_mode override: {model_cfg.get('film_mode')} (yaml) -> {args.film_mode} (cli)",
            UserWarning,
        )
    model_cfg["film_mode"] = resolved_film_mode

    resolved_delta, delta_source = resolve_delta(loss_cfg)
    material_dropout_prob = (
        args.material_dropout_prob
        if args.material_dropout_prob is not None
        else float(train_cfg.get("material_dropout_prob", 0.0))
    )
    material_jitter_std = (
        args.material_jitter_std
        if args.material_jitter_std is not None
        else float(train_cfg.get("material_jitter_std", 0.0))
    )
    if not 0.0 <= material_dropout_prob < 1.0:
        raise ValueError(f"material_dropout_prob must be in [0, 1), got {material_dropout_prob}")
    if material_jitter_std < 0.0:
        raise ValueError(f"material_jitter_std must be >= 0, got {material_jitter_std}")
    train_cfg["material_dropout_prob"] = float(material_dropout_prob)
    train_cfg["material_jitter_std"] = float(material_jitter_std)

    use_wandb = args.use_wandb if args.use_wandb is not None else bool(train_cfg.get("use_wandb", False))
    wandb_project = train_cfg.get("wandb_project", "pt-hicnet")

    project_root = PROJECT_ROOT
    for env_key, yaml_key in [
        ("PT_HICNET_MATERIAL_LOOKUP_PATH", "material_lookup_path"),
        ("PT_HICNET_NORMALIZATION_PARAMS_PATH", "normalization_params_path"),
    ]:
        rel = data_cfg.get(yaml_key)
        if rel:
            os.environ[env_key] = str(project_root / rel)

    from data_utils.HICLoader_feather import HICDataLoader  # noqa: E402
    from losses import KDEWeightedHuberLoss  # noqa: E402
    from pt_hicnet import PT_HICnet  # noqa: E402

    seed = args.seed if args.seed is not None else exp_cfg["seeds"][0]
    set_seed(seed)

    # Resolve test vehicles early so we can disambiguate run directories.
    # Only append vehicle tag when --test_vehicles is explicitly passed via CLI;
    # otherwise keep backward-compatible naming (Phase 3 LOCO exp_names already
    # include fold/vehicle info).
    test_vehicles = args.test_vehicles if args.test_vehicles is not None else data_cfg.get("test_vehicles", ["JX65"])

    exp_name = args.exp_name if args.exp_name is not None else exp_cfg["name"]
    if args.test_vehicles is not None:
        vehicle_tag = "_".join(test_vehicles) if len(test_vehicles) <= 2 else f"{len(test_vehicles)}cars"
        run_name = f"{exp_name}_{vehicle_tag}_seed{seed}_film-{resolved_film_mode}"
    else:
        run_name = f"{exp_name}_seed{seed}_film-{resolved_film_mode}"
    # Disambiguate material augmentation variants to prevent P0/P1 collision
    if material_dropout_prob > 0:
        run_name += f"_md{material_dropout_prob:g}"
    if material_jitter_std > 0:
        run_name += f"_mj{material_jitter_std:g}"
    exp_dir = PROJECT_ROOT / exp_cfg["output_root"] / run_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    runtime_cfg.setdefault("runtime", {})
    runtime_cfg["model"].pop("ablation_mode", None)
    runtime_cfg["runtime"].update(
        {
            "seed": int(seed),
            "film_mode": resolved_film_mode,
            "delta": float(resolved_delta),
            "delta_source": delta_source,
        }
    )
    config_hash = compute_config_hash(runtime_cfg, resolved_delta)
    runtime_cfg["runtime"]["config_hash"] = config_hash

    with open(exp_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(runtime_cfg, f, sort_keys=True, allow_unicode=True)
    with open(exp_dir / "config_hash.txt", "w", encoding="utf-8") as f:
        f.write(config_hash + "\n")
    print(f"[Config] saved config_used.yaml and config_hash.txt to {exp_dir}")

    if use_wandb and wandb is not None:
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "seed": int(seed),
                "film_mode": resolved_film_mode,
                "batch_size": int(train_cfg["batch_size"]),
                "learning_rate": float(train_cfg["learning_rate"]),
                "weight_decay": float(train_cfg["decay_rate"]),
                "epochs": int(train_cfg["epoch"]),
                "in_channels": 22 if data_cfg.get("use_normals", False) else 19,
                "loss_type": loss_cfg.get("type", "kde_weighted_huber"),
                "loss_delta": float(resolved_delta),
                "test_vehicles": data_cfg.get("test_vehicles", ["JX65"]),
            },
            dir=str(exp_dir),
        )
    elif use_wandb and wandb is None:
        print("[WandB] wandb not installed; skipping logging.")

    runtime_args = argparse.Namespace(
        num_point=int(data_cfg["num_point"]),
        use_uniform_sample=bool(data_cfg["use_uniform_sample"]),
        use_normals=bool(data_cfg["use_normals"]),
    )

    run_smoke_test(
        project_root=project_root,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        loss_cfg=loss_cfg,
        film_mode=resolved_film_mode,
        resolved_delta=resolved_delta,
        delta_source=delta_source,
        runtime_args=runtime_args,
        HICDataLoader=HICDataLoader,
        PT_HICnet=PT_HICnet,
        KDEWeightedHuberLoss=KDEWeightedHuberLoss,
    )

    data_root = data_cfg.get("data_root", "")
    search_dir = os.path.join(data_root, data_cfg["train_data_dir"]) if data_cfg["train_data_dir"] else data_root
    all_files = collect_data_files(search_dir)
    # test_vehicles resolved above (before run_name construction)
    train_files, test_files = split_train_test(all_files, test_vehicles)
    print(f"Train files: {len(train_files)}  |  Test files: {len(test_files)}  (test vehicles: {test_vehicles})")

    train_loader = build_loader(
        train_files, runtime_args,
        int(train_cfg["batch_size"]), int(train_cfg["num_workers"]),
        early_fusion=True, normalize_thickness=bool(data_cfg.get("normalize_thickness", True)),
        shuffle=True, eval_deterministic=False,  # stochastic = data augmentation
    )
    test_loader = build_loader(
        test_files, runtime_args,
        int(train_cfg["batch_size"]), int(train_cfg["num_workers"]),
        early_fusion=True, normalize_thickness=bool(data_cfg.get("normalize_thickness", True)),
        shuffle=False, eval_deterministic=True,  # fixed sampling = reproducible metrics
    )

    # --- Build validation split from training data (stratified by age_group + HIC bucket) ---
    val_loader = None
    val_split = args.val_split
    if val_split > 0 and val_split < 1.0:
        train_dataset = train_loader.dataset
        # Collect stratification labels
        stratify_labels = []
        for i in range(len(train_dataset)):
            _, _, _, age_group, target = train_dataset[i]
            hic = target.item() if hasattr(target, 'item') else float(target)
            if hic < 500:
                hic_bin = 0
            elif hic < 1000:
                hic_bin = 1
            elif hic < 1500:
                hic_bin = 2
            elif hic < 2000:
                hic_bin = 3
            else:
                hic_bin = 4
            ag = age_group.item() if hasattr(age_group, 'item') else int(age_group)
            stratify_labels.append((ag, hic_bin))

        # Group indices by stratification label
        label_to_indices = defaultdict(list)
        for idx, label in enumerate(stratify_labels):
            label_to_indices[label].append(idx)

        rng = np.random.RandomState(args.split_seed)
        train_indices = []
        val_indices = []
        for label, indices in label_to_indices.items():
            rng.shuffle(indices)
            n_val = max(1, int(len(indices) * val_split))
            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])

        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)
        train_loader = DataLoader(
            train_subset, batch_size=int(train_cfg["batch_size"]), shuffle=True,
            num_workers=int(train_cfg["num_workers"]),
        )
        val_loader = DataLoader(
            val_subset, batch_size=int(train_cfg["batch_size"]), shuffle=False,
            num_workers=int(train_cfg["num_workers"]),
        )
        print(f"[Split] train={len(train_indices)} val={len(val_indices)} test={len(test_loader.dataset)}"
              f"  split_seed={args.split_seed}  stratified_by=(age_group,hic_bucket)")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PT_HICnet(
        in_channels=22 if data_cfg.get("use_normals", False) else 19,
        use_normals=bool(data_cfg.get("use_normals", False)),
        film_mode=resolved_film_mode,
        pt_radius=tuple(model_cfg["pt_radius"]),
        pt_nsample=tuple(model_cfg["pt_nsample"]),
    ).to(device)
    if use_wandb and wandb is not None:
        wandb.watch(model, log="all", log_freq=10)
    kde_csv = loss_cfg.get("kde_reference_csv")
    if kde_csv:
        kde_csv = str(PROJECT_ROOT / kde_csv)
    criterion = KDEWeightedHuberLoss(
        kde_reference_csv=kde_csv,
        delta=resolved_delta,
        delta_source=delta_source,
    ).to(device)

    # --- 诊断日志: Loss 配置确认 ---
    print(f"[Diagnostics] Loss config locked:")
    print(f"  delta          = {resolved_delta:.6g} (source: {delta_source})")
    print(f"  KDE            = {'ON' if kde_csv else 'OFF'}")
    print(f"  film_mode      = {resolved_film_mode}")
    print(f"  seed           = {seed}")
    print(f"  pt_radius      = {model.pt_radius if hasattr(model, 'pt_radius') else 'N/A'}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["decay_rate"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(train_cfg["scheduler_step_size"]),
        gamma=float(train_cfg["scheduler_gamma"]),
    )

    # Early stopping setup
    resolved_patience = args.patience
    if resolved_patience is None:
        resolved_patience = int(train_cfg.get("patience", 50))
    if resolved_patience < 0:
        resolved_patience = 0
    resolve_restore = args.restore_best
    if resolve_restore is None:
        resolve_restore = bool(train_cfg.get("restore_best", True))
    adult_hic_weight = args.adult_hic_weight
    if adult_hic_weight is None:
        adult_hic_weight = float(train_cfg.get("adult_hic_weight", 0.0))
    adult_hic_threshold = args.adult_hic_threshold
    if adult_hic_threshold is None:
        adult_hic_threshold = float(train_cfg.get("adult_hic_threshold", 1000.0))
    print(
        f"[EarlyStopping] patience={resolved_patience} "
        f"{'(disabled)' if resolved_patience == 0 else ''}  restore_best={resolve_restore}"
    )
    if adult_hic_weight > 0:
        print(f"[LossWeight] adult_hic_weight={adult_hic_weight}  threshold={adult_hic_threshold}")
    if material_dropout_prob > 0 or material_jitter_std > 0:
        print(
            f"[MaterialAug] dropout_prob={material_dropout_prob:.4g} "
            f"jitter_std={material_jitter_std:.4g} on channels 4:19"
        )

    best_mse = float("inf")
    best_acc = 0.0
    best_acc_epoch = 0
    best_acc_mse = float("inf")
    epochs_since_best = 0
    stopped_early = False
    history = []
    for epoch in range(int(train_cfg["epoch"])):
        model.train()
        # Restore stochastic point sampling for training (data augmentation)
        _set_datasets_eval_deterministic(train_loader, False)
        train_losses = []
        feat_norms = []
        for fused_input, hic_point, _, age_group, target in train_loader:
            optimizer.zero_grad()

            # B3: 3D 点云数据增强（与 baseline 一致，仅作用于 XYZ 通道 0:3）
            fused_input_np = fused_input.data.numpy()
            fused_input_np = provider.random_point_dropout(fused_input_np)
            fused_input_np[:, :, 0:3] = provider.random_scale_point_cloud(fused_input_np[:, :, 0:3])
            fused_input_np[:, :, 0:3] = provider.shift_point_cloud(fused_input_np[:, :, 0:3])
            fused_input_np = apply_material_augmentation(
                fused_input_np,
                dropout_prob=material_dropout_prob,
                jitter_std=material_jitter_std,
            )
            fused_input = torch.tensor(fused_input_np, dtype=torch.float32)

            fused_input = fused_input.to(device).transpose(2, 1)
            hic_point = hic_point.to(device)
            age_group = age_group.to(device)
            target = target.to(device)
            pred, trans_feat = model(fused_input, hic_point, age_group)
            loss = criterion(pred, target)
            # Optional: up-weight adult high-HIC samples
            if adult_hic_weight > 0:
                adult_hic_mask = (age_group.squeeze() == 1) & (target.squeeze() > adult_hic_threshold)
                if adult_hic_mask.any():
                    diff = pred[adult_hic_mask] - target[adult_hic_mask]
                    delta = resolved_delta
                    sample_loss = torch.where(
                        diff.abs() <= delta,
                        0.5 * diff ** 2,
                        delta * (diff.abs() - 0.5 * delta),
                    )
                    extra_loss = sample_loss.mean() * (adult_hic_weight - 1.0)
                    loss = loss + extra_loss
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            feat_norms.append(trans_feat.data.norm().item())
        scheduler.step()

        model.eval()
        # Enable deterministic point sampling for reproducible eval metrics
        _set_datasets_eval_deterministic(test_loader, True)
        if val_loader is not None:
            _set_datasets_eval_deterministic(val_loader, True)
        # --- Evaluate on val (for early stopping) if val_loader exists ---
        monitor_loader = val_loader if val_loader is not None else test_loader
        monitor_name = "val" if val_loader is not None else "test"
        all_pred, all_true = [], []
        with torch.no_grad():
            for fused_input, hic_point, _, age_group, target in monitor_loader:
                fused_input = fused_input.to(device).transpose(2, 1)
                hic_point = hic_point.to(device)
                age_group = age_group.to(device)
                target = target.to(device)
                pred, _ = model(fused_input, hic_point, age_group)
                all_pred.append(pred)
                all_true.append(target)
        pred_tensor = torch.cat(all_pred, dim=0)
        true_tensor = torch.cat(all_true, dim=0)
        mse = torch.nn.functional.mse_loss(pred_tensor, true_tensor).item()
        acc = accuracy_ratio(pred_tensor, true_tensor)
        nr = neg_rate(pred_tensor)

        # --- Evaluate on test (record only, NOT used for early stopping) ---
        test_mse = float("nan")
        test_acc = 0.0
        test_nr = 0.0
        if val_loader is not None:
            all_pred, all_true = [], []
            with torch.no_grad():
                for fused_input, hic_point, _, age_group, target in test_loader:
                    fused_input = fused_input.to(device).transpose(2, 1)
                    hic_point = hic_point.to(device)
                    age_group = age_group.to(device)
                    target = target.to(device)
                    pred, _ = model(fused_input, hic_point, age_group)
                    all_pred.append(pred)
                    all_true.append(target)
            test_pred_tensor = torch.cat(all_pred, dim=0)
            test_true_tensor = torch.cat(all_true, dim=0)
            test_mse = torch.nn.functional.mse_loss(test_pred_tensor, test_true_tensor).item()
            test_acc = accuracy_ratio(test_pred_tensor, test_true_tensor)
            test_nr = neg_rate(test_pred_tensor)

        epoch_train_loss = float(np.mean(train_losses))
        if val_loader is not None:
            print(f"[Epoch {epoch+1:3d}] train_loss={epoch_train_loss:>10.6g}  "
                  f"val_mse={mse:>10.6g}  val_acc={acc:.4f}  "
                  f"test_mse={test_mse:>10.6g}  test_acc={test_acc:.4f}  "
                  f"neg_rate={nr:.4f}  feat_norm={np.mean(feat_norms):.4f}")
        else:
            print(f"[Epoch {epoch+1:3d}] train_loss={epoch_train_loss:>10.6g}  "
                  f"test_mse={mse:>10.6g}  test_acc={acc:.4f}  "
                  f"neg_rate={nr:.4f}  feat_norm={np.mean(feat_norms):.4f}")
        epoch_feat_norm = float(np.mean(feat_norms))
        history_entry = {
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss,
            f"{monitor_name}_mse": mse,
            f"{monitor_name}_accuracy": acc,
            "neg_rate": nr,
            "feat_norm": epoch_feat_norm,
        }
        if val_loader is not None:
            history_entry["test_mse"] = test_mse
            history_entry["test_accuracy"] = test_acc
            history_entry["test_neg_rate"] = test_nr
        history.append(history_entry)

        if use_wandb and wandb is not None:
            wb_log = {
                "epoch": epoch + 1,
                "train_loss": epoch_train_loss,
                f"{monitor_name}_mse": mse,
                f"{monitor_name}_accuracy": acc,
                "neg_rate": nr,
                "feat_norm": epoch_feat_norm,
                "lr": scheduler.get_last_lr()[0],
            }
            if val_loader is not None:
                wb_log["test_mse"] = test_mse
                wb_log["test_accuracy"] = test_acc
            wandb.log(wb_log)

        if mse < best_mse:
            best_mse = mse
            torch.save(
                {
                    "epoch": epoch + 1,
                    "mse": mse,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config_path": args.config,
                    "film_mode": resolved_film_mode,
                    "model_hparams": {
                        "in_channels": 22 if data_cfg.get("use_normals", False) else 19,
                        "use_normals": bool(data_cfg.get("use_normals", False)),
                        "film_mode": resolved_film_mode,
                        "pt_radius": list(model_cfg["pt_radius"]),
                        "pt_nsample": list(model_cfg["pt_nsample"]),
                    },
                    "resolved_loss": {
                        "delta": float(resolved_delta),
                        "source": delta_source,
                    },
                    "training_hparams": {
                        "batch_size": int(train_cfg["batch_size"]),
                    },
                    "config_hash": config_hash,
                },
                ckpt_dir / "best_model.pth",
            )

        # Track best accuracy separately for early stopping
        if acc > best_acc:
            best_acc = acc
            best_acc_epoch = epoch + 1
            best_acc_mse = mse
            epochs_since_best = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "mse": mse,
                    "accuracy": acc,
                    "model_state_dict": model.state_dict(),
                    "config_path": args.config,
                    "film_mode": resolved_film_mode,
                    "model_hparams": {
                        "in_channels": 22 if data_cfg.get("use_normals", False) else 19,
                        "use_normals": bool(data_cfg.get("use_normals", False)),
                        "film_mode": resolved_film_mode,
                        "pt_radius": list(model_cfg["pt_radius"]),
                        "pt_nsample": list(model_cfg["pt_nsample"]),
                    },
                    "resolved_loss": {
                        "delta": float(resolved_delta),
                        "source": delta_source,
                    },
                    "training_hparams": {
                        "batch_size": int(train_cfg["batch_size"]),
                    },
                    "config_hash": config_hash,
                },
                ckpt_dir / "best_acc_model.pth",
            )
        else:
            epochs_since_best += 1

        if resolved_patience > 0 and epochs_since_best >= resolved_patience:
            print(
                f"[EarlyStopping] No accuracy improvement for {resolved_patience} epochs "
                f"(best_acc={best_acc:.4f} @ ep{best_acc_epoch}). Stopping at epoch {epoch+1}."
            )
            stopped_early = True

            # Restore best accuracy weights if requested
            if resolve_restore:
                print(f"[EarlyStopping] Restoring best_acc weights from epoch {best_acc_epoch}")
                ckpt = torch.load(ckpt_dir / "best_acc_model.pth", map_location=device)
                model.load_state_dict(ckpt["model_state_dict"])
                # Re-evaluate restored weights on monitor set (val if exists, else test)
                model.eval()
                all_pred, all_true = [], []
                with torch.no_grad():
                    for fused_input, hic_point, _, age_group, target in monitor_loader:
                        fused_input = fused_input.to(device).transpose(2, 1)
                        hic_point = hic_point.to(device)
                        age_group = age_group.to(device)
                        target = target.to(device)
                        pred, _ = model(fused_input, hic_point, age_group)
                        all_pred.append(pred)
                        all_true.append(target)
                pred_tensor = torch.cat(all_pred, dim=0)
                true_tensor = torch.cat(all_true, dim=0)
                restored_mse = torch.nn.functional.mse_loss(pred_tensor, true_tensor).item()
                restored_acc = accuracy_ratio(pred_tensor, true_tensor)
                restored_nr = neg_rate(pred_tensor)
                restored_entry = {
                    "epoch": f"restored@{best_acc_epoch}",
                    f"{monitor_name}_mse": restored_mse,
                    f"{monitor_name}_accuracy": restored_acc,
                    "neg_rate": restored_nr,
                    "restored_from_epoch": best_acc_epoch,
                }
                # Also eval test if val was used for monitoring
                if val_loader is not None:
                    all_pred, all_true = [], []
                    with torch.no_grad():
                        for fused_input, hic_point, _, age_group, target in test_loader:
                            fused_input = fused_input.to(device).transpose(2, 1)
                            hic_point = hic_point.to(device)
                            age_group = age_group.to(device)
                            target = target.to(device)
                            pred, _ = model(fused_input, hic_point, age_group)
                            all_pred.append(pred)
                            all_true.append(target)
                    test_pred_tensor = torch.cat(all_pred, dim=0)
                    test_true_tensor = torch.cat(all_true, dim=0)
                    restored_test_mse = torch.nn.functional.mse_loss(test_pred_tensor, test_true_tensor).item()
                    restored_test_acc = accuracy_ratio(test_pred_tensor, test_true_tensor)
                    restored_test_nr = neg_rate(test_pred_tensor)
                    restored_entry["test_mse"] = restored_test_mse
                    restored_entry["test_accuracy"] = restored_test_acc
                    restored_entry["test_neg_rate"] = restored_test_nr
                print(
                    f"[EarlyStopping] Restored {monitor_name} metrics: acc={restored_acc:.4f} mse={restored_mse:.6g} neg_rate={restored_nr:.4f}"
                )
                history.append(restored_entry)
            break

    with open(exp_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"saved run: {exp_dir}")
    print(
        f"[Summary] best_mse={best_mse:.6g}  "
        f"best_acc={best_acc:.4f} @ ep{best_acc_epoch} (mse={best_acc_mse:.6g})  "
        f"early_stopped={stopped_early}  patience={resolved_patience}  "
        f"restore_best={resolve_restore}"
    )

    if use_wandb and wandb is not None:
        wandb.summary["best_acc"] = best_acc
        wandb.summary["best_acc_epoch"] = best_acc_epoch
        wandb.summary["best_acc_mse"] = best_acc_mse
        wandb.summary["early_stopped"] = stopped_early
        wandb.summary["restore_best"] = resolve_restore
        wandb.finish()


if __name__ == "__main__":
    main()
