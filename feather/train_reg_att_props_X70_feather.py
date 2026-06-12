"""
Author: Benny (modified for new JSON by ChatGPT)
Date: Nov 2019  |  Updated: 2025-11
"""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import sys
import pickle
import gc
import datetime
import logging
import importlib
import argparse
import random
from pathlib import Path
import inspect

import torch
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns  # 仅用于可视化，可按需去掉
import yaml

from torch.utils.data import ConcatDataset
# HICDataLoader 延迟到 main() 中 import，确保 env 变量先设置

try:
    import wandb
except ImportError:
    wandb = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
PROJECT_ROOT = Path(BASE_DIR).resolve().parents[0]
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append(str(PROJECT_ROOT / 'models'))

# ----------------------------
# Args
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='yaml config path')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=15, help='batch size in training')
    parser.add_argument('--model', default='pointnet2_reg_att_props', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=8192, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampling')
    parser.add_argument('--train_data_dir', type=str, default='data/train', help='train data folder')
    parser.add_argument('--test_data_dir', type=str, default='data/test', help='test data folder')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
    parser.add_argument('--use_wandb', action='store_true', default=None, help='log to wandb')
    parser.add_argument('--use_early_fusion', action='store_true', default=False, help='use fused input [xyz+thickness+material]')
    parser.add_argument('--normalize_thickness', action='store_true', default=False, help='normalize thickness into [0,1]')
    parser.add_argument('--ablation_mode', type=str, default=None, choices=['baseline', 'early_fusion_clean', 'pt_hicnet'], help='ablation mode')
    parser.add_argument('--shape_audit', action='store_true', default=False, help='print E1 shape audit once')
    parser.add_argument('--kde_reference_csv', type=str, default='feather/data_utils/y_train(1).csv', help='kde reference csv')
    parser.add_argument('--delta', type=float, default=5.0, help='Huber delta for regression loss')
    parser.add_argument('--test_vehicles', type=str, nargs='+', default=None, help='Override test vehicles for LOCO-CV')
    parser.add_argument('--val_split', type=float, default=0.0, help='Fraction of training data for validation (0=skip)')
    parser.add_argument('--split_seed', type=int, default=2026, help='Seed for train/val split')
    parser.add_argument('--patience', type=int, default=0, help='Early stopping patience (0=disable)')
    parser.add_argument('--exp_name', type=str, default=None, help='Override experiment name')
    args = parser.parse_args()
    return load_config_defaults(args)


def load_config_defaults(args):
    config_path = Path(args.config)
    if not config_path.exists():
        return args
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    data_cfg = cfg.get('data', {})
    train_cfg = cfg.get('training', {})
    model_cfg = cfg.get('model', {})
    loss_cfg = cfg.get('loss', {})
    data_root = data_cfg.get('data_root', '')
    if args.train_data_dir == 'data/train':
        args.train_data_dir = data_cfg.get('train_data_dir', args.train_data_dir)
    if args.test_data_dir == 'data/test':
        args.test_data_dir = data_cfg.get('test_data_dir', args.test_data_dir)
    # 拼接 data_root
    if data_root and args.train_data_dir:
        args.train_data_dir = os.path.join(data_root, args.train_data_dir)
    elif data_root:
        args.train_data_dir = data_root
    if data_root and args.test_data_dir:
        args.test_data_dir = os.path.join(data_root, args.test_data_dir)
    elif data_root:
        args.test_data_dir = data_root
    args.num_point = int(data_cfg.get('num_point', args.num_point))
    args.use_uniform_sample = bool(data_cfg.get('use_uniform_sample', args.use_uniform_sample))
    args.use_normals = bool(data_cfg.get('use_normals', args.use_normals))
    args.use_early_fusion = bool(data_cfg.get('early_fusion', args.use_early_fusion))
    args.normalize_thickness = bool(data_cfg.get('normalize_thickness', args.normalize_thickness))
    args.batch_size = int(train_cfg.get('batch_size', args.batch_size))
    args.epoch = int(train_cfg.get('epoch', args.epoch))
    args.learning_rate = float(train_cfg.get('learning_rate', args.learning_rate))
    args.decay_rate = float(train_cfg.get('decay_rate', args.decay_rate))
    if args.ablation_mode is None:
        args.ablation_mode = model_cfg.get('ablation_mode', 'baseline')
    # E0 baseline: model needs separate thickness & material_props, not fused
    if args.ablation_mode == 'baseline':
        args.use_early_fusion = False
    args.delta = float(loss_cfg.get('delta', args.delta))
    if 'kde_reference_csv' in loss_cfg:
        args.kde_reference_csv = loss_cfg.get('kde_reference_csv')
    else:
        args.kde_reference_csv = None
    args.test_vehicles = args.test_vehicles if args.test_vehicles is not None else data_cfg.get('test_vehicles', ['JX65'])
    if args.use_wandb is None:
        args.use_wandb = bool(train_cfg.get('use_wandb', False))
    args.wandb_project = train_cfg.get('wandb_project', 'pt-hicnet')
    return args


def unpack_batch(batch, use_early_fusion):
    if use_early_fusion:
        fused_input, hic_point, category, age_group, target = batch
        return fused_input, hic_point, category, None, None, age_group, target
    return batch

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed):
    """设置随机种子以确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置 PyTorch 的确定性操作（可能会降低性能，但提高可复现性）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def inplace_relu(m):
    if m.__class__.__name__.find('ReLU') != -1:
        m.inplace = True

def pick_test_files(test_dir: str) -> list:
    """从测试目录查找所有测试文件（支持 JSON 或 Feather）。
    支持两种结构：
    1. 直接包含文件：/test/*.feather
    2. 按车辆组织的目录：/test/X70/batch_0.feather
    """
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test data dir not found: {test_dir}")
    
    test_files = []
    
    # 首先检查是否直接包含文件
    direct_files = [os.path.join(test_dir, f) 
                    for f in os.listdir(test_dir) 
                    if f.endswith(('.json', '.feather')) and os.path.isfile(os.path.join(test_dir, f))]
    
    if direct_files:
        # 方式1：直接包含文件
        test_files = direct_files
    else:
        # 方式2：按车辆组织的目录结构
        # 遍历所有子目录，查找每个子目录下的所有 feather/json 文件
        for subdir in os.listdir(test_dir):
            subdir_path = os.path.join(test_dir, subdir)
            if os.path.isdir(subdir_path):
                # 查找该子目录下的所有 feather/json 文件
                subdir_files = [os.path.join(subdir_path, f) 
                               for f in os.listdir(subdir_path) 
                               if f.endswith(('.json', '.feather')) and os.path.isfile(os.path.join(subdir_path, f))]
                test_files.extend(subdir_files)
    
    if not test_files:
        raise FileNotFoundError(f"No .json or .feather found under {test_dir}")
    
    return test_files

def test(model, test_dataset, device, args):
    regressor = model.eval()
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=0)

    mse_losses, accuracies = [], []
    y_test_true, y_test_pred = [], []
    test_results = []
    
    for _, batch in enumerate(loader):
        points, hic_point, category, thickness, material_props, age_group, target = unpack_batch(batch, args.use_early_fusion)
        points = points.to(device).transpose(2, 1)
        hic_point = hic_point.to(device)
        target = target.to(device)
        category = category.to(device)
        age_group = age_group.to(device)
        if thickness is not None:
            thickness = thickness.to(device)
        if material_props is not None:
            material_props = material_props.to(device)

        if args.ablation_mode == "pt_hicnet":
            pred, trans_feat = regressor(points, hic_point, age_group)
        elif args.use_early_fusion:
            pred, trans_feat = regressor(points, hic_point, category, age_group)
        else:
            pred, trans_feat = regressor(points, hic_point, category, thickness, material_props, age_group)

        mse_loss = torch.nn.functional.mse_loss(pred, target)
        mse_losses.append(mse_loss.item())

        y_true = target.detach().cpu().numpy()
        y_pred = pred.detach().cpu().numpy()
        y_test_true.extend(y_true)
        y_test_pred.extend(y_pred)

        for i in range(len(y_true)):
            try:
                min_val = min(abs(y_true[i]), abs(y_pred[i]))
                max_val = max(abs(y_true[i]), abs(y_pred[i]))
                acc = (min_val / max_val) * 100 if max_val != 0 else 0
            except Exception:
                acc = 0.0
            accuracies.append(acc)

    y_test_true = np.array(y_test_true)
    y_test_pred = np.array(y_test_pred)
    test_loss = (y_test_true - y_test_pred) ** 2
    test_results.append({'y_true': y_test_true, 'y_pred': y_test_pred, 'loss': test_loss})

    return float(np.mean(mse_losses)), float(np.mean(accuracies)), float(np.std(accuracies)), test_results

# ----------------------------
# Main
# ----------------------------
def main(args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # --- Set random seed for reproducibility
    set_seed(args.seed)

    # --- Create dirs
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    if args.exp_name is not None:
        exp_dir = Path('experiments') / args.exp_name
    else:
        exp_dir = Path('experiments') / 'X70' / 'regression' / (args.log_dir if args.log_dir is not None else timestr)
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    if args.use_wandb and wandb is not None:
        wandb.init(
            project=args.wandb_project,
            name=exp_dir.name,
            config={
                "seed": args.seed,
                "ablation_mode": args.ablation_mode,
                "use_early_fusion": args.use_early_fusion,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.decay_rate,
                "epochs": args.epoch,
                "test_vehicles": args.test_vehicles,
            },
            dir=str(exp_dir),
        )
    elif args.use_wandb and wandb is None:
        print("[WandB] wandb not installed; skipping logging.")

    # --- Logger
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    if logger.hasHandlers():  # 防止重复添加 handler
        logger.handlers.clear()
    logger.addHandler(file_handler)
    def log_string(s): 
        logger.info(s); print(s)

    log_string('PARAMETER ...')
    log_string(str(args))

    # B2: 在 import DataLoader 之前注入 pkl 路径环境变量
    _cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8')) or {}
    _dc = _cfg.get('data', {})
    _mat_pkl = _dc.get("material_lookup_path")
    _norm_pkl = _dc.get("normalization_params_path")
    if _mat_pkl:
        os.environ["PT_HICNET_MATERIAL_LOOKUP_PATH"] = str(PROJECT_ROOT / _mat_pkl)
    if _norm_pkl:
        os.environ["PT_HICNET_NORMALIZATION_PARAMS_PATH"] = str(PROJECT_ROOT / _norm_pkl)
    from data_utils.HICLoader_feather import HICDataLoader, CAR_TO_VEHICLE  # noqa: E402

    # 收集全部文件并按车辆划分训练/测试集
    def _collect_all_files(data_dir):
        files = []
        for parent, _, names in os.walk(data_dir):
            for name in names:
                if name.endswith(('.json', '.feather')):
                    files.append(os.path.join(parent, name))
        return sorted(files)

    def _split_by_vehicle(all_files, test_vehicles):
        vehicle_to_car = {v: k for k, v in CAR_TO_VEHICLE.items()}
        test_car_dirs = {vehicle_to_car.get(v, v.lower()) for v in test_vehicles}
        train_f, test_f = [], []
        for fp in all_files:
            parent = os.path.basename(os.path.dirname(fp))
            if parent.lower() in test_car_dirs:
                test_f.append(fp)
            else:
                train_f.append(fp)
        return train_f, test_f

    # --- DATA
    log_string('Load dataset ...')
    all_data_files = _collect_all_files(args.train_data_dir)
    train_files, test_files = _split_by_vehicle(all_data_files, args.test_vehicles)
    log_string(f'Train: {len(train_files)} files | Test: {len(test_files)} files (test vehicles: {args.test_vehicles})')
    all_train_datasets = [
        HICDataLoader(
            root=fp,
            args=args,
            early_fusion=args.use_early_fusion,
            normalize_thickness=args.normalize_thickness,
        )
        for fp in train_files
    ]
    train_loader = torch.utils.data.DataLoader(
        ConcatDataset(all_train_datasets),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    # --- Build validation split from training data (stratified) ---
    val_loader = None
    if args.val_split > 0 and args.val_split < 1.0:
        from collections import defaultdict
        from torch.utils.data import Subset
        train_dataset = train_loader.dataset
        stratify_labels = []
        for i in range(len(train_dataset)):
            batch = train_dataset[i]
            # E0 returns 7 items, E1 returns 5 — age_group and target are always last two
            age_group, target = batch[-2], batch[-1]
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

        label_to_indices = defaultdict(list)
        stratify_mode = "(age_group, hic_bucket)"
        for idx, label in enumerate(stratify_labels):
            label_to_indices[label].append(idx)

        # Fallback: if any stratum is too small, coarsen to age_group only
        min_stratum = min(len(v) for v in label_to_indices.values())
        if min_stratum < 2:
            label_to_indices = defaultdict(list)
            for idx, (ag, _) in enumerate(stratify_labels):
                label_to_indices[ag].append(idx)
            stratify_mode = "(age_group) — fallback: hic_bucket strata too small"
            min_stratum = min(len(v) for v in label_to_indices.values())

        # Final fallback: random split
        if min_stratum < 2:
            indices = list(range(len(train_dataset)))
            rng = np.random.RandomState(args.split_seed)
            rng.shuffle(indices)
            n_val = max(1, int(len(indices) * args.val_split))
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            stratify_mode = "random — fallback: age_group strata too small"
        else:
            rng = np.random.RandomState(args.split_seed)
            train_indices, val_indices = [], []
            for label, indices in label_to_indices.items():
                rng.shuffle(indices)
                n_val = max(1, int(len(indices) * args.val_split))
                val_indices.extend(indices[:n_val])
                train_indices.extend(indices[n_val:])

        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)
        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True,
            num_workers=0, drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False,
            num_workers=0, drop_last=False,
        )
        log_string(f'[Split] train={len(train_indices)} val={len(val_indices)}  '
                   f'split_seed={args.split_seed}  mode={stratify_mode}')

    # --- MODEL
    kde_reference_csv = args.kde_reference_csv if args.kde_reference_csv else None
    if args.ablation_mode == "pt_hicnet":
        from models.pt_hicnet import PT_HICnet
        from models.losses import KDEWeightedHuberLoss
        regressor = PT_HICnet(
            in_channels=22 if args.use_normals else 19,
            use_normals=args.use_normals,
            film_mode="none",
        )
        criterion = KDEWeightedHuberLoss(
            kde_reference_csv=kde_reference_csv,
            delta=float(args.delta),
            delta_source="yaml",
        )
        args.use_early_fusion = True
    else:
        model_name = "pointnet2_reg_ablation" if args.ablation_mode == "early_fusion_clean" else args.model
        model = importlib.import_module(model_name)
        model_kwargs = {"normal_channel": args.use_normals, "num_point": args.num_point}
        sig = inspect.signature(model.get_model)
        if "shape_audit" in sig.parameters:
            model_kwargs["shape_audit"] = args.shape_audit
        regressor = model.get_model(**model_kwargs)
        try:
            criterion = model.get_loss(kde_reference_csv=kde_reference_csv, delta=float(args.delta))
        except TypeError:
            try:
                criterion = model.get_loss(delta=float(args.delta))
            except TypeError:
                criterion = model.get_loss()
    regressor.apply(inplace_relu)
    effective_delta = getattr(criterion, "delta", args.delta)
    log_string(
        f"[LossConfig] ablation_mode={args.ablation_mode} "
        f"delta={float(effective_delta):.6g} "
        f"kde_reference_csv={kde_reference_csv if kde_reference_csv else '<disabled>'}"
    )

    # Device
    if torch.cuda.is_available():
        try:
            device = torch.device(f"cuda:{int(args.gpu)}")
        except Exception:
            device = torch.device("cuda:0")
            print(f"指定 GPU {args.gpu} 无效，改用 cuda:0")
    else:
        device = torch.device("cpu")
        print("CUDA 不可用，使用 CPU")

    regressor = regressor.to(device)
    criterion = criterion.to(device)

    if args.use_wandb and wandb is not None:
        wandb.watch(regressor, log="all", log_freq=10)

    # Optim / Scheduler
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(regressor.parameters(), lr=args.learning_rate,
                                     betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(regressor.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    # --- Helper: evaluate model on a loader ---
    def eval_on_loader(loader):
        regressor.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for batch in loader:
                points, hic_point, category, thickness, material_props, age_group, target = unpack_batch(batch, args.use_early_fusion)
                points = points.to(device).transpose(2, 1)
                hic_point = hic_point.to(device)
                target = target.to(device)
                category = category.to(device)
                age_group = age_group.to(device)
                if thickness is not None:
                    thickness = thickness.to(device)
                if material_props is not None:
                    material_props = material_props.to(device)
                if args.ablation_mode == "pt_hicnet":
                    pred, _ = regressor(points, hic_point, age_group)
                elif args.use_early_fusion:
                    pred, _ = regressor(points, hic_point, category, age_group)
                else:
                    pred, _ = regressor(points, hic_point, category, thickness, material_props, age_group)
                all_pred.append(pred)
                all_true.append(target)
        pred_tensor = torch.cat(all_pred, dim=0)
        true_tensor = torch.cat(all_true, dim=0)
        mse = torch.nn.functional.mse_loss(pred_tensor, true_tensor).item()
        # accuracy_ratio (same as train_pt_hicnet.py)
        p = torch.abs(pred_tensor)
        t = torch.abs(true_tensor)
        denom = torch.maximum(p, t)
        score = torch.where(denom > 0, torch.minimum(p, t) / denom, torch.zeros_like(denom))
        acc = torch.mean(score).item()
        return mse, acc

    # --- Training state
    monitor_loader = val_loader if val_loader is not None else test_loader
    monitor_name = "val" if val_loader is not None else "test"

    global_epoch = 0
    best_instance_mse = float('inf')
    best_mean_accuracy = float('-inf')
    best_std_accuracy = float('-inf')
    best_epoch = 0
    best_acc = 0.0
    best_acc_epoch = 0
    best_acc_mse = float('inf')
    epochs_since_best = 0
    stopped_early = False
    history = []

    train_results = []
    test_results_all = []

    # --- 预加载测试数据集（避免每个epoch重复加载）
    log_string('Loading test dataset...')
    all_test_datasets = [
        HICDataLoader(
            root=fp,
            args=args,
            early_fusion=args.use_early_fusion,
            normalize_thickness=args.normalize_thickness,
        )
        for fp in test_files
    ]
    test_dataset = ConcatDataset(all_test_datasets)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False,
    )
    log_string(f'Test dataset loaded: {len(test_files)} file(s)')
    for fp in test_files:
        log_string(f"  - {os.path.basename(fp)}")

    # --- Train loop
    logger.info('Start training...')
    for epoch in range(args.epoch):
        print(f"\nEpoch {epoch + 1}/{args.epoch}")
        regressor.train()
        epoch_losses = []
        mean_correct = []

        y_train_true, y_train_pred = [], []

        for batch_id, batch in enumerate(train_loader):
            print(f"Progress: {batch_id + 1}/{len(train_loader)} batches", end='\r')
            points, hic_point, category, thickness, material_props, age_group, target = unpack_batch(batch, args.use_early_fusion)

            optimizer.zero_grad()

            # 简单数据增强（与你原来相同）
            points_np = points.data.numpy()
            import provider
            points_np = provider.random_point_dropout(points_np)
            points_np[:, :, 0:3] = provider.random_scale_point_cloud(points_np[:, :, 0:3])
            points_np[:, :, 0:3] = provider.shift_point_cloud(points_np[:, :, 0:3])

            points = torch.tensor(points_np, dtype=torch.float32, device=device).transpose(2, 1)
            hic_point = hic_point.to(device)
            target = target.to(device)
            category = category.to(device)
            age_group = age_group.to(device)
            if thickness is not None:
                thickness = thickness.to(device)
            if material_props is not None:
                material_props = material_props.to(device)

            if args.ablation_mode == "pt_hicnet":
                pred, trans_feat = regressor(points, hic_point, age_group)
            elif args.use_early_fusion:
                pred, trans_feat = regressor(points, hic_point, category, age_group)
            else:
                pred, trans_feat = regressor(points, hic_point, category, thickness, material_props, age_group)
            loss = criterion(pred, target.float())
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            y_true = target.detach().cpu().numpy()
            y_pred = pred.detach().cpu().numpy()
            y_train_true.extend(y_true)
            y_train_pred.extend(y_pred)

            for i in range(len(y_true)):
                min_val = min(abs(y_true[i]), abs(y_pred[i]))
                max_val = max(abs(y_true[i]), abs(y_pred[i]))
                acc = (min_val / max_val) * 100 if max_val != 0 else 0
                mean_correct.append(acc)

            # 清理
            del points, hic_point, target, category, age_group, pred, trans_feat, loss
            torch.cuda.empty_cache()

        scheduler.step()

        # 记录训练结果
        epoch_loss = float(np.mean(epoch_losses))
        epoch_accuracy = float(np.mean(mean_correct))
        y_train_true = np.array(y_train_true); y_train_pred = np.array(y_train_pred)
        train_results.append({'y_true': y_train_true, 'y_pred': y_train_pred,
                              'loss': (y_train_pred - y_train_true) ** 2})

        # --- Evaluate on monitor set (val if exists, else test) for early stopping ---
        monitor_mse, monitor_acc = eval_on_loader(monitor_loader)

        # --- Evaluate on test set (record only, NOT for early stopping when val exists) ---
        test_mse = float("nan")
        test_acc = 0.0
        if val_loader is not None:
            test_mse, test_acc = eval_on_loader(test_loader)

        # Track best monitor accuracy for early stopping
        if monitor_acc > best_acc:
            best_acc = monitor_acc
            best_acc_epoch = epoch + 1
            best_acc_mse = monitor_mse
            epochs_since_best = 0
            # Save best_acc_model.pth
            state = {
                'epoch': epoch + 1,
                'mse': monitor_mse,
                'accuracy': monitor_acc,
                'model_state_dict': regressor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'film_mode': args.ablation_mode,
            }
            torch.save(state, checkpoints_dir / 'best_acc_model.pth')
        else:
            epochs_since_best += 1

        # Also track best_mse model (legacy)
        if monitor_mse < best_instance_mse:
            best_instance_mse = monitor_mse
        if monitor_acc > best_mean_accuracy:
            best_mean_accuracy = monitor_acc
            best_epoch = epoch + 1

        # Log
        log_line = (f"[Epoch {epoch+1:3d}] train_loss={epoch_loss:>10.6g}  "
                    f"{monitor_name}_mse={monitor_mse:>10.6g}  {monitor_name}_acc={monitor_acc:.4f}")
        if val_loader is not None:
            log_line += f"  test_mse={test_mse:>10.6g}  test_acc={test_acc:.4f}"
        log_string(log_line)

        history_entry = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            f"{monitor_name}_mse": monitor_mse,
            f"{monitor_name}_accuracy": monitor_acc,
        }
        if val_loader is not None:
            history_entry["test_mse"] = test_mse
            history_entry["test_accuracy"] = test_acc
        history.append(history_entry)

        if args.use_wandb and wandb is not None:
            wb_log = {
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                f"{monitor_name}_mse": monitor_mse,
                f"{monitor_name}_accuracy": monitor_acc,
                "lr": scheduler.get_last_lr()[0],
            }
            if val_loader is not None:
                wb_log["test_mse"] = test_mse
                wb_log["test_accuracy"] = test_acc
            wandb.log(wb_log)

        # --- Early stopping ---
        if args.patience > 0 and epochs_since_best >= args.patience:
            log_string(f"[EarlyStopping] No {monitor_name}_acc improvement for {args.patience} epochs "
                       f"(best_acc={best_acc:.4f} @ ep{best_acc_epoch}). Stopping at epoch {epoch+1}.")
            stopped_early = True
            # Restore best_acc weights
            log_string(f"[EarlyStopping] Restoring best_acc weights from epoch {best_acc_epoch}")
            ckpt = torch.load(checkpoints_dir / 'best_acc_model.pth', map_location=device)
            regressor.load_state_dict(ckpt["model_state_dict"])
            restored_mse, restored_acc = eval_on_loader(monitor_loader)
            restored_entry = {
                "epoch": f"restored@{best_acc_epoch}",
                f"{monitor_name}_mse": restored_mse,
                f"{monitor_name}_accuracy": restored_acc,
                "restored_from_epoch": best_acc_epoch,
            }
            if val_loader is not None:
                rest_test_mse, rest_test_acc = eval_on_loader(test_loader)
                restored_entry["test_mse"] = rest_test_mse
                restored_entry["test_accuracy"] = rest_test_acc
            history.append(restored_entry)
            log_string(f"[EarlyStopping] Restored {monitor_name} metrics: acc={restored_acc:.4f} mse={restored_mse:.6g}")
            break

        global_epoch += 1

        # 持久化曲线
        with open(str(exp_dir / 'train_results.pkl'), 'wb') as f:
            pickle.dump(train_results, f)
        with open(str(exp_dir / 'test_results.pkl'), 'wb') as f:
            pickle.dump(test_results_all, f)

    logger.info('End of training...')
    log_string(f'训练结束，最佳 epoch: {best_epoch}, 最佳 {monitor_name}_accuracy: {best_mean_accuracy:.4f}')

    # Save history.json
    import json
    with open(exp_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    log_string(f"[Summary] best_{monitor_name}_acc={best_acc:.4f} @ ep{best_acc_epoch} "
               f"early_stopped={stopped_early}  patience={args.patience}")
    log_string(f"saved run: {exp_dir}")

    if args.use_wandb and wandb is not None:
        wandb.summary["best_acc"] = best_acc
        wandb.summary["best_acc_epoch"] = best_acc_epoch
        wandb.summary["best_acc_mse"] = best_acc_mse
        wandb.summary["early_stopped"] = stopped_early
        wandb.finish()


if __name__ == '__main__':
    args = parse_args()
    main(args)