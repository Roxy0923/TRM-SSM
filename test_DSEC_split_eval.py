import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from train_TRM_DSEC import build_model, compute_basic_metrics
from loader.DSEC import DSECEventFlow


def parse_args():
    parser = argparse.ArgumentParser(description="Eval TRM DSEC checkpoint on test_split")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=str(REPO_ROOT / "exp_DSEC_meshflow/dsec_step1_0116_1725/lasted_ckpt.pth.tar"),
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(REPO_ROOT / "exp_DSEC_meshflow/dsec_step1_0116_1725/config.json"),
    )
    parser.add_argument(
        "--train_cfg_path",
        type=str,
        default=str(REPO_ROOT / "exp_DSEC_meshflow/dsec_step1_0116_1725/train_config.json"),
    )
    parser.add_argument(
        "--train_root",
        type=str,
        default=str(REPO_ROOT.parent / "data/DSEC/train"),
        help="DSEC train 目录（含 flow/ 和 events/）",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=str(REPO_ROOT.parent / "data/DSEC/test_split.txt"),
        help="测试划分文件（默认使用官方 test_split.txt）",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--step_scale", type=float, default=None, help="可覆盖 train_cfg 内的 step_scale")
    parser.add_argument("--flow_scale", type=float, default=None, help="可覆盖 train_cfg 内的 flow_scale")
    parser.add_argument("--num_bins", type=int, default=None, help="可覆盖 train_cfg/config 内的 num_bins")
    parser.add_argument("--test_fps", action="store_true", help="测试推理速度 (FPS)")
    parser.add_argument("--fps_warmup", type=int, default=10, help="FPS测试前的预热迭代数")
    parser.add_argument("--fps_iterations", type=int, default=100, help="FPS测试迭代数")
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt_path = Path(args.ckpt_path)
    config_path = Path(args.config_path)
    train_cfg_path = Path(args.train_cfg_path)
    train_root = Path(args.train_root)
    split_file = Path(args.split_file)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not train_cfg_path.exists():
        raise FileNotFoundError(f"Train config not found: {train_cfg_path}")
    if not train_root.exists():
        raise FileNotFoundError(f"Train root not found: {train_root}")
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    config = json.load(open(config_path))
    train_cfg = json.load(open(train_cfg_path))

    num_bins = (
        args.num_bins
        if args.num_bins is not None
        else train_cfg.get("num_bins", config["data_loader"]["train"]["args"].get("num_voxel_bins", 5))
    )
    flow_scale = args.flow_scale if args.flow_scale is not None else train_cfg.get("flow_scale", 1.0)
    step_scale = args.step_scale if args.step_scale is not None else train_cfg.get("step_scale", 1.0)

    model_args = SimpleNamespace(
        model_name="TRM",
        num_bins=num_bins,
        use_cdc=True,
        use_ssm=train_cfg.get("use_ssm", False),
        use_temporal_ssm=train_cfg.get("use_temporal_ssm", False),
        ssm_state_dim=train_cfg.get("ssm_state_dim", 64),
        temporal_state_dim=train_cfg.get("temporal_state_dim", 64),
        blend_weight=train_cfg.get("blend_weight", 0.3),
        ssm_dropout=train_cfg.get("ssm_dropout", 0.0),
        h2_reg_weight=train_cfg.get("h2_reg_weight", 0.0),
        step_scale=step_scale,
        best_epe=1e5,
    )

    config["data_loader"]["train"]["args"]["num_voxel_bins"] = num_bins
    config["data_loader"]["test"]["args"]["num_voxel_bins"] = num_bins

    print("=== Build model ===")
    model = build_model(model_args, config)
    if hasattr(model, "change_imagesize"):
        model.change_imagesize((480, 640))

    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("=== Load dataset (test_split) ===")
    val_set = DSECEventFlow(
        root_dir=str(train_root),
        num_bins=num_bins,
        sequence_length=2,
        flow_scale=flow_scale,
        train=False,
        split_file=str(split_file),
    )
    loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(
        f"Dataset: {len(val_set)} pairs | seq={len(val_set.sequences)} | "
        f"bins={num_bins} | flow_scale={flow_scale} | step_scale={step_scale} | device={device}"
    )

    acc = {k: 0.0 for k in ("epe", "p1", "p3", "ae")}
    acc["count"] = 0

    if args.test_fps:
        print("\n=== FPS 测试模式 ===")
        print(f"预热迭代: {args.fps_warmup}")
        print(f"测试迭代: {args.fps_iterations}")

        test_batch = next(iter(loader))
        events1 = test_batch["event_volume_old"].to(device, non_blocking=True)
        events2 = test_batch["event_volume_new"].to(device, non_blocking=True)

        print("预热中...")
        for _ in range(args.fps_warmup):
            with torch.no_grad():
                _ = model(events1, events2, step_scale=step_scale)

        if device.type == "cuda":
            torch.cuda.synchronize()

        print("开始FPS测试...")
        start_time = time.time()

        for _ in range(args.fps_iterations):
            with torch.no_grad():
                _ = model(events1, events2, step_scale=step_scale)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()
        total_time = end_time - start_time
        fps = args.fps_iterations / total_time
        avg_time_ms = (total_time / args.fps_iterations) * 1000

        print("\n===== FPS 测试结果 =====")
        print(f"总时间: {total_time:.4f} 秒")
        print(f"平均推理时间: {avg_time_ms:.4f} ms/帧")
        print(f"FPS: {fps:.2f}")
        print(f"Batch size: {args.batch_size}")
        print(f"Input shape: {events1.shape}")
        print(f"Device: {device}")
        return

    for batch in loader:
        with torch.no_grad():
            events1 = batch["event_volume_old"].to(device, non_blocking=True)
            events2 = batch["event_volume_new"].to(device, non_blocking=True)
            flow_gt = batch["flow"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)

            _, flow_list = model(events1, events2, step_scale=step_scale)
            flow_pred = flow_list[-1] if isinstance(flow_list, (list, tuple)) else flow_list

            metrics = compute_basic_metrics(flow_pred, flow_gt, valid)
            for k in ("epe", "p1", "p3", "ae"):
                acc[k] += metrics[k] * metrics["count"]
            acc["count"] += metrics["count"]

    if acc["count"] == 0:
        print("No valid pixels found.")
        return

    epe_mean = acc["epe"] / acc["count"]
    p1 = acc["p1"] / acc["count"]
    p3 = acc["p3"] / acc["count"]
    ae_mean = acc["ae"] / acc["count"]

    print("\n===== DSEC Eval (test_split) =====")
    print(f"Samples: {len(val_set)} | Pixels: {acc['count']}")
    print(f"EPE mean: {epe_mean:.4f}")
    print(f"AE  mean: {ae_mean:.4f}")
    print(f"1PE (<1px): {p1*100:.2f}%")
    print(f"3PE (<3px): {p3*100:.2f}%")


if __name__ == "__main__":
    main()
