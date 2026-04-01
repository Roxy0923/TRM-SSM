import os
import sys
import json
from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader

from loader.DSEC import DSECEventFlow
from test_mvsec import *
from train_mvsec import *
from utils.logger import *

proc_path = Path(__file__).resolve().parent
repo_root = proc_path


def get_visualizer(args):
    return visualization.FlowVisualizerEvents


def compute_basic_metrics(flow_pred, flow_gt, valid_mask):
    valid = valid_mask.bool()
    if valid.sum() == 0:
        return {"epe": 0.0, "p1": 0.0, "p3": 0.0, "ae": 0.0, "count": 0}

    gt_norm = torch.norm(flow_gt, dim=1)
    finite = torch.isfinite(flow_gt).all(dim=1)
    mask = valid & finite & (gt_norm > 0)
    if mask.sum() == 0:
        return {"epe": 0.0, "p1": 0.0, "p3": 0.0, "ae": 0.0, "count": 0}

    diff = flow_pred - flow_gt
    epe_map = torch.norm(diff, dim=1)
    epe_valid = epe_map[mask]
    gt_norm_valid = gt_norm[mask]
    epe_mean = epe_valid.mean().item()

    p1 = (epe_valid < 1.0).float().mean().item()
    p3 = ((epe_valid < 3.0) | (epe_valid < 0.1 * gt_norm_valid)).float().mean().item()

    gt_norm_safe = gt_norm + 1e-8
    pred_norm_safe = torch.norm(flow_pred, dim=1) + 1e-8
    cos_sim = ((flow_pred * flow_gt).sum(dim=1) / (gt_norm_safe * pred_norm_safe)).clamp(-1.0, 1.0)
    ae_map = torch.acos(cos_sim) * 180.0 / torch.pi
    ae_mean = ae_map[mask].mean().item()

    return {
        "epe": epe_mean,
        "p1": p1,
        "p3": p3,
        "ae": ae_mean,
        "count": int(mask.sum().item()),
    }


def build_model(args, config):
    if args.model_name == "eraft":
        from model.eraft import ERAFT as RAFT
        model = RAFT(config=config)
    elif args.model_name == "kpaflow":
        from model.KPAflow.KPAFlow import KPAFlow
        model = KPAFlow(config=config)
    elif args.model_name == "GMA":
        from model.GMA.network import RAFTGMA
        model = RAFTGMA(
            config=config,
            n_first_channels=config['data_loader']['train']['args']['num_voxel_bins']
        )
    elif args.model_name == "flowformer":
        from model.flowformer.FlowFormer import build_flowformer
        from model.flowformer.config import get_cfg
        cfg = get_cfg()
        input_dim = config['data_loader']['train']['args']['num_voxel_bins']
        model = build_flowformer(cfg, input_dim=input_dim)
    elif args.model_name == "skflow":
        from model.SKflow.models.sk_decoder import SK_Decoder
        model = SK_Decoder(config=config)
    elif args.model_name == "irrpwc":
        from model.IRRPWC.pwcnet_irr import PWCNet
        model = PWCNet(config=config)
    elif args.model_name in ("TRM", "EEMFlow"):
        if hasattr(args, 'use_cdc') and args.use_cdc:
            import importlib.util
            eemflow_plus_path = os.path.join(proc_path, 'model', 'EEMFlow', 'EEMFlow+.py')
            spec = importlib.util.spec_from_file_location("EEMFlow_plus", eemflow_plus_path)
            eemflow_plus = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(eemflow_plus)
            EEMFlow_cdc = eemflow_plus.EEMFlow_cdc
            model = EEMFlow_cdc(config=config, n_first_channels=args.num_bins, args=args)
            print(f"[Train] 使用 TRM_cdc 模型 (n_first_channels={args.num_bins})")
        else:
            from model.EEMFlow.EEMFlow import EEMFlow
            model = EEMFlow(config=config, n_first_channels=args.num_bins, out_mesh_size=True)
            print(f"[Train] 使用 TRM baseline 模型 (n_first_channels={args.num_bins})")
    else:
        raise ValueError(f"未知模型 {args.model_name}")
    return model


def train(args):
    config_path = os.path.join(proc_path, 'config', 'a_meshflow.json')
    config = json.load(open(config_path))

    config["train"]["lr"] = args.lr
    config["train"]["wdecay"] = args.wd
    config["train"]["num_steps"] = args.train_iters
    config['data_loader']['train']['args']['batch_size'] = args.batch_size
    config['data_loader']['train']['args']['num_voxel_bins'] = args.num_bins
    config['data_loader']['test']['args']['num_voxel_bins'] = args.num_bins

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    save_name = f"bs{args.batch_size}_lr{args.lr:.0e}_wd{args.wd:.0e}"
    if args.output_dir:
        save_name = args.output_dir
    save_path = os.path.join(proc_path, "exp_DSEC_meshflow", save_name)
    os.makedirs(save_path, exist_ok=True)
    print(f"Storing output in folder {save_path}")
    json.dump(config, open(os.path.join(save_path, 'config.json'), 'w'), indent=4, sort_keys=False)

    model = build_model(args, config)
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    train_set = DSECEventFlow(
        root_dir=args.dataset_root,
        num_bins=args.num_bins,
        sequence_length=2,
        flow_scale=args.flow_scale,
        train=True,
        split_file=getattr(args, 'train_split_file', None)
    )
    val_set = DSECEventFlow(
        root_dir=args.dataset_root_val,
        num_bins=args.num_bins,
        sequence_length=2,
        flow_scale=args.flow_scale,
        train=False,
        split_file=getattr(args, 'val_split_file', None)
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    print(f"[Train] 训练样本: {len(train_set)}, 验证样本: {len(val_set)}")

    train_logger = Logger(save_path, custom_name='train.log')
    train_logger.initialize_file("train")
    visualizer = get_visualizer(args)

    trainer = TrainRaftEvents(
        model=model,
        config=config,
        args=args,
        data_loader=train_loader,
        train_logger=train_logger,
        save_path=save_path,
        visualizer=visualizer,
        visualizer_map=False
    )

    trainer.fetch_optimizer(model)

    total_epochs = args.train_iters // args.val_iters
    remaining_iters = args.train_iters % args.val_iters
    print(f"[Train] 总迭代数: {args.train_iters}, 验证间隔: {args.val_iters}, 总 epoch: {total_epochs}")

    best_val = 1e5
    global_iter = 0

    for epoch in range(total_epochs + (1 if remaining_iters > 0 else 0)):
        current_val_iters = remaining_iters if (epoch == total_epochs and remaining_iters > 0) else args.val_iters
        trainer.summary()
        model = trainer.train_iters(model, start_epoch=epoch, val_iters=current_val_iters)
        global_iter += current_val_iters

        model.eval()
        acc = {"epe": 0.0, "p1": 0.0, "p3": 0.0, "ae": 0.0, "count": 0}
        with torch.no_grad():
            for b_idx, batch in enumerate(val_loader):
                batch = trainer.move_batch_to_cuda(batch)
                trainer.run_network(model, batch, step_scale=args.step_scale)
                f_est, f_flow_mask = trainer.get_estimation_and_target(batch)
                if isinstance(f_est, (list, tuple)):
                    f_est = f_est[-1]
                f_gt = f_flow_mask[0]
                f_mask = f_flow_mask[1]
                m = compute_basic_metrics(f_est, f_gt, f_mask)
                acc["epe"] += m["epe"] * m["count"]
                acc["p1"] += m["p1"] * m["count"]
                acc["p3"] += m["p3"] * m["count"]
                acc["ae"] += m["ae"] * m["count"]
                acc["count"] += m["count"]
                if b_idx >= 10:
                    break
        if acc["count"] > 0:
            val_epe = acc["epe"] / acc["count"]
            val_p1 = acc["p1"] / acc["count"]
            val_p3 = acc["p3"] / acc["count"]
            val_ae = acc["ae"] / acc["count"]
        else:
            val_epe = val_p1 = val_p3 = val_ae = 0.0

        log_line = (
            f"[Val] epoch {epoch+1}, iter {global_iter}, "
            f"EPE={val_epe:.4f}, 1PE={val_p1:.4f}, 3PE={val_p3:.4f}, AE={val_ae:.4f}"
        )
        print(log_line)
        train_logger.write_line(log_line, verbose=True)
        model.train()

        save_dict = {
            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'val_epe': val_epe,
            'best_val_epe': best_val,
            'iter': global_iter,
        }
        torch.save(save_dict, os.path.join(save_path, 'lasted_ckpt.pth.tar'))
        if val_epe < best_val:
            best_val = val_epe
            torch.save(save_dict, os.path.join(save_path, 'best_ckpt.pth.tar'))
            print(f"[Checkpoint] new best {best_val:.4f}")

    print(f"[Train] 训练完成，总迭代 {global_iter}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-model', default='TRM', type=str)
    parser.add_argument('--dataset_root', type=str, default=str(repo_root / "data" / "DSEC" / "train"))
    parser.add_argument('--dataset_root_val', type=str, default=str(repo_root / "data" / "DSEC" / "train"))
    parser.add_argument('--train_split_file', type=str, default=None, help='Train split file (optional)')
    parser.add_argument('--val_split_file', type=str, default=None, help='Validation split file (optional)')
    parser.add_argument('--num_bins', type=int, default=5, help='voxel bins')
    parser.add_argument('--batch_size', '-bs', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--train_iters', type=int, default=50000)
    parser.add_argument('--val_iters', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--wd', type=float, default=5e-5)
    parser.add_argument('--flow_scale', type=float, default=1.0)
    parser.add_argument('--step_scale', type=float, default=1.0)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--best_epe', type=float, default=1e5)

    parser.add_argument('--use_cdc', action='store_true')
    parser.add_argument('--use_temporal_ssm', action='store_true')
    parser.add_argument('--temporal_state_dim', type=int, default=32)
    parser.add_argument('--use_ssm', action='store_true')
    parser.add_argument('--ssm_state_dim', type=int, default=64)
    parser.add_argument('--blend_weight', type=float, default=0.3)
    parser.add_argument('--ssm_dropout', type=float, default=0.0)
    parser.add_argument('--h2_reg_weight', type=float, default=0.01)
    parser.add_argument('--temporal_debug_iters', type=int, default=0)

    args = parser.parse_args()
    train(args)
