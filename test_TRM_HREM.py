import os,sys
current_path = os.path.dirname(os.path.abspath(__file__))
proc_path = current_path
sys.path.append(current_path)
sys.path.append(proc_path)
from loader.HREM import HREMEventFlow
from test_mvsec import *
from train_mvsec import *

from utils.logger import *
import utils.helper_functions as helper
import json
from torch.utils.data.dataloader import DataLoader
from utils import visualization as visualization
import argparse
import time

import git
import torch.nn

DT_MS = {
    "dt1": 50.0,
    "dt4": 200.0,
}


def compute_step_scale(train_input_type: str, test_input_type: str, fallback: float) -> float:
    if train_input_type in DT_MS and test_input_type in DT_MS:
        scale = DT_MS[test_input_type] / DT_MS[train_input_type]
        print(f"[StepScale] train={train_input_type}, test={test_input_type}, auto step_scale={scale}", flush=True)
        return scale
    print(f"[StepScale] 未识别的频率标识，使用回退 step_scale={fallback}", flush=True)
    return fallback


def get_visualizer(args):
        return visualization.FlowVisualizerEvents
        
def _format_param_count(num_params: int) -> str:
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.2f} B"
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f} M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.2f} K"
    return str(num_params)

def _report_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n=== 模型参数量 ===")
    print(f"总参数量: {total_params} ({_format_param_count(total_params)})")
    print(f"可训练参数量: {trainable_params} ({_format_param_count(trainable_params)})")

def _run_fps_test(model, device, data_loader, warmup_iters: int, test_iters: int):
    print("\n=== FPS 测试模式 ===")
    print(f"预热迭代: {warmup_iters}")
    print(f"测试迭代: {test_iters}")

    test_batch = next(iter(data_loader))
    events1 = test_batch["event_volume_old"].to(device, non_blocking=True)
    events2 = test_batch["event_volume_new"].to(device, non_blocking=True)

    print("预热中...")
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(events1=events1, events2=events2)

    if device.type == "cuda":
        torch.cuda.synchronize()

    print("开始FPS测试...")
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(test_iters):
            _ = model(events1=events1, events2=events2)

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_time = time.perf_counter() - start_time
    fps = test_iters / total_time if total_time > 0 else 0.0
    avg_time_ms = (total_time / test_iters) * 1000 if test_iters > 0 else 0.0

    print("\n===== FPS 测试结果 =====")
    print(f"总时间: {total_time:.4f} 秒")
    print(f"平均推理时间: {avg_time_ms:.4f} ms/帧")
    print(f"FPS: {fps:.2f}")
    print(f"Batch size: {events1.shape[0]}")
    print(f"Input shape: {events1.shape}")
    print(f"Device: {device}")

def train(args):

    config_path = 'config/a_meshflow.json'

    config = json.load(open(config_path))

    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
        print("Loading checkpoint from: {}".format(checkpoint_path))
    else:
        checkpoint_path = os.path.join(proc_path, 'checkpoints', 'TRM_HREM_{}.pth.tar'.format(args.input_type))
        print("Loading default checkpoint from: {}".format(checkpoint_path))

    states = torch.load(checkpoint_path, map_location='cpu')
    raw_state_dict = states['state_dict'] if isinstance(states, dict) and 'state_dict' in states else states

    if hasattr(args, 'use_cdc') and args.use_cdc:
        has_temporal_ssm = any(k.startswith('temporal_ssm.') for k in raw_state_dict.keys())
        if has_temporal_ssm and not getattr(args, 'use_temporal_ssm', False):
            print("[Compat] 检测到 temporal_ssm 权重，自动开启 --use_temporal_ssm 以匹配检查点")
            args.use_temporal_ssm = True
        elif getattr(args, 'use_temporal_ssm', False) and not has_temporal_ssm:
            print("[Warning] 启用了 use_temporal_ssm，但检查点中不存在 temporal_ssm 权重，可能导致加载失败")

    if hasattr(args, 'use_ssm') and (args.use_ssm or getattr(args, 'use_temporal_ssm', False)):
        train_input = getattr(args, 'train_input_type', None) or args.input_type
        args.step_scale = compute_step_scale(train_input, args.input_type, getattr(args, 'step_scale', 1.0))
    
    if(args.model_name == "eraft"):
        from model.eraft import ERAFT as RAFT
        model = RAFT(config=config)
    elif(args.model_name == "kpaflow"):
        from model.KPAflow.KPAFlow import KPAFlow
        model = KPAFlow(config=config)
    elif(args.model_name == "GMA"):
        from model.GMA.network import RAFTGMA
        model = RAFTGMA(
            config=config,
            n_first_channels=config['data_loader']['train']['args']['num_voxel_bins']
        )
    elif(args.model_name == "flowformer"):
        from model.flowformer.FlowFormer import build_flowformer
        from model.flowformer.config import get_cfg
        cfg = get_cfg()
        model = build_flowformer(cfg)
    elif(args.model_name == "skflow"):
        from model.SKflow.models.sk_decoder import SK_Decoder
        model = SK_Decoder(config=config)
    elif(args.model_name == "irrpwc"):
        from model.IRRPWC.pwcnet_irr import PWCNet
        model = PWCNet(config=config)
    elif(args.model_name in ("TRM", "EEMFlow")):
        if hasattr(args, 'use_cdc') and args.use_cdc:
            import importlib.util
            eemflow_plus_path = os.path.join(proc_path, 'model', 'EEMFlow', 'EEMFlow+.py')
            spec = importlib.util.spec_from_file_location("EEMFlow_plus", eemflow_plus_path)
            eemflow_plus = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(eemflow_plus)
            EEMFlow_cdc = eemflow_plus.EEMFlow_cdc
            model = EEMFlow_cdc(config=config, n_first_channels=5, args=args)
            print("[Test] 使用 TRM_cdc 模型")
        else:
            from model.EEMFlow.EEMFlow import EEMFlow
            model = EEMFlow(config=config, n_first_channels=5)
            print(f"[Test] 使用 TRM baseline 模型")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))

    state_dict = {}
    use_ssm = getattr(args, 'use_ssm', False)
    use_temporal_ssm = getattr(args, 'use_temporal_ssm', False)
    
    filtered_keys = []
    for key, param in raw_state_dict.items():
        clean_key = key.replace('module.', '')
        
        if clean_key.startswith('cdc_model.ssm_refiner.') and not use_ssm:
            filtered_keys.append(clean_key)
            continue
        
        if clean_key.startswith('temporal_ssm.') and not use_temporal_ssm:
            filtered_keys.append(clean_key)
            continue
        
        state_dict[clean_key] = param
    
    if filtered_keys:
        print(f"[Compat] 过滤掉 {len(filtered_keys)} 个不需要的权重键（当前配置未启用对应模块）")
        if len(filtered_keys) <= 10:
            for fk in filtered_keys:
                print(f"  - {fk}")
        else:
            for fk in filtered_keys[:5]:
                print(f"  - {fk}")
            print(f"  ... 还有 {len(filtered_keys) - 5} 个")
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"[Warning] 模型中有 {len(missing_keys)} 个权重未从检查点加载")
        if len(missing_keys) <= 5:
            for mk in missing_keys:
                print(f"  - {mk}")
    if unexpected_keys:
        print(f"[Warning] 检查点中有 {len(unexpected_keys)} 个权重未使用")
        if len(unexpected_keys) <= 5:
            for uk in unexpected_keys:
                print(f"  - {uk}")

    start_epoch = states['epoch'] if isinstance(states, dict) and 'epoch' in states else 0
    print("Loaded checkpoint from epoch: {}".format(start_epoch))

    if args.output_dir:
        save_path = os.path.join(current_path, "test_output/{}".format(args.output_dir))
    else:
        save_path = os.path.join(current_path, "test_output/{}_{}".format(args.model_name, args.input_type))


    os.makedirs(save_path, exist_ok=True)

    config["data_loader"]["test"]["args"].update({"event_interval":args.input_type})

    print('Storing output in folder {}'.format(save_path))
    json.dump(config, open(os.path.join(save_path, 'config.json'), 'w'),
              indent=4, sort_keys=False)
    test_logger = Logger(save_path, custom_name='test.log')
    test_logger.initialize_file("test")

    test_set = HREMEventFlow(
        args = config["data_loader"]["test"]["args"],
        train = False
    )

    test_set_loader = DataLoader(test_set,
                                 batch_size=config['data_loader']['test']['args']['batch_size'],
                                 shuffle=config['data_loader']['test']['args']['shuffle'],
                                 num_workers=args.num_workers,
                                 pin_memory=True,
                                 drop_last=True)
    visualizer = get_visualizer(args)

    test = TestRaftEvents(
        model=model,
        config=config,
        data_loader=test_set_loader,
        test_logger=test_logger,
        save_path=save_path,
        visualizer=visualizer,
        visualizer_map=True,
        save_excel=True
    )

    model.to(device)
    model.eval()

    _report_model_params(model)

    if args.test_fps:
        _run_fps_test(
            model=model,
            device=device,
            data_loader=test_set_loader,
            warmup_iters=args.fps_warmup,
            test_iters=args.fps_iterations,
        )

    available_sequences = list(test_set_loader.dataset.nori_list.keys())
    if args.test_sequences:
        sequence_list = [seq for seq in args.test_sequences if seq in available_sequences]
        if not sequence_list:
            print(f"⚠️  警告: 指定的序列 {args.test_sequences} 都不存在，将评估所有可用序列")
            sequence_list = available_sequences
        else:
            print(f"📋 将评估指定序列: {sequence_list}")
    else:
        sequence_list = available_sequences
        print(f"📋 将评估所有可用序列: {sequence_list}")
    
    test.test_multi_sequence(
        model,
        start_epoch + 1,
        sequence_list=sequence_list,
        stride=1,
        visualize_map=args.visualize,
        print_epe=True,
        vis_events=args.vis_events
    )

    return

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--visualize', action='store_true', help='可视化（主要用于 DSEC）；HREM 默认可不加')
    parser.add_argument('-n', '--num_workers', default=0, type=int, help='DataLoader 进程数，测试建议 0 避免多进程开销')
    parser.add_argument('--train_iters', default=1000000, type=int, metavar='N', help='占位参数，测试可忽略')
    parser.add_argument('-se','--start-epoch', action='store_true', help='占位参数，测试可忽略')

    parser.add_argument('--val_iters', default=3000, type=int, metavar='N',help='占位参数，测试可忽略')
    parser.add_argument('--lr', default=1e-4, type=float, help='占位参数，测试可忽略')
    parser.add_argument('--wd', default=1e-5, type=float, help='占位参数，测试可忽略')

    parser.add_argument('--batch_size', '-bs', default=2, type=int, help='测试 batch 大小，HREM 数据集通常用 1-2')

    parser.add_argument('--test_only', action='store_true', help='占位参数，测试可忽略')
    parser.add_argument('--test_sequence', '-sq', default='', type=str, help='占位参数；真正的序列过滤用 --test_sequences')

    parser.add_argument('--model_name', '-model', default='TRM', type=str, help='保持 TRM')
    parser.add_argument('--input_type', '-int', default='dt1', type=str, help='测试事件窗口 dt1/dt4')
    parser.add_argument('--train_input_type', '-train_int', default=None, type=str, help='模型训练用的频率；用于自动计算 step_scale=dt_test/dt_train')
    parser.add_argument('--is_using_dynamic', '-dynamic', action='store_true', help='占位参数，测试可忽略')
    parser.add_argument('--checkpoint_path', '-ckpt', default=None, type=str, help='权重路径，必填')
    parser.add_argument('--output_dir', '-o', default=None, type=str, help='输出目录名（位于 test_output/ 下）')
    parser.add_argument('--vis_events', action='store_true', help='可视化事件图（便于查看场景内容）')
    parser.add_argument('--use_cdc', action='store_true', help='使用 TRM_cdc（含 CDC/SSM 分支），一般要开启')
    parser.add_argument('--use_temporal_ssm', action='store_true', help='启用时间 SSM；若 checkpoint 含 temporal_ssm.* 权重需开启')
    parser.add_argument('--temporal_state_dim', type=int, default=32, help='时间 SSM 状态维度（与训练保持一致）')
    parser.add_argument('--temporal_debug_iters', type=int, default=0, help='前若干次 forward 打印 temporal SSM 统计，0 关闭')
    parser.add_argument('--use_ssm', action='store_true', help='启用空间 SSM Refiner（与 ssm 权重匹配时需开启）')
    parser.add_argument('--step_scale', type=float, default=1.0, help='若提供 train_input_type，会自动覆盖为 dt_test/dt_train')
    parser.add_argument('--ssm_state_dim', type=int, default=64, help='空间 SSM 维度（与训练一致）')
    parser.add_argument('--blend_weight', type=float, default=0.3, help='SSM 混合权重')
    parser.add_argument('--ssm_dropout', type=float, default=0.1, help='SSM Dropout 概率')
    parser.add_argument('--test_sequences', '-seq', nargs='+', default=None, help='只评估指定序列，留空评估全部')
    parser.add_argument('--test_fps', action='store_true', help='测试推理 FPS（与 DSEC eval 输出风格一致）')
    parser.add_argument('--fps_warmup', type=int, default=10, help='FPS 测试预热迭代数')
    parser.add_argument('--fps_iterations', type=int, default=100, help='FPS 测试迭代数')
    args = parser.parse_args()
    train(args)
