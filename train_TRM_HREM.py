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

import git
import torch.nn

def get_visualizer(args):
    return visualization.FlowVisualizerEvents
        
def train(args):
    config_path = 'config/a_meshflow.json'
    config = json.load(open(config_path))

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
        input_dim = config['data_loader']['train']['args']['num_voxel_bins']
        model = build_flowformer(cfg, input_dim=input_dim)
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
            print("[Train] 使用 TRM_cdc 模型")
        else:
            from model.EEMFlow.EEMFlow import EEMFlow
            model = EEMFlow(config=config, n_first_channels=5, out_mesh_size=True)
            print(f"[Train] 使用 TRM baseline 模型")

    config = json.load(open(config_path))
    config["train"]["lr"] = args.lr
    config["train"]["wdecay"] = args.wd
    config["train"]["num_steps"] =  args.train_iters
    config['data_loader']['train']['args']['batch_size'] = args.batch_size

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))

    if hasattr(args, 'checkpoint_path') and args.checkpoint_path and os.path.exists(args.checkpoint_path):
        checkpoint_dir = os.path.dirname(args.checkpoint_path)
        if os.path.isabs(checkpoint_dir):
            save_path = checkpoint_dir
        else:
            save_path = os.path.join(proc_path, checkpoint_dir)
        print(f"[Resume] 使用检查点目录作为保存路径: {save_path}")
        config_json_path = os.path.join(save_path, 'config.json')
        if os.path.exists(config_json_path):
            saved_config = json.load(open(config_json_path))
            if 'name' in saved_config:
                config['name'] = saved_config['name']
    else:
        base_name = "bs{}_lr{:.0e}_wd{:.0e}".format(args.batch_size, args.lr, args.wd)
        
        from datetime import datetime
        if hasattr(args, 'output_dir') and args.output_dir:
            config['name'] = args.output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if hasattr(args, 'use_ssm') and args.use_ssm:
                ssm_suffix = "_ssm"
                if hasattr(args, 'step_scale'):
                    ssm_suffix += f"_ss{args.step_scale}"
                if hasattr(args, 'ssm_state_dim'):
                    ssm_suffix += f"_sd{args.ssm_state_dim}"
                config['name'] = f"{base_name}{ssm_suffix}_{timestamp}"
            else:
                config['name'] = f"{base_name}_{timestamp}"
        
        if hasattr(args, 'output_dir') and args.output_dir:
            save_path = os.path.join(proc_path, "exp_HREM_meshflow", config['name'].lower())
        else:
            save_path = os.path.join(proc_path, "exp_HREM_meshflow/{}_{}".format(args.model_name, args.input_type), config['name'].lower())
        os.makedirs(save_path, exist_ok=True)

    config["data_loader"]["train"]["args"].update({'type':'train', 'event_interval':args.input_type})

    print('Storing output in folder {}'.format(save_path))
    json.dump(config, open(os.path.join(save_path, 'config.json'), 'w'),
              indent=4, sort_keys=False)

    checkpoint_path = None
    if hasattr(args, 'checkpoint_path') and args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        checkpoint_path = os.path.join(save_path, 'lasted_ckpt.pth.tar')
    
    best_val_aee = getattr(args, 'best_epe', 1e5)

    if args.start_epoch and os.path.exists(checkpoint_path):
        print(f"[Resume] 从检查点加载: {checkpoint_path}")
        states = torch.load(checkpoint_path)
        state_dict = {}
        for key, param in states['state_dict'].items():
            state_dict.update({key.replace('module.',''): param})
        model.load_state_dict(state_dict)

        start_epoch = states['epoch'] + 1
        if 'best_val_aee' in states:
            best_val_aee = states['best_val_aee']
            print(f"[Resume] 历史最佳 Val AEE = {best_val_aee:.6f}")
        print(f"[Resume] 从检查点恢复训练，从第 {start_epoch} 个 epoch 开始")
    else:
        start_epoch = 0

    train_logger = Logger(save_path, custom_name='train.log')
    train_logger.initialize_file("train")

    if hasattr(args, 'use_ssm') and args.use_ssm:
        model_config = f"[Model Config] Using SSM with step_scale={getattr(args, 'step_scale', 1.0)}, "
        model_config += f"ssm_state_dim={getattr(args, 'ssm_state_dim', 64)}, "
        model_config += f"blend_weight={getattr(args, 'blend_weight', 0.3)}, "
        model_config += f"h2_reg_weight={getattr(args, 'h2_reg_weight', 0.0)}"
        train_logger.write_line(model_config, verbose=True)
    
    train_set = HREMEventFlow(
        args = config["data_loader"]["train"]["args"],
        train = True
    )
    train_set_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.num_workers,pin_memory=True)

    val_input_type = args.val_input_type if args.val_input_type else args.input_type
    print(f"[Train] 创建验证集... (数据类型: {val_input_type})")

    val_config_args = config["data_loader"]["train"]["args"].copy()
    val_config_args['event_interval'] = val_input_type

    val_set = HREMEventFlow(
        args = val_config_args,
        train = False
    )
    val_set_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"[Train] 验证集大小: {len(val_set)} 样本")

    visualizer = get_visualizer(args)

    train = TrainRaftEvents(
        model=model,
        config=config,
        args=args,
        data_loader=train_set_loader,
        train_logger=train_logger,
        save_path=save_path,
        visualizer=visualizer,
        visualizer_map=True
    )


    model.to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        print(f"[Train] 使用 {torch.cuda.device_count()} 个 GPU 进行并行训练")
    else:
        print(f"[Train] 使用 CPU 训练（警告：训练速度会非常慢）")
    train.fetch_optimizer(model)

    total_epochs = args.train_iters // args.val_iters
    remaining_iters = args.train_iters % args.val_iters
    
    completed_iters = start_epoch * args.val_iters
    remaining_total_iters = args.train_iters - completed_iters
    
    print(f"[Train] 总迭代数: {args.train_iters}, 已完成: {completed_iters}, 剩余: {remaining_total_iters}")
    print(f"[Train] 总epoch数: {total_epochs}, 当前epoch: {start_epoch}")

    for epoch in range(start_epoch, total_epochs + (1 if remaining_iters > 0 else 0)):
        train.summary()

        if epoch == total_epochs and remaining_iters > 0:
            current_val_iters = remaining_iters
            print(f"[Train] 最后一个epoch，训练 {current_val_iters} 个迭代")
        else:
            current_val_iters = args.val_iters

        model = train.train_iters(model, start_epoch=epoch, val_iters=current_val_iters)
        val_step_scale = args.val_step_scale if args.val_step_scale is not None else args.step_scale
        val_input_type_str = args.val_input_type if args.val_input_type else args.input_type
        print(f"\n[Validation] Epoch {epoch+1}: 在 {val_input_type_str} 验证集上评估 (step_scale={val_step_scale})...")
        model.eval()
        val_aee_sum = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_set_loader):
                batch = train.move_batch_to_cuda(batch)
                train.run_network(model, batch, step_scale=val_step_scale)
                f_est, f_flow_mask = train.get_estimation_and_target(batch)
                f_gt = f_flow_mask[0]
                f_mask = f_flow_mask[1]
                _, metrics = train.sequence_loss(f_est, f_gt, f_mask, train.gamma)
                val_aee_sum += metrics['epe'] * f_gt.shape[0]
                val_samples += f_gt.shape[0]

        val_aee = val_aee_sum / val_samples if val_samples > 0 else 0.0
        val_log = f"[Validation] Epoch {epoch+1}: Val AEE ({val_input_type_str}) = {val_aee:.6f}"
        print(val_log)
        train_logger.write_line(val_log, verbose=True)

        model.train()

        save_dict = {
            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'val_aee': val_aee,
            'best_val_aee': best_val_aee,
        }
        torch.save(save_dict, os.path.join(save_path, 'lasted_ckpt.pth.tar'))

        if val_aee < best_val_aee:
            best_val_aee = val_aee
            best_log = f"[Checkpoint] New best AEE {best_val_aee:.6f} at epoch {epoch+1}, saved best_ckpt.pth.tar"
            print(best_log)
            train_logger.write_line(best_log, verbose=True)
            torch.save(save_dict, os.path.join(save_path, 'best_ckpt.pth.tar'))
        
    print(f"[Train] 训练完成！总共训练了 {args.train_iters} 个迭代")
    return

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--visualize', action='store_true', help='是否可视化结果（DSEC数据集）。MVSEC实验始终可视化。')
    parser.add_argument('-n', '--num_workers', default=0, type=int, help='数据加载使用的子进程数量')
    parser.add_argument('--train_iters', default=50000, type=int, metavar='N', help='总训练迭代次数')
    parser.add_argument('-se','--start-epoch', action='store_true', help='是否从检查点重新开始训练')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='检查点文件路径（如果指定，将从该路径加载）')
    parser.add_argument('-be','--best_epe', default=1e5, type=float, help='最佳端点误差（EPE）')
    parser.add_argument('--val_iters', default=2000, type=int, metavar='N',help='验证间隔（每隔多少次迭代进行一次验证）')
    parser.add_argument('--lr', default=2.5e-4, type=float, help='学习率')
    parser.add_argument('--wd', default=5e-5, type=float, help='权重衰减（L2正则化）')

    parser.add_argument('--batch_size', '-bs', default=16, type=int, help='训练批次大小')
    parser.add_argument('--test_only', action='store_true', help='仅进行测试（不训练）')
    parser.add_argument('--test_sequence', '-sq', default='indoor_flying2', type=str, help='测试序列名称')
    parser.add_argument('--dense', action='store_true', help='是否使用稠密模式')
    parser.add_argument('--density', '-d', default='ct0.05', type=str, help='密度设置')
    parser.add_argument('--model_name', '-model', default='TRM', type=str, help='模型名称（eraft/kpaflow/GMA/flowformer/skflow/irrpwc/TRM）')
    parser.add_argument('--input_type', '-int', default='dt4', type=str, help='输入事件间隔类型')
    parser.add_argument('--val_input_type', type=str, default=None, help='验证集事件间隔类型（默认与input_type相同，设置为不同值可进行跨频率验证）')
    parser.add_argument('--val_step_scale', type=float, default=2.0, help='验证时的step_scale（跨频率验证用，默认与训练相同）')
    parser.add_argument('--use_cdc', action='store_true', help='使用 TRM_cdc 版本（支持 SSM）')
    parser.add_argument('--use_temporal_ssm', action='store_true', help='启用时间SSM（建模15个bins的时序依赖）')
    parser.add_argument('--temporal_state_dim', type=int, default=32, help='时间SSM状态维度（推荐：dt1=16-32, dt4=32-64）')
    parser.add_argument('--use_ssm', action='store_true', help='在 CDC 模块中启用空间 SSM Refiner')
    parser.add_argument('--ssm_state_dim', type=int, default=64, help='空间SSM状态维度（论文推荐：60Hz输入64-128，15Hz输入32-64）')
    parser.add_argument('--step_scale', type=float, default=2.0, help='SSM step_scale（dt1=1.0, dt4=4.0, 训练和推理保持一致）')
    parser.add_argument('--blend_weight', type=float, default=0.3, help='SSM 混合权重（0.3-0.7）')
    parser.add_argument('--ssm_dropout', type=float, default=0.0, help='SSM Dropout 概率')
    parser.add_argument('--h2_reg_weight', type=float, default=0.0, help='H2 正则化权重（0 关闭）')
    parser.add_argument('--temporal_debug_iters', type=int, default=0, help='Temporal SSM 调试打印次数（仅前若干次 forward 打印 mean/std）')
    parser.add_argument('--output_dir', '-o', type=str, default=None, help='自定义输出目录名称（如果不指定，则使用时间戳自动生成）')
    args = parser.parse_args()
    train(args)
