#!/usr/bin/env python3
"""
将 EEMFlow checkpoint 文件转换为标准 PyTorch checkpoint 格式
支持转换：
1. EEMFlow_HREM_dt1.tar -> EEMFlow_HREM_dt1.pth.tar 和 .pth
2. mvsec_dt1.pth.tar -> mvsec_dt1.pth (用于 evaluate_MVSEC_flow.py)

输出格式：
1. test_EEMFlow_HREM.py 需要的格式：{'state_dict': ..., 'epoch': ...}
2. evaluate_HREM_flow.py / evaluate_MVSEC_flow.py 需要的格式：支持 model_state_dict 或 state_dict
"""

import os
import sys
import torch

def convert_checkpoint(input_path, output_path_pth_tar=None, output_path_pth=None):
    """
    转换 checkpoint 格式
    
    Args:
        input_path: 输入的 .tar 文件路径
        output_path_pth_tar: 输出为 .pth.tar 格式（用于 test_EEMFlow_HREM.py）
        output_path_pth: 输出为 .pth 格式（用于 evaluate_HREM_flow.py）
    """
    print(f"📦 加载原始 checkpoint: {input_path}")
    
    # 加载原始文件
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    
    print(f"✅ 文件加载成功")
    print(f"   类型: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"   包含的键: {list(checkpoint.keys())[:10]}")
    
    # 提取 state_dict
    state_dict = None
    epoch = None
    
    if isinstance(checkpoint, dict):
        # 尝试不同的键名
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"✅ 找到 'state_dict'")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"✅ 找到 'model_state_dict'")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            print(f"✅ 找到 'model'")
        else:
            # 可能是直接的 state_dict
            state_dict = checkpoint
            print(f"✅ 使用整个字典作为 state_dict")
        
        # 提取 epoch
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch']
            print(f"✅ 找到 epoch: {epoch}")
        else:
            epoch = 0
            print(f"⚠️  未找到 epoch，设置为 0")
    else:
        # 如果不是字典，假设整个对象就是 state_dict
        state_dict = checkpoint
        epoch = 0
        print(f"⚠️  输入不是字典，假设整个对象是 state_dict")
    
    if state_dict is None:
        raise ValueError("无法提取 state_dict")
    
    # 处理 module. 前缀（DataParallel 添加的）
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        cleaned_state_dict[new_key] = value
    
    print(f"📊 State dict 统计:")
    print(f"   原始键数: {len(state_dict)}")
    print(f"   清理后键数: {len(cleaned_state_dict)}")
    print(f"   示例键: {list(cleaned_state_dict.keys())[:5]}")
    
    # 格式1: test_EEMFlow_HREM.py 需要的格式
    if output_path_pth_tar:
        output_dict_1 = {
            'state_dict': cleaned_state_dict,
            'epoch': epoch
        }
        print(f"\n💾 保存格式1 (test_EEMFlow_HREM.py): {output_path_pth_tar}")
        torch.save(output_dict_1, output_path_pth_tar)
        print(f"✅ 保存成功")
    
    # 格式2: evaluate_HREM_flow.py 需要的格式（支持多种键名）
    if output_path_pth:
        output_dict_2 = {
            'model_state_dict': cleaned_state_dict,  # 优先使用 model_state_dict
            'state_dict': cleaned_state_dict,  # 也提供 state_dict 作为备选
            'epoch': epoch
        }
        print(f"\n💾 保存格式2 (evaluate_HREM_flow.py): {output_path_pth}")
        torch.save(output_dict_2, output_path_pth)
        print(f"✅ 保存成功")
    
    print(f"\n🎉 转换完成！")
    return cleaned_state_dict, epoch


if __name__ == "__main__":
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 处理多个文件
    files_to_convert = [
        {
            "input": "EEMFlow_HREM_dt1.tar",
            "output_pth_tar": "EEMFlow_HREM_dt1.pth.tar",
            "output_pth": "EEMFlow_HREM_dt1.pth",
            "description": "HREM 预训练模型"
        },
        {
            "input": "mvsec_dt1.pth.tar",
            "output_pth_tar": None,  # MVSEC 不需要 .pth.tar 格式
            "output_pth": "mvsec_dt1.pth",
            "description": "MVSEC 预训练模型"
        }
    ]
    
    print("=" * 60)
    print("🔄 EEMFlow Checkpoint 格式转换工具")
    print("=" * 60)
    print()
    
    success_count = 0
    for file_info in files_to_convert:
        input_file = os.path.join(script_dir, file_info["input"])
        
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"⚠️  跳过: 找不到输入文件 {input_file}")
            continue
        
        print(f"\n{'='*60}")
        print(f"📦 处理: {file_info['description']}")
        print(f"   输入: {file_info['input']}")
        print(f"{'='*60}\n")
        
        try:
            convert_checkpoint(
                input_path=input_file,
                output_path_pth_tar=os.path.join(script_dir, file_info["output_pth_tar"]) if file_info["output_pth_tar"] else None,
                output_path_pth=os.path.join(script_dir, file_info["output_pth"]) if file_info["output_pth"] else None
            )
            success_count += 1
        except Exception as e:
            print(f"\n❌ 转换 {file_info['input']} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 打印使用说明
    if success_count > 0:
        print("\n" + "=" * 60)
        print("📝 使用说明:")
        print("=" * 60)
        
        # HREM 文件说明
        hrem_pth_tar = os.path.join(script_dir, "EEMFlow_HREM_dt1.pth.tar")
        hrem_pth = os.path.join(script_dir, "EEMFlow_HREM_dt1.pth")
        if os.path.exists(hrem_pth_tar):
            print(f"\n1. test_EEMFlow_HREM.py:")
            print(f"   使用: {hrem_pth_tar}")
            print(f"   代码: states = torch.load('{hrem_pth_tar}')")
            print(f"        state_dict = states['state_dict']")
        
        if os.path.exists(hrem_pth):
            print(f"\n2. evaluate_HREM_flow.py:")
            print(f"   使用: {hrem_pth}")
            print(f"   参数: --flow_checkpoint {hrem_pth}")
        
        # MVSEC 文件说明
        mvsec_pth = os.path.join(script_dir, "mvsec_dt1.pth")
        if os.path.exists(mvsec_pth):
            print(f"\n3. evaluate_MVSEC_flow.py:")
            print(f"   使用: {mvsec_pth}")
            print(f"   参数: --flow_checkpoint {mvsec_pth}")
        
        print("\n" + "=" * 60)
        print(f"✅ 成功转换 {success_count} 个文件")
        print("=" * 60)
    else:
        print("\n❌ 没有成功转换任何文件")
        sys.exit(1)

