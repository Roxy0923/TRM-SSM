"""
工具函数：根据序列类型动态确定 step_scale

针对 Fast 序列（outdoor_fast, indoor_fast）使用更小的 step_scale，
以捕获更快速的运动并帮助模型预测更大的光流幅度。
"""

def get_step_scale_from_scene(scene_name, base_step_scale=1.0):
    """
    根据序列场景名称确定 step_scale
    
    Args:
        scene_name (str): 序列场景名称，如 'outdoor_fast', 'indoor_slow' 等
        base_step_scale (float): 基础 step_scale（根据 dt1/dt4 设置）
            - dt1 (60Hz) → 通常为 0.7
            - dt4 (15Hz) → 通常为 1.5
    
    Returns:
        float: 调整后的 step_scale
        
    策略：
        - Fast 序列（outdoor_fast, indoor_fast）: 使用更小的 step_scale（base * 0.7-0.8）
          以捕获更快速运动，帮助模型预测更大光流幅度
        - Slow 序列（outdoor_slow, indoor_slow）: 保持或略微增大 step_scale（base * 1.0-1.1）
          提供更多平滑
        - 未知序列: 返回基础值
    """
    if scene_name is None:
        return base_step_scale
    
    scene_lower = str(scene_name).lower()
    
    # Fast 序列：使用更小的 step_scale 以捕获快速运动
    if 'fast' in scene_lower:
        # 针对 Fast 序列，减小 step_scale（更细的时间分辨率）
        # 这有助于模型学习预测更大的光流幅度
        adjusted_scale = base_step_scale * 0.7  # 减小30%，更细时间分辨率
        return max(adjusted_scale, 0.4)  # 确保不小于0.4（避免过小导致不稳定）
    
    # Slow 序列：保持或略微增大 step_scale（更多平滑）
    elif 'slow' in scene_lower:
        adjusted_scale = base_step_scale * 1.1  # 增大10%，更多平滑
        return adjusted_scale
    
    # 未知序列：返回基础值
    else:
        return base_step_scale


def get_step_scale_from_batch(batch, base_step_scale=1.0, default_scene=None):
    """
    从 batch 中提取序列类型并确定 step_scale
    
    Args:
        batch: 数据批次，可能包含 'scene' 字段
        base_step_scale (float): 基础 step_scale
        default_scene (str, optional): 默认序列类型
    
    Returns:
        float or torch.Tensor: step_scale 值（如果是批次，返回 Tensor）
    """
    # 尝试从 batch 中获取 scene 信息
    if isinstance(batch, dict) and 'scene' in batch:
        scenes = batch['scene']
        
        # 如果是列表或Tensor，为每个样本计算 step_scale
        if isinstance(scenes, (list, tuple)):
            step_scales = [get_step_scale_from_scene(scene, base_step_scale) for scene in scenes]
            import torch
            return torch.tensor(step_scales, device=next(iter(batch.values())).device if hasattr(batch, 'values') else 'cpu')
        else:
            return get_step_scale_from_scene(scenes, base_step_scale)
    
    # 如果没有 scene 信息，使用默认值
    if default_scene:
        return get_step_scale_from_scene(default_scene, base_step_scale)
    
    return base_step_scale

