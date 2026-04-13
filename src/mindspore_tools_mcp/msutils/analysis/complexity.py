"""
msutils.analysis.complexity - 模型复杂度分析

分析模型的参数量、计算量、内存占用等
"""

import numpy as np
from typing import Dict, Tuple, List


def count_parameters(model) -> Dict[str, int]:
    """
    统计模型参数量
    
    Args:
        model: MindSpore 模型
    
    Returns:
        参数量统计字典
    
    Example:
        >>> from msutils.analysis import count_parameters
        >>> stats = count_parameters(model)
        >>> print(f"Total params: {stats['total']:,}")
    """
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    
    for param in model.trainable_params():
        num_params = param.size
        total_params += num_params
        trainable_params += num_params
    
    for param in model.parameters_dict().values():
        if param not in model.trainable_params():
            non_trainable_params += param.size
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params
    }


def measure_flops(model, input_shape: Tuple = (1, 3, 224, 224)) -> Dict[str, float]:
    """
    估算模型 FLOPs
    
    Args:
        model: MindSpore 模型
        input_shape: 输入形状
    
    Returns:
        FLOPs 统计字典
    
    Example:
        >>> from msutils.analysis import measure_flops
        >>> stats = measure_flops(model, input_shape=(1, 3, 224, 224))
        >>> print(f"FLOPs: {stats['flops']/1e9:.2f} G")
    """
    # 简化实现
    # 实际需要 hooks 或 trace
    total_flops = 0
    
    for name, param in model.parameters_dict().items():
        # 估算每个参数的 FLOPs
        # 卷积: 2 * H * W * C_in * C_out * K_h * K_w
        # 全连接: 2 * N_in * N_out
        if 'conv' in name.lower() or 'dense' in name.lower():
            flops = param.size * 2
            total_flops += flops
    
    return {
        'flops': total_flops,
        'flops_g': total_flops / 1e9,
        'flops_m': total_flops / 1e6
    }


def model_summary(model, input_shape: Tuple = (1, 3, 224, 224)) -> str:
    """
    生成模型摘要信息
    
    Args:
        model: MindSpore 模型
        input_shape: 输入形状
    
    Returns:
        模型摘要字符串
    
    Example:
        >>> from msutils.analysis import model_summary
        >>> summary = model_summary(model)
        >>> print(summary)
    """
    # 统计参数
    params = count_parameters(model)
    flops = measure_flops(model, input_shape)
    
    # 生成摘要
    summary = []
    summary.append("=" * 70)
    summary.append("Model Summary")
    summary.append("=" * 70)
    summary.append("")
    summary.append("Parameters:")
    summary.append(f"  Total:           {params['total']:,}")
    summary.append(f"  Trainable:      {params['trainable']:,}")
    summary.append(f"  Non-trainable:  {params['non_trainable']:,}")
    summary.append("")
    summary.append("Computational Complexity:")
    summary.append(f"  FLOPs:          {flops['flops']:,.0f}")
    summary.append(f"  FLOPs:          {flops['flops_g']:.2f} G")
    summary.append("")
    summary.append("Layer Details:")
    summary.append("-" * 70)
    
    for name, param in model.parameters_dict().items():
        summary.append(f"  {name:<40} {param.size:>15,}")
    
    summary.append("=" * 70)
    
    return "\n".join(summary)


def calculate_model_size(model) -> Dict[str, float]:
    """
    计算模型文件大小
    
    Args:
        model: MindSpore 模型
    
    Returns:
        模型大小统计
    
    Example:
        >>> from msutils.analysis import calculate_model_size
        >>> size = calculate_model_size(model)
        >>> print(f"Model size: {size['size_mb']:.2f} MB")
    """
    import tempfile
    from mindspore import save_checkpoint
    
    # 保存到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.ckpt') as f:
        temp_path = f.name
    
    save_checkpoint(model, temp_path)
    
    import os
    size_bytes = os.path.getsize(temp_path)
    os.remove(temp_path)
    
    return {
        'size_bytes': size_bytes,
        'size_kb': size_bytes / 1024,
        'size_mb': size_bytes / (1024 ** 2),
        'size_gb': size_bytes / (1024 ** 3)
    }


__all__ = [
    'count_parameters',
    'measure_flops',
    'model_summary',
    'calculate_model_size'
]
