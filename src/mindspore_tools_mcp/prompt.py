"""Prompt definitions for MCP."""

from __future__ import annotations

from typing import Callable, Dict

# central registry for prompts: name -> function
PROMPT_REGISTRY: Dict[str, Callable] = {}


def prompt(name: str | None = None) -> Callable:
    """Decorator to tag and register a function as an MCP prompt with an optional name."""

    def decorator(func: Callable) -> Callable:
        prompt_name = name or func.__name__
        PROMPT_REGISTRY[prompt_name] = func
        setattr(func, "__mcp_prompt_name__", prompt_name)
        return func

    return decorator


@prompt("model_lookup")
def model_lookup(task: str, limit: int = 5) -> str:
    """Generate a simple prompt for looking up models by task."""
    return f"Find up to {limit} MindSpore models relevant to task: {task}"


@prompt("model_recommend")
def model_recommend(query: str, hardware: str | None = None) -> str:
    """Generate a prompt for intelligent model recommendation.
    
    Args:
        query: Natural language description of the task/requirement
        hardware: Optional hardware constraint (ascend/gpu/cpu)
    
    Example:
        >>> model_recommend("图像分类", hardware="ascend")
    """
    hw_hint = f" on {hardware.upper()}" if hardware else ""
    return f"""推荐适合以下需求的 MindSpore 模型{hw_hint}:

需求描述: {query}

请使用 recommend_models 工具获取推荐结果，并分析推荐理由。
"""


@prompt("model_compare")
def model_compare(models: str) -> str:
    """Generate a prompt for comparing multiple models.
    
    Args:
        models: Comma-separated model names or IDs
    
    Example:
        >>> model_compare("resnet50, vit, swin_transformer")
    """
    model_list = [m.strip() for m in models.split(",")]
    return f"""对比以下 MindSpore 模型:

模型列表: {', '.join(model_list)}

请使用 compare_models 工具获取详细对比，并给出选择建议。
"""


@prompt("migration_guide")
def migration_guide(from_framework: str = "pytorch", to_framework: str = "mindspore") -> str:
    """Generate a prompt for framework migration guidance.
    
    Args:
        from_framework: Source framework (default: pytorch)
        to_framework: Target framework (default: mindspore)
    """
    return f"""提供从 {from_framework} 迁移到 {to_framework} 的指南:

1. 使用 query_op_mapping 工具检查 API 映射
2. 列出常用 API 的迁移差异
3. 提供代码迁移最佳实践建议
"""


@prompt("performance_optimize")
def performance_optimize(model_name: str, hardware: str = "ascend") -> str:
    """Generate a prompt for model performance optimization.
    
    Args:
        model_name: The model to optimize
        hardware: Target hardware (default: ascend)
    """
    return f"""优化 {model_name} 模型在 {hardware.upper()} 上的性能:

1. 分析模型架构特点
2. 提供算子融合建议
3. 推荐优化配置参数
4. 列出常见性能瓶颈及解决方案
"""
