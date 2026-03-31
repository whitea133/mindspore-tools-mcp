"""MindSpore MCP 工具集合，基于官方模型与映射表。"""

from __future__ import annotations

import json
from functools import lru_cache
import re
from pathlib import Path
from typing import Any

from mindspore_tools_mcp.resource import get_official_models

MODEL_FILE = Path(__file__).resolve().parents[2] / "data" / "mindspore_official_models.json"
OPMAP_CONSISTENT_FILE = Path(__file__).resolve().parents[2] / "data" / "pytorch_ms_api_mapping_consistent.json"
OPMAP_DIFF_FILE = Path(__file__).resolve().parents[2] / "data" / "pytorch_ms_api_mapping_diff.json"
OPMAP_SECTION_CONS_DIR = Path(__file__).resolve().parents[2] / "data" / "convert" / "consistent"
OPMAP_SECTION_DIFF_DIR = Path(__file__).resolve().parents[2] / "data" / "convert" / "diff"

SHAPE_HINT_APIS = {
    "torch.addmm",
    "torch.mm",
    "torch.matmul",
    "torch.bmm",
}


@lru_cache(maxsize=1)
def _load_registry() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """加载模型清单及列表。"""
    try:
        with MODEL_FILE.open("r", encoding="utf-8") as handle:
            payload: Any = json.load(handle)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing model registry file: {MODEL_FILE}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON payload in model registry: {MODEL_FILE}") from exc

    if not isinstance(payload, dict) or "models" not in payload:
        raise RuntimeError("Model registry JSON must be an object with a 'models' array")
    models = payload.get("models", [])
    if not isinstance(models, list):
        raise RuntimeError("Model registry 'models' field must be a list")
    return payload, models


def list_models(
    group: str | None = None,
    category: str | None = None,
    task: str | None = None,
    suite: str | None = None,
    q: str | None = None,
) -> list[dict[str, Any]]:
    """列出模型，可按 group/category/task/suite 或名称关键字过滤。"""
    _, models = _load_registry()

    def match(model: dict[str, Any]) -> bool:
        if group and model.get("group", "").lower() != group.lower():
            return False
        if category and model.get("category", "").lower() != category.lower():
            return False
        if suite and model.get("suite", "").lower() != suite.lower():
            return False
        if task:
            tasks = [t.lower() for t in model.get("task", []) if isinstance(t, str)]
            if task.lower() not in tasks:
                return False
        if q:
            q_lower = q.lower()
            if q_lower not in model.get("id", "").lower() and q_lower not in model.get("name", "").lower():
                return False
        return True

    filtered = [m for m in models if match(m)]

    # 仅返回核心字段，避免多余数据噪声
    projection_keys = {"id", "name", "group", "category", "task", "suite", "variants", "links", "dataset", "metrics", "hardware"}
    projected: list[dict[str, Any]] = []
    for m in filtered:
        projected.append({k: m.get(k) for k in projection_keys if k in m})
    return projected


def get_model_info(model_id: str) -> dict[str, Any]:
    """按 id 或 name（不区分大小写）返回完整模型记录。"""
    payload, models = _load_registry()
    needle = model_id.lower()
    for m in models:
        if m.get("id", "").lower() == needle or m.get("name", "").lower() == needle:
            return m
    raise ValueError(f"Model '{model_id}' not found in registry (version={payload.get('version')})")


def fetch_official_models() -> dict:
    """返回官方模型清单的完整 JSON。"""
    return get_official_models()


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing mapping file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON payload in mapping: {path}") from exc


def _load_section_map(folder: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if not folder.exists():
        return data
    for file in folder.glob("*.json"):
        try:
            data[file.stem] = _load_json(file)
        except Exception:
            continue
    return data


def _sorted_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 PyTorch API 长度排序，优先匹配更长的名称。"""
    return sorted(items, key=lambda r: len(r.get("pytorch", "")), reverse=True)


def _collect_mapping_items(section: str | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """汇总一致/差异映射条目，可选按 section 限定。"""
    base_cons = _load_json(OPMAP_CONSISTENT_FILE)
    base_diff = _load_json(OPMAP_DIFF_FILE)
    cons_items = base_cons.get("items", []) if isinstance(base_cons, dict) else []
    diff_items = base_diff.get("items", []) if isinstance(base_diff, dict) else []

    if section:
        sec_cons_all = _load_section_map(OPMAP_SECTION_CONS_DIR)
        sec_diff_all = _load_section_map(OPMAP_SECTION_DIFF_DIR)
        cons_sec = sec_cons_all.get(section, {})
        diff_sec = sec_diff_all.get(section, {})
        if cons_sec and isinstance(cons_sec, dict):
            cons_items = cons_sec.get("items", cons_items)
        if diff_sec and isinstance(diff_sec, dict):
            diff_items = diff_sec.get("items", diff_items)
    else:
        # 未指定 section 时合并所有分表，覆盖更多映射（如 torch_tensor 等）
        sec_cons_all = _load_section_map(OPMAP_SECTION_CONS_DIR)
        sec_diff_all = _load_section_map(OPMAP_SECTION_DIFF_DIR)
        for sec in sec_cons_all.values():
            if isinstance(sec, dict):
                cons_items += sec.get("items", [])
        for sec in sec_diff_all.values():
            if isinstance(sec, dict):
                diff_items += sec.get("items", [])

    return _sorted_items(cons_items), _sorted_items(diff_items)


def _count_occurrences(text: str, target: str) -> int:
    """统计边界安全的目标符号出现次数。"""
    if not target:
        return 0
    pattern = rf"(?<![\w.]){re.escape(target)}(?![\w.])"
    return len(list(re.finditer(pattern, text)))


def query_op_mapping(op: str, section: str | None = None) -> dict[str, list[dict[str, str]]]:
    """查询 PyTorch→MindSpore API 映射（支持 section 过滤与模糊匹配）。

    Args:
        op: PyTorch API 名称或子串（如 "torch.addmm" 或 "addmm"）。
        section: 可选 section（如 "torch"、"torchvision"）缩小搜索范围。

    Returns:
        {"consistent": [...], "diff": [...]} 匹配条目。
    """
    key = op.lower()
    def match_row(row: dict[str, Any]) -> bool:
        pt = row.get("pytorch", "").lower()
        return key in pt

    # load data
    base_cons = _load_json(OPMAP_CONSISTENT_FILE)
    base_diff = _load_json(OPMAP_DIFF_FILE)
    cons_items = base_cons.get("items", []) if isinstance(base_cons, dict) else []
    diff_items = base_diff.get("items", []) if isinstance(base_diff, dict) else []

    sec_cons_all = _load_section_map(OPMAP_SECTION_CONS_DIR)
    sec_diff_all = _load_section_map(OPMAP_SECTION_DIFF_DIR)

    if section:
        cons_sec = sec_cons_all.get(section, {})
        diff_sec = sec_diff_all.get(section, {})
        if cons_sec and isinstance(cons_sec, dict):
            cons_items = cons_items + cons_sec.get("items", [])
        if diff_sec and isinstance(diff_sec, dict):
            diff_items = diff_items + diff_sec.get("items", [])
    else:
        # 未指定 section 时合并所有分表，覆盖更多映射（如 torch_tensor 等）
        for sec in sec_cons_all.values():
            if isinstance(sec, dict):
                cons_items += sec.get("items", [])
        for sec in sec_diff_all.values():
            if isinstance(sec, dict):
                diff_items += sec.get("items", [])

    cons_hits = [r for r in cons_items if match_row(r)]
    diff_hits = [r for r in diff_items if match_row(r)]

    return {
        "consistent": cons_hits,
        "diff": diff_hits,
    }


# =============================================================================
# 智能模型推荐系统
# =============================================================================

# 任务关键词映射：自然语言 -> 标准任务类型
TASK_KEYWORDS: dict[str, list[str]] = {
    # 文本相关
    "文本生成": ["text-generation"],
    "生成文本": ["text-generation"],
    "对话": ["text-generation"],
    "聊天": ["text-generation"],
    "大模型": ["text-generation"],
    "llm": ["text-generation"],
    "语言模型": ["text-generation"],
    
    # 图像相关
    "图像分类": ["image-classification"],
    "图片分类": ["image-classification"],
    "分类图片": ["image-classification"],
    "分类图像": ["image-classification"],
    
    # OCR 相关
    "ocr": ["text-detection", "text-recognition"],
    "文字识别": ["text-recognition"],
    "文字检测": ["text-detection"],
    "文本识别": ["text-recognition"],
    "文本检测": ["text-detection"],
    
    # 推荐与搜索
    "推荐": ["recommendation"],
    "推荐系统": ["recommendation"],
    
    # 科学计算
    "科学计算": ["scientific-computing"],
    "物理仿真": ["scientific-computing"],
    "分子模拟": ["scientific-computing"],
    
    # 强化学习
    "强化学习": ["reinforcement-learning"],
    "游戏ai": ["reinforcement-learning"],
    "机器人": ["reinforcement-learning"],
}

# 套件关键词映射
SUITE_KEYWORDS: dict[str, str] = {
    "mindformers": "mindformers",
    "大模型": "mindformers",
    "大语言模型": "mindformers",
    "llm": "mindformers",
    "transformer": "mindformers",
    
    "mindcv": "mindcv",
    "计算机视觉": "mindcv",
    "视觉": "mindcv",
    "图像": "mindcv",
    "cv": "mindcv",
    
    "mindocr": "mindocr",
    "文字识别": "mindocr",
    "ocr": "mindocr",
    
    "mindrl": "mindrl",
    "强化学习": "mindrl",
    
    "mindrec": "mindrec",
    "推荐": "mindrec",
    
    "mindscience": "mindscience",
    "科学计算": "mindscience",
}

# 模型特性评分权重
MODEL_FEATURES: dict[str, dict[str, float]] = {
    # 热门模型加成
    "llama3": {"popularity": 1.0, "quality": 0.95},
    "llama2": {"popularity": 0.9, "quality": 0.9},
    "qwen1.5": {"popularity": 0.95, "quality": 0.95},
    "qwen": {"popularity": 0.9, "quality": 0.9},
    "glm3": {"popularity": 0.85, "quality": 0.85},
    "baichuan2": {"popularity": 0.8, "quality": 0.85},
    "resnet": {"popularity": 0.9, "quality": 0.85},
    "vit": {"popularity": 0.85, "quality": 0.9},
    "yolo": {"popularity": 0.95, "quality": 0.9},
    "bert": {"popularity": 0.9, "quality": 0.85},
}


def _extract_tasks(query: str) -> list[str]:
    """从自然语言查询中提取任务类型。"""
    query_lower = query.lower()
    tasks: set[str] = set()
    
    for keyword, task_list in TASK_KEYWORDS.items():
        if keyword.lower() in query_lower:
            tasks.update(task_list)
    
    return list(tasks)


def _extract_suite(query: str) -> str | None:
    """从查询中提取套件偏好。"""
    query_lower = query.lower()
    
    for keyword, suite in SUITE_KEYWORDS.items():
        if keyword.lower() in query_lower:
            return suite
    
    return None


def _compute_model_score(
    model: dict[str, Any],
    query: str,
    tasks: list[str],
    suite: str | None,
    hardware: str | None,
) -> tuple[float, list[str]]:
    """计算模型匹配分数和推荐理由。
    
    Returns:
        (score, reasons)
    """
    score = 0.0
    reasons: list[str] = []
    
    model_id = model.get("id", "")
    model_name = model.get("name", "")
    model_tasks = [t.lower() for t in model.get("task", [])]
    model_suite = model.get("suite", "")
    model_category = model.get("category", "")
    model_variants = model.get("variants", [])
    model_metrics = model.get("metrics", {})
    model_hardware = model.get("hardware", {})
    
    # 1. 任务匹配 (权重: 40%)
    if tasks:
        task_match = any(t.lower() in model_tasks for t in tasks)
        if task_match:
            score += 0.4
            reasons.append(f"✓ 支持任务: {', '.join(tasks)}")
    else:
        # 无明确任务时，检查查询词是否匹配模型名或类别
        query_lower = query.lower()
        if query_lower in model_id.lower() or query_lower in model_name.lower():
            score += 0.35
            reasons.append(f"✓ 名称匹配: {model_name}")
        if query_lower in model_category.lower():
            score += 0.15
            reasons.append(f"✓ 类别匹配: {model_category}")
    
    # 2. 套件匹配 (权重: 20%)
    if suite and suite.lower() == model_suite.lower():
        score += 0.2
        reasons.append(f"✓ 套件: {model_suite}")
    
    # 3. 硬件兼容性 (权重: 20%)
    if hardware:
        hw_lower = hardware.lower()
        hw_info = model_hardware.get(hw_lower) if model_hardware else None
        if hw_info is True or hw_info == "✅" or hw_info == "supported":
            score += 0.2
            reasons.append(f"✓ 支持 {hardware.upper()}")
        elif hw_info is False:
            score -= 0.1
            reasons.append(f"⚠ 不支持 {hardware.upper()}")
    
    # 4. 模型特性加成 (权重: 20%)
    for model_key, features in MODEL_FEATURES.items():
        if model_key in model_id.lower() or model_key in model_name.lower():
            # 综合考虑热度和质量
            feature_score = (features.get("popularity", 0.5) + features.get("quality", 0.5)) / 2
            score += 0.2 * feature_score
            if features.get("popularity", 0) > 0.8:
                reasons.append("★ 热门模型")
            break
    
    # 5. 变体数量加成（变体多意味着更灵活）
    if len(model_variants) > 2:
        score += 0.05
        reasons.append(f"✓ {len(model_variants)} 个变体可选")
    
    # 6. 有性能指标加成
    if model_metrics:
        score += 0.05
        metric_str = ", ".join(f"{k}: {v}" for k, v in model_metrics.items() if v is not None)
        if metric_str:
            reasons.append(f"📊 性能: {metric_str}")
    
    return min(score, 1.0), reasons


def recommend_models(
    query: str,
    task: str | None = None,
    suite: str | None = None,
    hardware: str | None = None,
    limit: int = 5,
    min_score: float = 0.3,
) -> dict[str, Any]:
    """智能模型推荐 - 根据自然语言描述推荐合适的 MindSpore 模型。
    
    Args:
        query: 自然语言查询描述
            - 示例: "图像分类", "文本生成", "OCR"
            - 支持: 任务描述、模型名称、套件名称、应用场景
        task: 任务类型过滤（可选）
            - 如: "text-generation", "image-classification", "object-detection"
        suite: 套件过滤（可选）
            - 如: "mindformers", "mindcv", "mindocr", "mindrl", "mindscience"
        hardware: 硬件约束（可选）
            - "ascend", "gpu", "cpu"
        limit: 返回数量（默认 5）
        min_score: 最低匹配分数阈值（默认 0.3）
    
    Returns:
        {
            "query": "原始查询",
            "interpreted": {
                "tasks": ["解析出的任务类型"],
                "suite": "解析出的套件偏好"
            },
            "recommendations": [
                {
                    "model": {...模型信息...},
                    "score": 0.85,
                    "reasons": ["推荐理由"]
                }
            ],
            "total_found": 匹配模型总数,
            "suggestion": "额外建议",
            "note": "注意事项（如有）"
        }
    
    Examples:
        >>> recommend_models("图像分类")
        >>> recommend_models("文本生成", hardware="ascend")
        >>> recommend_models("大模型对话", suite="mindformers")
    """
    _, models = _load_registry()
    
    # 解析查询
    interpreted_tasks = _extract_tasks(query)
    interpreted_suite = _extract_suite(query)
    
    # 合并显式参数和解析结果
    final_tasks = [task] if task else interpreted_tasks
    final_suite = suite or interpreted_suite
    
    # 检查是否有明确的任务需求
    has_explicit_task = bool(final_tasks) or bool(final_suite) or bool(hardware)
    
    # 计算所有模型的匹配分数
    scored_models: list[tuple[dict[str, Any], float, list[str]]] = []
    
    for model in models:
        score, reasons = _compute_model_score(
            model, query, final_tasks, final_suite, hardware
        )
        if score >= min_score:
            scored_models.append((model, score, reasons))
    
    # 按分数排序
    scored_models.sort(key=lambda x: x[1], reverse=True)
    
    # 构建推荐结果
    recommendations: list[dict[str, Any]] = []
    for model, score, reasons in scored_models[:limit]:
        # 精简模型信息
        model_info = {
            "id": model.get("id"),
            "name": model.get("name"),
            "category": model.get("category"),
            "task": model.get("task"),
            "suite": model.get("suite"),
            "variants": model.get("variants"),
            "links": model.get("links"),
            "metrics": model.get("metrics"),
        }
        recommendations.append({
            "model": model_info,
            "score": round(score, 2),
            "reasons": reasons,
        })
    
    # 生成额外建议
    suggestion = ""
    note = ""
    
    if not final_tasks:
        note = f"未识别到明确的任务类型。当前支持的任务: 图像分类、文本生成、OCR、推荐系统、科学计算、强化学习"
    
    if not recommendations:
        suggestion = f"未找到匹配模型。请尝试更具体的描述，如 '图像分类'、'文本生成'、'OCR' 等"
    elif len(recommendations) == 1:
        suggestion = f"推荐使用 {recommendations[0]['model']['name']}，这是针对您需求的首选模型"
    else:
        top_model = recommendations[0]['model']['name']
        suggestion = f"首选推荐 {top_model}，也可考虑其他备选模型根据具体需求选择"
    
    return {
        "query": query,
        "interpreted": {
            "tasks": final_tasks,
            "suite": final_suite,
        },
        "recommendations": recommendations,
        "total_found": len(scored_models),
        "suggestion": suggestion,
        "note": note,
    }


def compare_models(model_ids: list[str]) -> dict[str, Any]:
    """对比多个模型，帮助用户选择最适合的模型。
    
    Args:
        model_ids: 模型 ID 列表（最多 5 个）
    
    Returns:
        {
            "models": [{...模型详情...}],
            "comparison": {
                "tasks": "任务对比",
                "suites": "套件对比",
                "variants_count": "变体数量对比",
                "metrics": "性能指标对比"
            },
            "recommendation": "选择建议"
        }
    """
    if len(model_ids) > 5:
        model_ids = model_ids[:5]
    
    _, models = _load_registry()
    
    # 查找模型
    found_models: list[dict[str, Any]] = []
    for model_id in model_ids:
        needle = model_id.lower()
        for m in models:
            if m.get("id", "").lower() == needle or m.get("name", "").lower() == needle:
                found_models.append(m)
                break
    
    if not found_models:
        return {
            "models": [],
            "comparison": {},
            "recommendation": "未找到任何匹配模型",
        }
    
    # 构建对比表
    comparison = {
        "tasks": {},
        "suites": {},
        "variants_count": {},
        "metrics": {},
        "hardware": {},
    }
    
    for m in found_models:
        name = m.get("name", m.get("id", ""))
        comparison["tasks"][name] = m.get("task", [])
        comparison["suites"][name] = m.get("suite", "未知")
        comparison["variants_count"][name] = len(m.get("variants", []))
        comparison["metrics"][name] = m.get("metrics", {})
        comparison["hardware"][name] = m.get("hardware", {})
    
    # 生成选择建议
    recommendation = _generate_comparison_recommendation(found_models)
    
    return {
        "models": [
            {
                "id": m.get("id"),
                "name": m.get("name"),
                "category": m.get("category"),
                "task": m.get("task"),
                "suite": m.get("suite"),
                "variants": m.get("variants"),
                "metrics": m.get("metrics"),
                "links": m.get("links"),
            }
            for m in found_models
        ],
        "comparison": comparison,
        "recommendation": recommendation,
    }


def _generate_comparison_recommendation(models: list[dict[str, Any]]) -> str:
    """生成模型对比建议。"""
    if len(models) == 1:
        return f"只有一个模型 {models[0].get('name')}，无需对比"
    
    # 检查是否有变体数量差异
    variants_counts = [len(m.get("variants", [])) for m in models]
    if max(variants_counts) > min(variants_counts) * 2:
        max_idx = variants_counts.index(max(variants_counts))
        return f"{models[max_idx].get('name')} 提供更多变体选择，灵活性更高"
    
    # 检查性能指标
    has_metrics = [bool(m.get("metrics")) for m in models]
    if any(has_metrics) and not all(has_metrics):
        idx = has_metrics.index(True)
        return f"{models[idx].get('name')} 提供了性能基准数据，便于评估"
    
    # 检查套件
    suites = [m.get("suite", "") for m in models]
    if len(set(suites)) == 1:
        return f"所有模型都属于 {suites[0]} 套件，可根据具体任务需求选择"
    
    return "请根据具体任务需求和硬件环境选择适合的模型"


def diagnose_translation(original_code: str, translated_code: str, section: str | None = None) -> dict[str, Any]:
    """诊断 LLM 翻译结果：基于映射表检查替换是否到位。

    Args:
        original_code: 原始 PyTorch 代码。
        translated_code: LLM 翻译后的 MindSpore 代码。
        section: 可选 section（如 "torchvision"）用于缩小映射范围。

    Returns:
        {
            "applied_mappings": [...],   # 原文命中的一致映射及替换计数
            "missing_mappings": [...],   # 原文命中但译文未出现的映射
            "diff_hits": [...],          # 差异映射命中
            "extra_calls": [...],        # 译文出现但原文未触发的 MindSpore API
            "annotated": "...",          # 在原文标注 TODO 的版本
        }
    """
    cons_items, diff_items = _collect_mapping_items(section)

    applied: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    extra: list[dict[str, Any]] = []
    diff_hits: list[dict[str, Any]] = []
    annotated = original_code

    def base_entry(row: dict[str, Any]) -> dict[str, Any]:
        return {k: row.get(k) for k in ("section", "pytorch", "mindspore", "description")}

    # 一致映射：检查源代码命中与译文替换情况
    for row in cons_items:
        pt = row.get("pytorch", "")
        ms = row.get("mindspore", "")
        if not pt or not ms:
            continue
        source_count = _count_occurrences(original_code, pt)
        translated_count = _count_occurrences(translated_code, ms) if translated_code else 0
        if source_count == 0 and translated_count == 0:
            continue
        entry = base_entry(row)
        entry["source_count"] = source_count
        entry["translated_count"] = translated_count
        applied.append(entry)
        if source_count > 0 and translated_count == 0:
            missing.append(entry)
        if source_count == 0 and translated_count > 0:
            extra_entry = entry.copy()
            extra_entry["note"] = "MindSpore API present but no matching PyTorch call found"
            extra.append(extra_entry)

    # 差异映射：仅提示，不自动替换
    for row in diff_items:
        pt = row.get("pytorch", "")
        if not pt:
            continue
        source_count = _count_occurrences(original_code, pt)
        if source_count == 0:
            continue
        entry = base_entry(row)
        entry["count"] = source_count
        if pt in SHAPE_HINT_APIS:
            entry["shape_hint"] = "check input/output shapes (expects matrix/matched dims)"
        diff_hits.append(entry)
        pattern = rf"(?<![\w.]){re.escape(pt)}(?![\w.])"
        desc = row.get("description") or "diff"
        ms = row.get("mindspore") or "mindspore.*"
        def add_comment(match: re.Match[str]) -> str:
            return f"# TODO: check mapping {pt} -> {ms}: {desc}\n{match.group(0)}"
        annotated = re.sub(pattern, add_comment, annotated)

    # 对未替换的命中添加标注，便于人工复核
    for miss in missing:
        pt = miss.get("pytorch") or ""
        ms = miss.get("mindspore") or "mindspore.*"
        if not pt:
            continue
        pattern = rf"(?<![\w.]){re.escape(pt)}(?![\w.])"
        def add_comment(match: re.Match[str]) -> str:
            return f"# TODO: replace {pt} -> {ms} per mapping\n{match.group(0)}"
        annotated = re.sub(pattern, add_comment, annotated)

    return {
        "applied_mappings": applied,
        "missing_mappings": missing,
        "diff_hits": diff_hits,
        "extra_calls": extra,
        "annotated": annotated,
    }
