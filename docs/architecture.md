# 架构设计文档

> MindSpore Tools MCP 项目整体架构设计说明

---

## 1. 项目概述

**mindspore-tools-mcp** 是一个基于 **Model Context Protocol (MCP)** 的 MindSpore 开发工具集，提供模型检索、API 映射、代码质量评分、训练模板生成等功能，帮助开发者快速上手 MindSpore 深度学习框架。

### 核心定位

- **MCP 工具服务**：通过 MCP 协议暴露 29 个工具，供 AI Agent（如 Claude、Cline）调用
- **MindSpore 工具库**：内置 `msutils` 工具库，覆盖数据处理、训练、安全、评估等场景
- **数据与资源**：提供模型清单、API 映射、示例代码等资源

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      MCP 客户端层                          │
│  (Claude Code / Cline / Cursor / OpenClaw / 其他 MCP 客户端) │
└───────────────────────┬─────────────────────────────────┘
                        │ MCP 协议 (stdio/SSE)
┌───────────────────────▼─────────────────────────────────┐
│                    MCP 服务层                            │
│  server.py  (FastMCP 框架)                             │
│  - 工具注册 (tools.py / msutils_tools.py / ...)         │
│  - 资源暴露 (mindspore:// 协议)                          │
│  - Prompt 模板 (prompts/)                                │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                    核心功能层                             │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  模型检索    │  │  API 映射   │  │  代码评分   │    │
│  │  tools.py   │  │  tools.py   │  │ linter/     │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ 训练模板    │  │  API 示例   │  │  msutils/   │    │
│  │ templates/  │  │api_examples/│  │  工具库     │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
│                                                          │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                    数据资源层                             │
│  data/                                                   │
│  - mindspore_official_models.json (300+ 模型)           │
│  - pytorch_ms_api_mapping_consistent.json (95KB)        │
│  - pytorch_ms_api_mapping_diff.json (32KB)              │
│  - convert/ (按 section 分片的映射数据)                  │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 模块说明

### 3.1 MCP 服务入口 (`server.py`)

使用 **FastMCP** 框架构建 MCP 服务，负责：
- 注册 MCP 工具（29 个）
- 暴露 MCP 资源（`mindspore://` 协议）
- 提供 MCP Prompt 模板

```python
# server.py 核心结构
from fastmcp import FastMCP

mcp = FastMCP("mindspore-tools-mcp")

# 导入并注册工具
from mindspore_tools_mcp import tools
from mindspore_tools_mcp import msutils_tools
# ...

# 启动服务
if __name__ == "__main__":
    mcp.run()
```

---

### 3.2 工具模块

#### `tools.py` - 核心工具
| 工具 | 说明 |
|------|------|
| `list_models` | 模型列表检索 |
| `get_model_info` | 模型详情查询 |
| `recommend_models` | 智能模型推荐 |
| `compare_models` | 模型对比 |
| `query_op_mapping` | API 映射查询 |
| `diagnose_translation` | 翻译诊断 |

#### `msutils_tools.py` - msutils MCP 封装
将 `msutils/` 工具库封装为 MCP 工具，包括：
- 数据处理工具（增强、加载、变换）
- 训练工具（回调函数、学习率调度）
- AI 安全工具（攻击、防御、评估）

#### `linter_tools.py` - 代码评分器
- `lint_mindspore_code` - 代码质量评分
- `get_lint_rules` - 获取检查规则
- `compare_code_snippets` - 代码对比

#### `template_tools.py` - 训练模板生成器
- `generate_training_template` - 生成完整训练脚本
- `get_available_options` - 获取可用选项
- `generate_quick_start` - 生成快速入门脚本

#### `api_tools.py` - API 示例生成器
- `get_api_examples` - 获取 API 示例
- `search_apis` - 搜索 API
- `list_api_categories` - 列出 API 分类
- `get_related_apis` - 获取相关 API
- `get_quick_reference` - 快速参考

---

### 3.3 工具库 (`msutils/`)

```
msutils/
├── data/           # 数据处理
│   ├── augmentations.py   # 数据增强
│   ├── loaders.py         # 数据加载
│   └── transforms.py      # 数据变换
├── train/          # 训练工具
│   ├── callbacks.py       # 回调函数
│   └── schedulers.py      # 学习率调度
├── security/       # AI 安全
│   ├── attacks.py         # 攻击方法
│   ├── defenses.py        # 防御方法
│   └── evaluation.py       # 评估工具
├── analysis/       # 模型分析
│   └── complexity.py      # 复杂度计算
├── distributed/    # 分布式训练
│   └── ddp.py            # DDP 工具
├── deploy/         # 部署工具
│   ├── quantize.py        # 量化
│   └── convert.py         # 格式转换
├── eval/           # 评估指标
│   └── metrics.py         # 指标计算
└── nlp/            # NLP 工具
    ├── augmenters.py       # 文本增强
    └── tokenizers.py      # 分词器
```

---

### 3.4 数据资源 (`data/`)

| 文件 | 大小 | 说明 |
|------|------|------|
| `mindspore_official_models.json` | ~50KB | 300+ 官方模型清单 |
| `pytorch_ms_api_mapping_consistent.json` | 95KB | 一致 API 映射 |
| `pytorch_ms_api_mapping_diff.json` | 32KB | 差异 API 映射 |
| `convert/section_*.json` | 分片 | 按 section 分片的映射数据 |

---

## 4. 技术栈

| 组件 | 技术 |
|------|------|
| MCP 框架 | **FastMCP** (modelcontextprotocol.io) |
| 编程语言 | **Python 3.8+** |
| 包管理 | **uv** (推荐) / pip |
| 代码质量 | **Ruff** (linting), **Black** (formatting) |
| 测试框架 | **pytest** |
| 文档格式 | **Markdown** |

---

## 5. 工作流程

### 5.1 模型检索流程

```
用户输入 → recommend_models(topic, hardware)
               ↓
         tools.py → 加载 data/mindspore_official_models.json
               ↓
         过滤 + 排序（精度↓ / 参数量↓ / 硬件兼容）
               ↓
         返回推荐列表（含评分理由）
```

### 5.2 API 映射查询流程

```
用户输入 → query_op_mapping(pytorch_api)
               ↓
         tools.py → 加载 data/pytorch_ms_api_mapping_*.json
               ↓
         模糊匹配（支持 "Conv2d" / "nn.Conv2d" / "conv2d"）
               ↓
         返回映射结果（MS 实现 + 差异说明 + 代码示例）
```

### 5.3 代码评分流程

```
用户输入 → lint_mindspore_code(code, check_level)
               ↓
         linter_tools.py → linter/rating.py
               ↓
         4 维度 × 26 条规则检查
               ↓
         返回评分报告（总分 + 各维度得分 + 改进建议）
```

---

## 6. 扩展指南

### 6.1 添加新的 MCP 工具

1. 在对应 `tools_*.py` 文件中定义函数
2. 使用 `@mcp.tool()` 装饰器注册
3. 编写清晰的 docstring（会被 MCP 客户端展示）
4. 更新 README.md 和 README_CN.md

```python
@mcp.tool()
def my_new_tool(param1: str, param2: int = 10) -> dict:
    """工具简要说明。
    
    Args:
        param1: 参数1说明
        param2: 参数2说明（默认: 10）
    
    Returns:
        返回值说明
    """
    # 实现逻辑
    return {"result": "..."}
```

### 6.2 添加 msutils 模块

1. 在 `msutils/` 下创建新模块目录
2. 编写工具函数，添加完整 docstring
3. 在 `msutils_tools.py` 中封装为 MCP 工具
4. 添加测试用例在 `tests/`

### 6.3 更新数据文件

```bash
# 更新模型清单
uv run python scripts/update_model_list.py

# 更新 API 映射
uv run python scripts/fetch_api_mapping.py
```

---

## 7. 部署方式

### 7.1 本地开发

```bash
git clone https://github.com/whitea133/mindspore-tools-mcp.git
cd mindspore-tools-mcp
uv sync
uv run python -m mindspore_tools_mcp.server
```

### 7.2 MCP 客户端配置

#### Claude Code / Cline / Cursor

```json
{
  "mcpServers": {
    "mindspore-tools": {
      "command": "uv",
      "args": [
        "--directory",
        "E:/CodeProject/mindspore-tools-mcp",
        "run",
        "mindspore-tools-mcp"
      ]
    }
  }
}
```

---

## 8. 性能优化

### 8.1 数据加载优化
- 模型清单懒加载（首次调用时加载）
- API 映射建立索引（字典 O(1) 查询）

### 8.2 代码评分优化
- 规则检查并行化（multiprocessing）
- 缓存已评分代码（避免重复计算）

---

## 9. 测试策略

```
tests/
├── test_tools.py              # 工具函数测试
├── test_msutils_tools.py      # msutils 工具测试
├── test_linter_tools.py       # 代码评分测试
├── test_template_tools.py     # 训练模板测试
└── test_api_tools.py          # API 示例测试
```

**覆盖目标**：核心功能测试覆盖率 **80%+**

---

## 10. 版本管理

| 版本 | 说明 |
|------|------|
| `1.0.0` | 正式发布版本（29 个 MCP 工具，17 个示例） |
| `0.1.0` | 初始开发版本 |

---

## 11. 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 12. 联系方式

- **GitHub**: https://github.com/whitea133/mindspore-tools-mcp
- **Gitee**: https://gitee.com/whitea133/mindspore-tools-mcp
- **邮箱**: 1309848726@qq.com

---

*最后更新：2026-05-14*