# mindspore-tools-mcp

<div align="center">

![MindSpore](https://img.shields.io/badge/MindSpore-2.0+-orange?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![MCP](https://img.shields.io/badge/MCP-Ready-purple?style=flat-square)

**基于 MCP 的 MindSpore 模型与 API 映射工具包**

提供官方模型清单的标准化查询、智能模型推荐、以及 PyTorch → MindSpore API 映射与代码翻译辅助。

[English](./README.md) | [中文](./README_CN.md)

</div>

---

## ✨ 功能特性

### 🔍 官方模型检索

- `list_models` - 支持按 group/category/task/suite 或关键词过滤
- `get_model_info` - 返回单模型详情
- `mindspore://models/official` - 资源端点提供完整模型清单

### 🤖 智能模型推荐 (NEW!)

- `recommend_models` - 根据自然语言描述智能推荐合适的模型
  - 支持任务描述: "图像分类"、"文本生成"、"OCR"、"推荐系统" 等
  - 支持硬件约束: ascend、gpu、cpu
  - 提供推荐理由和匹配分数
- `compare_models` - 对比多个模型，帮助选择最适合的模型

### 🗺️ API 映射工具

- `query_op_mapping` - 支持 section 过滤与模糊匹配
- `diagnose_translation` - 检查 PyTorch→MindSpore 代码翻译是否完整
- `mindspore://opmap/...` - 资源暴露 PyTorch→MindSpore API 映射

### 📊 数据脚本

- `scripts/update_model_list.py` - 更新官方模型 JSON
- `scripts/fetch_api_mapping.py` - 抓取并刷新 API 映射

---

## 📦 目录结构

```
mindspore-tools-mcp/
├── data/                              # 官方模型与 API 映射数据
│   ├── convert/                       # 分 section 的映射分片
│   ├── mindspore_official_models.json
│   ├── pytorch_ms_api_mapping_consistent.json
│   └── pytorch_ms_api_mapping_diff.json
│
├── scripts/                           # 数据/映射更新脚本
│   ├── update_model_list.py
│   └── fetch_api_mapping.py
│
├── src/
│   └── mindspore_tools_mcp/           # MCP 服务
│       ├── server.py                  # MCP 入口
│       ├── tools.py                   # 工具定义
│       ├── resource.py                # 资源定义
│       └── prompt.py                  # Prompt 注册
│
├── tests/                             # 测试文件
├── pyproject.toml                     # 项目配置
└── uv.lock                            # 依赖锁文件
```

---

## 🚀 快速开始

### 1️⃣ 安装依赖

```bash
uv sync
```

### 2️⃣ 启动 MCP 服务

```bash
uv run python -m mindspore_tools_mcp.server
```

### 3️⃣ 客户端配置

```jsonc
// cline_mcp_settings.json
{
  "mindspore_tools_mcp": {
    "command": "uv",
    "args": [
      "--directory",
      "E:/CodeProject/mindspore-tools-mcp",
      "run",
      "python",
      "-m",
      "mindspore_tools_mcp.server"
    ],
    "autoApprove": []
  }
}
```

---

## 📖 API 参考

### 工具列表

| 工具名 | 说明 | 示例 |
|--------|------|------|
| `list_models` | 列出模型 | `list_models(task="text-generation")` |
| `get_model_info` | 获取模型详情 | `get_model_info("llama2")` |
| `recommend_models` | 智能模型推荐 | `recommend_models("图像分类")` |
| `compare_models` | 对比模型 | `compare_models(["resnet50", "vit"])` |
| `query_op_mapping` | 查询 API 映射 | `query_op_mapping("torch.add")` |
| `diagnose_translation` | 诊断代码翻译 | `diagnose_translation(py_code, ms_code)` |

### 资源列表

| 资源名 | 说明 |
|--------|------|
| `mindspore://models/official` | 官方模型完整清单 |
| `mindspore://opmap/pytorch/consistent` | 一致 API 映射 |
| `mindspore://opmap/pytorch/diff` | 差异 API 映射 |

### Prompt 列表

| Prompt 名 | 说明 |
|-----------|------|
| `model_lookup` | 按任务查找模型 |
| `model_recommend` | 智能模型推荐 |
| `model_compare` | 模型对比 |
| `migration_guide` | 迁移指南 |
| `performance_optimize` | 性能优化建议 |

---

## 💡 使用示例

### 智能模型推荐

```python
# 推荐"图像分类"模型
recommend_models("图像分类", limit=3)
# 返回: resnet18, resnet34, resnet50 (带推荐理由和性能指标)

# 推荐支持 Ascend 的文本生成模型
recommend_models("文本生成大模型", hardware="ascend", limit=3)
# 返回: llama2, qwen, baichuan2

# OCR 模型推荐
recommend_models("OCR文字识别", limit=5)
# 返回: dbnet_resnet18, dbnet_resnet50, ...
```

### 模型对比

```python
# 对比多个图像分类模型
compare_models(["resnet50", "vit", "swin_transformer"])
# 返回: 任务对比、性能对比、选择建议
```

### API 映射查询

```python
# 查询 PyTorch API 映射
query_op_mapping("torch.addmm")
# 返回: consistent/diff 映射列表
```

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📝 许可

MIT License - 详见 LICENSE 文件

---

## 👤 作者

- **GitHub**: [@whitea133](https://github.com/whitea133)
- **邮箱**: 1309848726@qq.com

---

## 🙏 致谢

- [MindSpore](https://www.mindspore.cn/) - 华为全场景深度学习框架
- [MCP](https://modelcontextprotocol.io/) - Model Context Protocol

---

<div align="center">

**如果这个项目对你有帮助，请给我们一个 ⭐**

</div>