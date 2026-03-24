# mindspore-tools-mcp

<div align="center">

![MindSpore](https://img.shields.io/badge/MindSpore-2.0+-orange?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![MCP](https://img.shields.io/badge/MCP-Ready-purple?style=flat-square)

**基于 MCP 的 MindSpore 模型与 API 映射工具包**

提供官方模型清单的标准化查询，并内置 PyTorch → MindSpore API 映射与代码翻译辅助。

[English](./README.md) | [中文](./README_CN.md)

</div>

---

## ✨ 功能特性

### 🔍 官方模型检索

- `list_models` - 支持按 group/category/task/suite 或关键词过滤
- `get_model_info` - 返回单模型详情
- `mindspore://models/official` - 资源端点提供完整模型清单

### 🗺️ API 映射工具

- `query_op_mapping` - 支持 section 过滤与模糊匹配
- `translate_pytorch_code` - 自动替换一致 API，并为差异项输出提示
- `mindspore://opmap/...` - 资源暴露 PyTorch→MindSpore API 映射（全量/分 section、consistent/diff）

### 📊 数据脚本

- `scripts/update_model_list.py` - 更新官方模型 JSON
- `scripts/fetch_api_mapping.py` - 抓取并刷新 API 映射

---

## 📦 目录结构

```
mindspore-tools-mcp/
│
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
│   └── mindspore_tools_mcp/         # MCP 服务
│       ├── server.py                 # MCP 入口
│       ├── tools.py                  # 工具定义
│       ├── resource.py               # 资源定义
│       ├── prompt.py                 # Prompt 注册
│       ├── backup_server.py
│       └── main.py
│
├── tests/
│   └── test.py                       # 冒烟测试
│
├── pyproject.toml                    # 项目配置
├── AGENTS.md                         # Agent 配置
└── uv.lock                           # 依赖锁文件
```

---

## 🚀 快速开始

### 1️⃣ 安装依赖

```bash
uv sync
```

### 2️⃣ 启动 MCP 服务

```bash
# stdio 模式
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

### 4️⃣ 调用示例

#### 工具调用

```python
# 列出所有 text-generation 模型
list_models(task="text-generation")

# 获取 llama2 模型详情
get_model_info("llama2")

# 查询 torch.addmm API 映射
query_op_mapping("torch.addmm")

# 翻译 PyTorch 代码
translate_pytorch_code("import torch; torch.addmm(...)")
```

#### 资源调用

```python
# 读取官方模型清单
mindspore://models/official

# 读取一致的 API 映射
mindspore://opmap/pytorch/consistent
```

### 5️⃣ 更新数据（可选）

```bash
# 刷新官方模型清单
uv run python scripts/update_model_list.py

# 刷新 API 映射数据
uv run python scripts/fetch_api_mapping.py
```

---

## 📖 API 参考

### 工具列表

| 工具名 | 说明 | 示例 |
|--------|------|------|
| `list_models` | 列出模型 | `list_models(task="text-generation")` |
| `get_model_info` | 获取模型详情 | `get_model_info("llama2")` |
| `query_op_mapping` | 查询 API 映射 | `query_op_mapping("torch.add")` |
| `translate_pytorch_code` | 翻译代码 | `translate_pytorch_code(code)` |

### 资源列表

| 资源名 | 说明 |
|--------|------|
| `mindspore://models/official` | 官方模型完整清单 |
| `mindspore://opmap/pytorch/consistent` | 一致 API 映射 |
| `mindspore://opmap/pytorch/diff` | 差异 API 映射 |

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
