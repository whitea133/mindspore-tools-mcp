# MindSpore Tools MCP 使用指南

本文档展示如何使用自然语言 Prompt 调用 MindSpore Tools MCP。

## 目录

- [快速开始](#快速开始)
- [可用 Prompt](#可用-prompt)
- [使用示例](#使用示例)
- [最佳实践](#最佳实践)

---

## 快速开始

### 1. 启动 MCP Server

```bash
cd E:\CodeProject\mindspore-tools-mcp
python -m mindspore_tools_mcp.server
```

### 2. 在 Claude/OpenCode 中使用

在对话中直接用自然语言描述你的需求，MCP 会自动选择合适的工具。

---

## 可用 Prompt

### 1. model_lookup - 模型查找

**用途**：根据任务查找 MindSpore 模型

**Prompt 示例**：
```
帮我找一个用于图像分类的 MindSpore 模型
```

**对应代码**：
```python
@prompt("model_lookup")
def model_lookup(task: str, limit: int = 5) -> str:
    return f"Find up to {limit} MindSpore models relevant to task: {task}"
```

---

### 2. model_recommend - 模型推荐

**用途**：根据需求智能推荐模型

**Prompt 示例**：
```
我需要一个能在昇腾芯片上运行的图像分割模型，推荐一下
```

**对应代码**：
```python
@prompt("model_recommend")
def model_recommend(query: str, hardware: str | None = None) -> str:
    # 支持 hardware 参数：ascend / gpu / cpu
```

---

### 3. model_compare - 模型对比

**用途**：对比多个模型的性能、参数量等

**Prompt 示例**：
```
对比一下 ResNet50 和 ViT 模型在 MindSpore 中的表现
```

**对应代码**：
```python
@prompt("model_compare")
def model_compare(models: str) -> str:
    # models: 逗号分隔的模型名
```

---

### 4. migration_guide - 迁移指南

**用途**：PyTorch 到 MindSpore 的迁移指导

**Prompt 示例**：
```
我想把 PyTorch 的 BERT 模型迁移到 MindSpore，给个指南
```

**对应代码**：
```python
@prompt("migration_guide")
def migration_guide(from_framework: str = "pytorch", to_framework: str = "mindspore") -> str:
```

---

### 5. performance_optimize - 性能优化

**用途**：获取模型在特定硬件上的优化建议

**Prompt 示例**：
```
如何优化 ResNet50 在昇腾上的推理性能？
```

**对应代码**：
```python
@prompt("performance_optimize")
def performance_optimize(model_name: str, hardware: str = "ascend") -> str:
```

---

## 使用示例

### 示例 1：查找模型

**用户输入**：
```
帮我找一个用于目标检测的 MindSpore 模型
```

**MCP 自动调用**：
```python
model_lookup(task="目标检测", limit=5)
```

**输出**：
```
根据您的需求，我找到了以下 MindSpore 目标检测模型：
1. YOLOv8 - 高效实时检测
2. Faster R-CNN - 高精度检测
3. SSD - 平衡性能与精度
...
```

---

### 示例 2：模型迁移

**用户输入**：
```
我有一段 PyTorch 代码，想迁移到 MindSpore：
import torch
model = torch.nn.Linear(10, 5)
```

**MCP 自动调用**：
```python
migration_guide(from_framework="pytorch", to_framework="mindspore")
```

**输出**：
```
PyTorch 到 MindSpore 迁移指南：

1. API 对照：
   torch.nn.Linear → mindspore.nn.Dense
   torch.relu → mindspore.ops.ReLU()

2. 代码转换：
   # PyTorch
   model = torch.nn.Linear(10, 5)
   
   # MindSpore
   model = mindspore.nn.Dense(10, 5)
```

---

### 示例 3：性能优化

**用户输入**：
```
我的 ResNet50 模型在昇腾上推理很慢，怎么优化？
```

**MCP 自动调用**：
```python
performance_optimize(model_name="ResNet50", hardware="ascend")
```

**输出**：
```
ResNet50 在昇腾上的优化建议：

1. 算子融合：
   - 启用 Conv+BN 融合
   - 启用 ReLU 融合

2. 量化优化：
   - 使用 INT8 量化减少计算量

3. 批处理优化：
   - 增大 batch_size 提高吞吐量
```

---

## 最佳实践

### 1. 使用具体的描述

**推荐**：
```
找一个在昇腾 910 上运行的高精度图像分类模型
```

**不推荐**：
```
找个模型
```

### 2. 提供硬件约束

如果知道目标硬件，务必说明：
- `昇腾` / `ascend`
- `GPU` / `gpu`
- `CPU` / `cpu`

### 3. 明确任务类型

| 任务 | 示例 Prompt |
|------|-----------|
| 图像分类 | `图像分类模型推荐` |
| 目标检测 | `目标检测任务用什么模型好` |
| 语义分割 | `语义分割的 MindSpore 模型` |
| NLP | `文本分类模型` |

### 4. 组合使用

可以一次提出多个需求：

```
我需要一个能在昇腾上运行的图像分割模型，
并且要和 PyTorch 的 DeepLabV3 性能相当
```

---

## Prompt 注册机制

MCP 使用装饰器注册 Prompt：

```python
from .prompt import prompt

@prompt("custom_name")
def my_custom_prompt(param: str) -> str:
    """自定义 Prompt"""
    return f"处理: {param}"
```

---

## 常见问题

### Q: Prompt 没响应怎么办？
A: 检查 MCP Server 是否正常运行，确保端口未被占用。

### Q: 如何添加新的 Prompt？
A: 在 `src/mindspore_tools_mcp/prompt.py` 中添加新函数，使用 `@prompt` 装饰器。

### Q: 支持哪些硬件平台？
A: 昇腾 (Ascend)、GPU、CPU 都支持。

---

## 更多资源

- MindSpore 官方文档：https://www.mindspore.cn/
- 项目 GitHub：https://github.com/whitea133/mindspore-tools-mcp
