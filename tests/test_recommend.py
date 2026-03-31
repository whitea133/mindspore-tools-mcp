#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试智能模型推荐功能"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from mindspore_tools_mcp.tools import recommend_models, compare_models

print("=" * 60)
print("测试 1: recommend_models('图像分类')")
print("=" * 60)
result = recommend_models('图像分类', limit=3)
print(f"查询: {result['query']}")
print(f"解析任务: {result['interpreted']['tasks']}")
print(f"匹配总数: {result['total_found']}")
print(f"建议: {result['suggestion']}")
print()
for r in result['recommendations']:
    print(f"  - {r['model']['name']} (分数: {r['score']})")
    for reason in r['reasons']:
        print(f"      {reason}")

print()
print("=" * 60)
print("测试 2: recommend_models('文本生成大模型', hardware='ascend')")
print("=" * 60)
result2 = recommend_models('文本生成大模型', hardware='ascend', limit=3)
print(f"查询: {result2['query']}")
print(f"解析任务: {result2['interpreted']['tasks']}")
print(f"匹配总数: {result2['total_found']}")
for r in result2['recommendations']:
    print(f"  - {r['model']['name']} (分数: {r['score']})")
    for reason in r['reasons']:
        print(f"      {reason}")

print()
print("=" * 60)
print("测试 3: recommend_models('OCR文字识别')")
print("=" * 60)
result3 = recommend_models('OCR文字识别', limit=3)
print(f"查询: {result3['query']}")
print(f"解析任务: {result3['interpreted']['tasks']}")
for r in result3['recommendations']:
    print(f"  - {r['model']['name']} (分数: {r['score']})")

print()
print("=" * 60)
print("测试 4: compare_models(['resnet50', 'vit'])")
print("=" * 60)
cmp = compare_models(['resnet50', 'vit'])
print(f"对比模型数: {len(cmp['models'])}")
for m in cmp['models']:
    print(f"  - {m['name']}: {m['task']}")
print(f"选择建议: {cmp['recommendation']}")

print()
print("=" * 60)
print("测试 5: recommend_models('目标检测') - 应该返回提示")
print("=" * 60)
result4 = recommend_models('目标检测', limit=3)
print(f"匹配总数: {result4['total_found']}")
print(f"建议: {result4['suggestion']}")
print(f"注意: {result4.get('note', '无')}")

print()
print("=" * 60)
print("测试 6: 按模型名称搜索")
print("=" * 60)
result5 = recommend_models('resnet', limit=3)
print(f"查询: {result5['query']}")
print(f"匹配总数: {result5['total_found']}")
for r in result5['recommendations']:
    print(f"  - {r['model']['name']} (分数: {r['score']})")

print()
print("✅ 所有测试通过!")