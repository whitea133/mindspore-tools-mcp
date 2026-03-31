#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查目标检测模型"""

import sys
import io
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

data = json.load(open('data/mindspore_official_models.json', encoding='utf-8'))

# 查找目标检测相关模型
print("=" * 60)
print("查找 '目标检测' 相关模型")
print("=" * 60)

# 1. 按 category 查找
det_cat = [m for m in data['models'] if '目标检测' in m.get('category', '')]
print(f"\n按 category='目标检测' 查找: {len(det_cat)}")
for m in det_cat[:5]:
    print(f"  - {m['name']}: task={m.get('task', [])}")

# 2. 按 task 包含 object 查找
obj_task = [m for m in data['models'] if 'object' in str(m.get('task', [])).lower()]
print(f"\n按 task 包含 'object' 查找: {len(obj_task)}")
for m in obj_task[:5]:
    print(f"  - {m['name']}: task={m.get('task', [])}")

# 3. 查看所有唯一的 task 类型
all_tasks = set()
for m in data['models']:
    for t in m.get('task', []):
        all_tasks.add(t)
print(f"\n所有 task 类型: {sorted(all_tasks)}")