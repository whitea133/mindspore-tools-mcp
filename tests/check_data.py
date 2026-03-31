#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查模型数据"""

import sys
import io
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

data = json.load(open('data/mindspore_official_models.json', encoding='utf-8'))

# 查找 YOLO 模型
yolo = [m for m in data['models'] if 'yolo' in m.get('id', '').lower() or 'yolo' in m.get('name', '').lower()]
print('YOLO 模型:')
for m in yolo[:5]:
    print(f"  - {m['name']}: {m.get('task', [])}")

# 查找检测任务模型
det = [m for m in data['models'] if 'detection' in str(m.get('task', [])).lower()]
print(f"\n检测任务模型数: {len(det)}")
for m in det[:5]:
    print(f"  - {m['name']}: {m.get('task', [])}")

# 查找 ViT 模型
vit = [m for m in data['models'] if 'vit' in m.get('id', '').lower() or 'vit' in m.get('name', '').lower()]
print(f"\nViT 模型:")
for m in vit[:3]:
    print(f"  - {m['name']}: {m.get('task', [])}, category: {m.get('category', '')}")