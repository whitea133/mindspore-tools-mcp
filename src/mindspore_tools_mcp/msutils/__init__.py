"""
msutils - MindSpore 开发必备工具箱

让 MindSpore 开发更简单、更高效
"""

__version__ = "1.0.0"
__author__ = "whitea133"

# 导入主要模块
from . import data
from . import train
from . import analysis
from . import eval as evaluation
from . import security
from . import deploy
from . import nlp
from . import distributed

__all__ = [
    "data",
    "train", 
    "analysis",
    "evaluation",
    "security",
    "deploy",
    "nlp",
    "distributed",
]
