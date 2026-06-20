"""
两相沸腾换热关联式库
=============================================
提供多种微通道两相换热关联式,便于对比验证

已实现:
- Gungor-Winterton (1987): 通用两相, 适合常规尺度
- Kim-Mudawar (2013): 微通道专用, 适合 Dh < 2mm
"""

from .gungor_winterton import GungorWinterton
from .kim_mudawar import KimMudawar

__all__ = ["GungorWinterton", "KimMudawar"]
