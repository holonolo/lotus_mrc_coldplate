"""
仿荷叶歧管微通道冷板 - 几何模型
=============================================
基于浙大吴赞课题组文献:
- Xin Z, et al. Energy, 2025 (单相水冷)
- Xin Z, et al. Energy Conversion and Management, 2026 (两相沸腾)

冷板结构: 歧管层 + 环形微通道层 (仿荷叶叶脉分布)
加热面积: 20mm × 20mm = 400 mm² (文献基准314 mm², 本模型参数可调)
总厚度: 4.2 mm
材料: 铜 + 银烧结组装
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ManifoldRingChannelGeometry:
    """仿荷叶歧管环形微通道冷板几何参数"""

    # ===== 加热区域 =====
    chip_area: float = 314.0          # 芯片面积 [mm²] (文献基准)
    chip_length: float = 20.0         # 芯片边长 [mm] (近似正方形)

    # ===== 冷板总尺寸 =====
    coldplate_length: float = 30.0    # 冷板长度 [mm]
    coldplate_width: float = 30.0     # 冷板宽度 [mm]
    total_thickness: float = 4.2      # 冷板总厚度 [mm]

    # ===== 歧管层 =====
    manifold_height: float = 1.0      # 歧管层高度 [mm]
    inlet_diameter: float = 3.0       # 中心入口直径 [mm]
    n_rings: int = 6                  # 环形歧管数

    # ===== 微通道层 =====
    channel_height: float = 1.0       # 微通道深度 [mm]
    channel_width: float = 0.3        # 微通道宽度 [mm]
    fin_width: float = 0.3            # 翅片厚度 [mm]
    base_thickness: float = 0.5       # 基板厚度 [mm]

    # ===== 环形通道 =====
    ring_spacing: float = 1.5         # 环形通道间距 [mm]
    ring_channel_width: float = 0.3   # 环形通道宽度 [mm]

    # ===== 材料属性 =====
    substrate_material: str = "copper"
    sintering_material: str = "silver"

    def __post_init__(self):
        """计算派生参数"""
        self._compute_derived()

    def _compute_derived(self):
        """计算派生几何参数"""
        # 水力直径
        self.hydraulic_diameter = (2 * self.channel_width * self.channel_height /
                                   (self.channel_width + self.channel_height))
        # 单通道截面积
        self.channel_cross_area = self.channel_width * self.channel_height  # [mm²]

        # 总微通道数 (环形分布估算)
        self.channels_per_ring = []
        self.ring_radii = []
        for i in range(self.n_rings):
            r = self.inlet_diameter / 2 + (i + 0.5) * self.ring_spacing
            self.ring_radii.append(r)
            n_ch = int(2 * np.pi * r / (self.channel_width + self.fin_width))
            self.channels_per_ring.append(n_ch)

        self.total_channels = sum(self.channels_per_ring)

        # 歧管结构: 进液/出液通道交替排列
        # 实际同时流动的有效通道约占总通道的25-30%
        self.active_channel_ratio = 0.267  # 由文献G=85-340 kg/(m²·s)与进口3-12g/s反推
        self.effective_channels = int(self.total_channels * self.active_channel_ratio)

        # 总换热面积
        self.total_heat_transfer_area = 0.0
        for i, n_ch in enumerate(self.channels_per_ring):
            perim = 2 * (self.channel_width + self.channel_height)
            L_ch = 2 * np.pi * self.ring_radii[i] / n_ch
            self.total_heat_transfer_area += n_ch * perim * L_ch
        self.total_heat_transfer_area /= 1e6  # [mm²] → [m²]

        # 孔隙率
        channel_volume = sum(
            n * self.channel_cross_area * 2 * np.pi * self.ring_radii[i] / n
            for i, n in enumerate(self.channels_per_ring)
        )
        chip_footprint = self.chip_area * self.channel_height
        self.porosity = channel_volume / chip_footprint if chip_footprint > 0 else 0.5

        # 有效截面积 (用于质量流速计算)
        self.effective_cross_area = (self.effective_channels * self.channel_cross_area * 1e-6)  # [m²]

        # 铜基材料属性
        self.k_substrate = 401.0     # 铜[W/(m·K)]
        self.k_sinter = 429.0        # 银[W/(m·K)]

    def get_params_dict(self) -> dict:
        """返回所有参数字典"""
        return {
            "chip_area_mm2": self.chip_area,
            "chip_length_mm": self.chip_length,
            "coldplate_L_W_mm": (self.coldplate_length, self.coldplate_width),
            "total_thickness_mm": self.total_thickness,
            "manifold_height_mm": self.manifold_height,
            "channel_height_mm": self.channel_height,
            "channel_width_mm": self.channel_width,
            "fin_width_mm": self.fin_width,
            "base_thickness_mm": self.base_thickness,
            "hydraulic_diameter_mm": round(self.hydraulic_diameter, 4),
            "total_channels": self.total_channels,
            "porosity": round(self.porosity, 4),
            "heat_transfer_area_m2": round(self.total_heat_transfer_area * 1e4, 4),  # cm²
            "n_rings": self.n_rings,
            "ring_radii_mm": [round(r, 2) for r in self.ring_radii],
            "channels_per_ring": self.channels_per_ring,
        }

    def summary(self) -> str:
        """打印几何参数摘要"""
        p = self.get_params_dict()
        lines = [
            "=" * 60,
            "仿荷叶歧管环形微通道冷板 - 几何参数",
            "=" * 60,
            f"芯片面积: {p['chip_area_mm2']} mm²",
            f"芯片边长: {p['chip_length_mm']} mm",
            f"冷板尺寸: {p['coldplate_L_W_mm'][0]}×{p['coldplate_L_W_mm'][1]} mm",
            f"冷板总厚: {p['total_thickness_mm']} mm",
            "-" * 40,
            f"歧管层高度: {p['manifold_height_mm']} mm",
            f"微通道深度: {p['channel_height_mm']} mm",
            f"微通道宽度: {p['channel_width_mm']} mm",
            f"翅片厚度:   {p['fin_width_mm']} mm",
            f"基板厚度:   {p['base_thickness_mm']} mm",
            "-" * 40,
            f"水力直径:   {p['hydraulic_diameter_mm']} mm",
            f"总通道数:   {p['total_channels']}",
            f"孔隙率:     {p['porosity']}",
            f"换热面积:   {p['heat_transfer_area_m2']} cm²",
            f"环形歧管数: {p['n_rings']}",
            "=" * 60,
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    geo = ManifoldRingChannelGeometry()
    print(geo.summary())
    print(f"\n水力直径: {geo.hydraulic_diameter:.3f} mm")
