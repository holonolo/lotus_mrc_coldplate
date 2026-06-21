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
    chip_area: float = 314.16          # 圆形芯片面积 [mm²] (直径20mm)
    chip_length: float = 20.0         # 芯片边长/直径 [mm]

    # ===== 冷板总尺寸 =====
    coldplate_length: float = 30.0    # 冷板长度 [mm]
    coldplate_width: float = 30.0     # 冷板宽度 [mm]
    total_thickness: float = 4.2      # 冷板总厚度 [mm]

    # ===== 歧管层 =====
    manifold_height: float = 1.8      # 歧管层高度 [mm] (文献基准 1.8mm)
    inlet_diameter: float = 4.5       # 中心入口直径 [mm] (文献基准 4.5mm)
    n_rings: int = 13                  # 环形微通道数 (文献基准 13圈)
    n_sectors: int = 16                # 总分流径向流道数 (8窄进液 + 8宽出液)

    # ===== 微通道层 =====
    channel_height: float = 1.4       # 微通道深度 [mm] (文献基准 1.4mm)
    channel_width: float = 0.3        # 微通道宽度 [mm] (文献基准 0.3mm)
    fin_width: float = 0.3            # 翅片厚度 [mm] (文献基准 0.3mm)
    base_thickness: float = 0.5       # 基板厚度 [mm] (文献基准 0.5mm)

    # ===== 环形通道 =====
    ring_spacing: float = 0.6         # 环形通道间距 [mm] (即通道宽+翅片宽 = 0.6mm)
    ring_channel_width: float = 0.3   # 环形通道宽度 [mm]

    # ===== 歧管缝宽 =====
    inlet_slot_width: float = 0.5     # 窄进液歧管缝宽 [mm] (文献基准 0.5mm)
    outlet_slot_width: float = 1.5    # 宽出液歧管缝宽 [mm] (文献基准 1.5mm)

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

        # 环半径计算
        self.ring_radii = []
        for i in range(self.n_rings):
            # 第一个环中心处于 r = r_inlet + 0.5 * pitch
            r = self.inlet_diameter / 2 + (i + 0.5) * self.ring_spacing
            self.ring_radii.append(r)

        # 估算每个环的等效通道数
        self.channels_per_ring = [
            int(2 * np.pi * r / (self.channel_width + self.fin_width))
            for r in self.ring_radii
        ]
        self.total_channels = sum(self.channels_per_ring)

        # 歧管缝隙为扇形(sector)，延长线汇聚于圆心
        # 参考半径取芯片边缘 r_ref = D_chip/2
        r_ref = self.chip_length / 2  # [mm]
        self.theta_in = self.inlet_slot_width / r_ref    # 进液歧管角跨度 [rad]
        self.theta_out = self.outlet_slot_width / r_ref   # 出液歧管角跨度 [rad]

        # 每一个环的有效流动长度 (Lee et al. 2024 1D MMC 模型适配扇形环形通道)
        # 扇形歧管宽度随半径变化: W_in(r) = r·θ_in, W_out(r) = r·θ_out
        # L_flow(r_i) = L_unit(r_i) - W_in(r_i)/4 - W_out(r_i)/4
        #             = r_i · (2π/n_sectors - θ_in/4 - θ_out/4)
        self.L_flow_factor = 2 * np.pi / self.n_sectors - self.theta_in / 4 - self.theta_out / 4
        self.L_flow_rings = [r * self.L_flow_factor for r in self.ring_radii]
        self.L_flow_avg = np.mean(self.L_flow_rings)

        # 并联流动路径数 = n_rings * n_sectors
        # 8个进水口在各环左右分流，共有 8 * 2 = 16条弧，13个环共 13 * 16 = 208条并联路径
        self.effective_channels = self.n_rings * self.n_sectors

        # 实际换热面积计算 (扇形歧管切割后的微通道壁)
        n_in = self.n_sectors // 2
        n_out = self.n_sectors // 2
        # 扇形歧管角跨度总和
        theta_cut_total = n_in * self.theta_in + n_out * self.theta_out
        # 各环实际通道弧长 = 2πr - r·θ_cut_total (单条通道)
        channel_length_per_path = [
            max(2 * np.pi * r - r * theta_cut_total, 0.1)
            for r in self.ring_radii
        ]
        # 总通道长度 = 各环(单条长度 × 该环通道数)
        actual_channel_length = sum(
            L * n for L, n in zip(channel_length_per_path, self.channels_per_ring)
        )

        perim_ch = self.channel_width + 2 * self.channel_height  # 3.1 mm
        A_wet_c = perim_ch * actual_channel_length

        r_inner = self.inlet_diameter / 2
        r_outer = self.ring_radii[-1] + self.ring_spacing / 2
        dr = max(r_outer - r_inner, 1.0)
        # 扇形歧管底面积 = 扇环面积 ½·θ·(R²-r²)
        A_wet_m_bottom = 0.5 * theta_cut_total * (r_outer**2 - r_inner**2)
        # 歧管侧壁面积 (每个扇形两侧径向壁)
        A_wet_m_sides = self.n_sectors * 2 * self.manifold_height * dr
        A_wet_m = A_wet_m_bottom + A_wet_m_sides

        self.total_heat_transfer_area = (A_wet_c + A_wet_m) * 1e-6  # [m²]

        # 孔隙率 (使用总通道体积 / 芯片足迹体积)
        # 总通道体积 = Σ(各环单条通道长度 × 该环通道数 × 单通道截面积)
        channel_volume = sum(
            L * n * self.channel_cross_area
            for L, n in zip(channel_length_per_path, self.channels_per_ring)
        )
        chip_footprint = self.chip_area * self.channel_height
        self.porosity = channel_volume / chip_footprint if chip_footprint > 0 else 0.5

        # 有效截面积 (用于质量流速计算)
        self.effective_cross_area = (self.effective_channels * self.channel_cross_area * 1e-6)  # [m²]

        # 铜基材料属性
        self.k_substrate = 401.0     # 铜[W/(m·K)]
        self.k_sinter = 254.0        # 银烧结[W/(m·K)] (文献基准 254 W/mK)

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
            "n_sectors": self.n_sectors,
            "ring_radii_mm": [round(r, 2) for r in self.ring_radii],
            "channels_per_ring": self.channels_per_ring,
            "L_flow_rings_mm": [round(l, 3) for l in self.L_flow_rings],
            "L_flow_avg_mm": round(self.L_flow_avg, 3),
        }

    def summary(self) -> str:
        """打印几何参数摘要"""
        p = self.get_params_dict()
        lines = [
            "=" * 60,
            "仿荷叶歧管环形微通道冷板 - 几何参数",
            "=" * 60,
            f"芯片面积: {p['chip_area_mm2']} mm^2",
            f"芯片边长: {p['chip_length_mm']} mm",
            f"冷板尺寸: {p['coldplate_L_W_mm'][0]}x{p['coldplate_L_W_mm'][1]} mm",
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
            f"换热面积:   {p['heat_transfer_area_m2']} cm^2",
            f"环形歧管:   {p['n_rings']} 环",
            f"分流扇区:   {p['n_sectors']} 区",
            f"平均有效流程: {p['L_flow_avg_mm']} mm",
            f"各环有效流程: {[round(l, 2) for l in self.L_flow_rings]} mm",
            "=" * 60,
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    geo = ManifoldRingChannelGeometry()
    print(geo.summary())
    print(f"\n水力直径: {geo.hydraulic_diameter:.3f} mm")
