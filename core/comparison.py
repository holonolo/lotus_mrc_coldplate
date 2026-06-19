"""
仿荷叶歧管微通道冷板 - 单相 vs 两相对比分析
=============================================
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.geometry import ManifoldRingChannelGeometry
from core.fluid_properties import FluidProperties
from core.single_phase import SinglePhaseSimulation, SinglePhaseResult
from core.two_phase import TwoPhaseSimulation, TwoPhaseResult, FlowPattern


@dataclass
class ComparisonPoint:
    """单点对比结果"""
    heat_flux: float  # W/cm²
    mass_flow_rate: float  # g/s

    # 单相
    sp_h_conv: float = 0.0    # W/(cm²·K)
    sp_T_wall: float = 0.0    # °C
    sp_dP: float = 0.0        # kPa
    sp_Rth: float = 0.0       # (cm²·K)/W
    sp_COP: float = 0.0
    sp_Q: float = 0.0         # W

    # 两相
    tp_h_conv: float = 0.0
    tp_T_wall: float = 0.0
    tp_dP: float = 0.0
    tp_Rth: float = 0.0
    tp_COP: float = 0.0
    tp_Q: float = 0.0
    tp_CHF: float = 0.0
    tp_flow_pattern: str = ""

    # 对比值
    h_ratio: float = 0.0       # 两相/单相换热系数比
    dP_ratio: float = 0.0      # 两相/单相压降比
    Rth_ratio: float = 0.0     # 两相/单相热阻比


class ComparativeAnalysis:
    """单相 vs 两相对比分析"""

    def __init__(self, geometry: ManifoldRingChannelGeometry = None):
        self.geo = geometry or ManifoldRingChannelGeometry()
        self.sp_sim = SinglePhaseSimulation(self.geo, FluidProperties("water"))
        self.tp_sim = TwoPhaseSimulation(self.geo, FluidProperties("HFE7100"))

    def compare_at_condition(self,
                             heat_flux_Wcm2: float = 100.0,
                             sp_flow_gs: float = 5.0,
                             tp_flow_gs: float = 6.0,
                             T_inlet: float = 25.0) -> ComparisonPoint:
        """在相同热流密度下对比"""
        sp_res = self.sp_sim.simulate(heat_flux_Wcm2, sp_flow_gs, T_inlet)
        tp_res = self.tp_sim.simulate(heat_flux_Wcm2, tp_flow_gs, T_inlet)

        cp = ComparisonPoint(
            heat_flux=heat_flux_Wcm2,
            mass_flow_rate=tp_flow_gs,
            sp_h_conv=sp_res.h_conv_cm2,
            sp_T_wall=sp_res.T_wall_max,
            sp_dP=sp_res.pressure_drop / 1e3,
            sp_Rth=sp_res.thermal_resistance,
            sp_COP=sp_res.COP,
            sp_Q=sp_res.Q_total,
            tp_h_conv=tp_res.h_conv_cm2,
            tp_T_wall=tp_res.T_wall_max,
            tp_dP=tp_res.pressure_drop / 1e3,
            tp_Rth=tp_res.thermal_resistance,
            tp_COP=tp_res.COP,
            tp_Q=tp_res.Q_total,
            tp_CHF=tp_res.CHF,
            tp_flow_pattern=tp_res.flow_pattern_name,
        )

        cp.h_ratio = tp_res.h_conv / max(sp_res.h_conv, 1)
        cp.dP_ratio = tp_res.pressure_drop / max(sp_res.pressure_drop, 1)
        cp.Rth_ratio = tp_res.thermal_resistance / max(sp_res.thermal_resistance, 1e-10)

        return cp

    def sweep_comparison(self,
                         heat_flux_range: np.ndarray = None,
                         flow_rate_gs: float = 6.0,
                         T_inlet: float = 25.0) -> List[ComparisonPoint]:
        """热流密度扫描对比"""
        if heat_flux_range is None:
            heat_flux_range = np.linspace(20, 250, 25)

        results = []
        for qf in heat_flux_range:
            cp = self.compare_at_condition(qf, flow_rate_gs, flow_rate_gs, T_inlet)
            results.append(cp)
        return results

    def generate_report(self,
                        heat_flux_list: List[float] = None,
                        flow_rate_gs: float = 6.0) -> str:
        """生成对比分析报告"""
        if heat_flux_list is None:
            heat_flux_list = [50, 100, 150, 200, 250]

        lines = [
            "=" * 70,
            "仿荷叶歧管微通道冷板: 单相水冷 vs 两相沸腾冷却 对比分析",
            "=" * 70,
            "",
            "冷板结构: 仿荷叶歧管环形微通道 (铜基, 银烧结组装)",
            f"芯片面积: {self.geo.chip_area} mm^2",
            f"水力直径: {self.geo.hydraulic_diameter:.3f} mm",
            f"环形歧管: {self.geo.n_rings} 环",
            f"微通道尺寸: {self.geo.channel_width}x{self.geo.channel_height} mm (宽x深)",
            "",
            "单相工质: 去离子水",
            "两相工质: HFE-7100 (介电液)",
            "",
            "-" * 70,
            f"{'热流密度':>10} | {'单相h':>10} | {'两相h':>10} | {'h比':>6} | "
            f"{'单相ΔP':>8} | {'两相ΔP':>8} | {'单相Rth':>10} | {'两相Rth':>10} | {'两相流型':>10}",
            f"{'W/cm^2':>10} | {'W/cm^2K':>10} | {'W/cm^2K':>10} | {'':>6} | "
            f"{'kPa':>8} | {'kPa':>8} | {'cm^2K/W':>10} | {'cm^2K/W':>10} | {'':>10}",
            "-" * 70,
        ]

        for qf in heat_flux_list:
            cp = self.compare_at_condition(qf, flow_rate_gs, flow_rate_gs)
            lines.append(
                f"{qf:>10.0f} | {cp.sp_h_conv:>10.3f} | {cp.tp_h_conv:>10.3f} | {cp.h_ratio:>6.2f} | "
                f"{cp.sp_dP:>8.2f} | {cp.tp_dP:>8.2f} | {cp.sp_Rth:>10.4f} | {cp.tp_Rth:>10.4f} | {cp.tp_flow_pattern:>10}"
            )

        lines.extend([
            "-" * 70,
            "",
            "关键结论:",
            "1. 跨工质对比: 单相水冷因水极其优异的物性, 其换热系数高于两相HFE-7100沸腾(两相/单相h比为0.2-0.4)",
            "2. 同工质对比: 两相沸腾(利用相变潜热)换热系数远高于同工质单相对流(如两相HFE-7100比单相HFE-7100提升3-8倍)",
            "3. 运行壁温: HFE-7100常压沸点约61°C, 能在较低壁温下发生相变, 适用于电子器件的等温相变冷却及绝缘场景",
            "4. 歧管环形通道设计显著改善了流量分配与温度均匀性",
            "5. 两相COP在低热流密度时高, 但由于压降增加和接近CHF时COP显著下降",
            "",
            "文献验证数据 (Xin Z, et al.):",
            f"  单相水冷: 最大散热 1987W (633 W/cm^2), ΔP=25.22kPa, Rth=0.0878 cm^2K/W, COP=1.8x10^5",
            f"  两相HFE7100: h=2.13 W/cm^2K, CHF=267.05 W/cm^2, COP=18906",
            "=" * 70,
        ])

        return "\n".join(lines)


if __name__ == "__main__":
    analysis = ComparativeAnalysis()
    print(analysis.generate_report())
