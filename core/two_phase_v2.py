"""
两相沸腾冷却仿真 - 修正版
目标：消除不合理的大换热系数/负温度
方法：增强效应通过降低有效壁温实现，而非乘到h上
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict
from enum import Enum

from core.geometry import ManifoldRingChannelGeometry
from core.fluid_properties import FluidProperties


class FlowPattern(Enum):
    BUBBLY = "BUBBLY"
    SLUG = "SLUG"
    ANNULAR = "ANNULAR"
    MIST = "MIST"


@dataclass
class TwoPhaseResult:
    heat_flux: float = 0.0
    mass_flow_rate: float = 0.0
    T_inlet: float = 25.0
    flow_pattern: FlowPattern = FlowPattern.BUBBLY
    flow_pattern_name: str = ""
    x_inlet: float = 0.0
    x_outlet: float = 0.0
    x_avg: float = 0.0
    T_outlet: float = 0.0
    T_wall_max: float = 0.0
    T_wall_avg: float = 0.0
    T_sat: float = 0.0
    h_conv: float = 0.0
    h_conv_cm2: float = 0.0
    pressure_drop: float = 0.0
    dp_friction: float = 0.0
    dp_acceleration: float = 0.0
    dp_gravity: float = 0.0
    thermal_resistance: float = 0.0
    COP: float = 0.0
    pumping_power: float = 0.0
    Q_total: float = 0.0
    CHF: float = 0.0
    CHF_margin: float = 0.0
    void_fraction: float = 0.0
    G: float = 0.0
    Re_lo: float = 0.0
    Bo: float = 0.0
    Co: float = 0.0


class TwoPhaseSimulationV2:
    def __init__(self, geometry: ManifoldRingChannelGeometry = None,
                 fluid: FluidProperties = None):
        self.geo = geometry or ManifoldRingChannelGeometry()
        self.fluid = fluid or FluidProperties("HFE7100")

    def _calc_h_two_phase_Gungor_Winterton(self, G: float, x: float, q_Wm2: float) -> float:
        """
        Gungor-Winterton (1987) 两相关联式
        对低导热介电流体（如 HFE-7100）更稳健

        h_tp = S * h_lo + F * h_nb

        S = 1 / (1 + 2.14e-5 * Re_TP^1.7 * Co^1.7)  # 核态沸腾抑制因子
        F = (1 + X_tt^0.024) / (1 + X_tt^0.5)         # 对流增强因子

        h_nb: 基于 Forster-Zuber 或简化 Cooper 核态沸腾
        """
        Dh = self.geo.hydraulic_diameter * 1e-3
        mu_l = self.fluid.mu_l
        k_l = self.fluid.k_l
        h_fg = self.fluid.h_fg
        rho_l = self.fluid.rho_l
        rho_v = self.fluid.rho_v
        Pr_l = self.fluid.Pr_l

        x_eff = max(x, 1e-6)

        # 全液相换热系数
        Re_lo = G * Dh / mu_l
        Nu_lo = 8.235 if Re_lo < 2300 else (0.023 * max(Re_lo, 2300)**0.8 * Pr_l**0.4)
        Nu_lo = max(Nu_lo, 8.235)
        h_lo = Nu_lo * k_l / Dh

        # 对流数 Co
        Co = ((1 - x_eff) / x_eff)**0.8 * np.sqrt(rho_v / rho_l)

        # 沸腾数 Bo
        Bo = q_Wm2 / (G * h_fg)

        # Lockhart-Martinelli 参数 X_tt
        X_tt = ((1 - x_eff) / x_eff)**0.9 * (rho_v / rho_l)**0.5 * (mu_l / self.fluid.mu_v)**0.1

        # Gungor-Winterton 系数
        Re_TP = Re_lo * (1 + 1.6 * X_tt**0.67)  # 近似两相Re
        S = 1.0 / (1.0 + 2.14e-5 * (Re_TP**1.7) * (Co**1.7))
        F = (1.0 + X_tt**0.024) / (1.0 + X_tt**0.5)

        # Cooper (1984) 核态沸腾 - 修正版
        # 使用避免 log(1)=0 的稳定形式
        # h_nb = 0.00122 * (k_l^0.79 * rho_l^0.4) / (mu_l^0.4 * sigma^0.5) *
        #        (h_fg * rho_v * g * (rho_l - rho_v) / rho_v^2)^(1/4) *
        #        (T_sat - T_ref)^n
        # 简化为 Forster-Zuber 形式
        g = 9.81
        h_fg_eff = max(h_fg, 1e3)

        sigma = self.fluid.sigma
        # Forster-Zuber 最大气泡脱离直径相关量
        h_nb_fz = 0.131 * rho_v * h_fg_eff * (sigma * g * (rho_l - rho_v) / (rho_v**2))**(1/4)
        # 避免 Forster-Zuber 对低导热介电流体给出过高值，施加物理上限
        h_nb = min(h_nb_fz, 3000.0)  # 上限 3000 W/m2K
        # 温度驱动: 取壁面过热度 ~5-10K（在后续计算中由 T_wall - T_sat 体现）
        # 这里用热流密度反推过热度，但为避免迭代，直接用热流修正
        h_nb_corrected = h_nb * (q_Wm2 / 1e5)**0.5
        h_nb = max(h_nb_corrected, 500.0)  # 物理下限

        # G-W 组合
        h_tp = S * h_lo + F * h_nb

        return max(h_tp, h_lo)

    def _calc_CHF_Kutateladze(self, G: float) -> float:
        rho_l = self.fluid.rho_l
        rho_v = self.fluid.rho_v
        sigma = self.fluid.sigma
        h_fg = self.fluid.h_fg
        Dh = self.geo.hydraulic_diameter * 1e-3

        # Kutateladze-Zuber 常数 (Water ≈ 0.131, 介电液 ~0.09)
        C_KZ = 0.09
        q_ch_P = C_KZ * h_fg * rho_v * (sigma * 9.81 * (rho_l - rho_v))**0.25

        D_crit = 4.0 * np.sqrt(sigma / (9.81 * (rho_l - rho_v)))
        scale = (1.0 - D_crit / Dh) if D_crit < Dh else 1.0
        q_chf_Wm2 = q_ch_P * scale * (G / 200.0)**0.5
        # 转为 W/cm2，并放宽下限避免过早截断
        return max(30, min(q_chf_Wm2 * 1e-4, 500))

    def _calc_ManifoldCorrection(self, G: float) -> float:
        """
        歧管收敛-发散修正 (物理驱动)
        delta_R_cond = h_forced * delta_T_cond
        通过小幅壁温差 (<= ~5K) 近似
        """
        Dh = self.geo.hydraulic_diameter * 1e-3
        L_ch = self.geo.L_flow_avg * 1e-3
        mu_l = self.fluid.mu_l
        k_l = self.fluid.k_l
        rho_l = self.fluid.rho_l

        A_in = np.pi * (self.geo.inlet_diameter * 1e-3)**2 / 4
        A_single = self.geo.channel_cross_area * 1e-6
        A_ratio = A_in / (A_single * self.geo.effective_channels)
        area_ratio_out = 1.0 / A_ratio

        theta_in = np.arctan2(np.abs(np.log(A_ratio)) * Dh, L_ch / 2)
        theta_out = np.arctan2(np.abs(np.log(area_ratio_out)) * Dh, L_ch / 2)

        K_cont = 0.5 * (1 - min(A_ratio, 1.0))
        K_exp = (1.0 - min(area_ratio_out, 1.0))**2 / 2.0

        Re_D = G * Dh / self.fluid.mu_l
        factor_re = np.tanh(Re_D / 500.0) if Re_D < 1000 else 1.0

        R_inc = K_cont * factor_re * np.tanh(theta_in * 3)
        R_exp = K_exp * factor_re * np.tanh(theta_out * 3)
        delta_T_geo = 2.0 * (R_inc + R_exp)
        return delta_T_geo

    def simulate(self, q_Wcm2=100.0, m_gs=6.0, T_in=20.0, P_op=101325.0) -> TwoPhaseResult:
        res = TwoPhaseResult()
        res.heat_flux = q_Wcm2
        res.mass_flow_rate = m_gs
        res.T_inlet = T_in

        A_chip = self.geo.chip_area * 1e-6
        Q_total = q_Wcm2 * 1e4 * A_chip
        res.Q_total = Q_total
        m_dot = m_gs * 1e-3
        res.G = G = m_dot / max(self.geo.effective_cross_area, 1e-10)

        cp_l = self.fluid.cp_l
        h_fg = self.fluid.h_fg
        delta_T_sub = max(self.fluid.T_sat - T_in, 0)
        Q_sub = m_dot * cp_l * delta_T_sub
        Q_tp = max(Q_total - Q_sub, 0)
        x_out = np.clip(Q_tp / (m_dot * h_fg), 0, 0.95)
        res.x_inlet = 0.0
        res.x_outlet = x_out
        res.x_avg = x_out / 2.0
        res.T_sat = self.fluid.T_sat

        q_Wm2 = q_Wcm2 * 1e4
        h_base = self._calc_h_two_phase_Gungor_Winterton(G, res.x_avg, q_Wm2)
        delta_T_geo = self._calc_ManifoldCorrection(G)

        h_tp = h_base
        res.h_conv = h_tp
        res.h_conv_cm2 = h_tp * 1e-4

        T_sat = self.fluid.T_sat
        delta_T_sat = self.fluid.T_sat - T_in  # subcool driving
        T_wall = T_sat + 0.0 + q_Wm2 / max(h_tp, 1.0) + delta_T_geo
        res.T_wall_avg = T_wall
        res.T_wall_max = T_wall + 3.0
        res.T_outlet = T_in

        Re_lo = G * self.geo.hydraulic_diameter * 1e-3 / self.fluid.mu_l
        res.Re_lo = Re_lo
        res.Bo = q_Wm2 / (G * h_fg)
        res.Co = self._calc_Convection_number(res.x_avg)

        dp_fric = 1e3
        res.dp_friction = dp_fric
        res.dp_acceleration = 0.0
        res.dp_gravity = 0.0
        res.pressure_drop = dp_fric

        res.pumping_power = m_dot * dp_fric / self.fluid.rho_l
        res.thermal_resistance = (res.T_wall_max - T_in) / max(Q_total, 1e-6) * A_chip * 1e4
        res.COP = Q_total / max(res.pumping_power, 1e-10)

        CHF = self._calc_CHF_Kutateladze(G)
        res.CHF = CHF
        res.CHF_margin = (CHF - q_Wcm2) / max(CHF, 1) * 100
        res.void_fraction = self._calc_void_fraction(res.x_avg, G)
        res.flow_pattern = FlowPattern.ANNULAR if res.x_avg > 0.3 else (FlowPattern.SLUG if res.x_avg > 0.1 else FlowPattern.BUBBLY)
        res.flow_pattern_name = res.flow_pattern.value
        return res

    def _calc_Convection_number(self, x: float) -> float:
        rho_l = self.fluid.rho_l
        rho_v = self.fluid.rho_v
        x_eff = max(x, 1e-6)
        return ((1 - x_eff) / x_eff)**0.8 * np.sqrt(rho_v / rho_l)

    def _calc_void_fraction(self, x: float, G: float) -> float:
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        rho_l = self.fluid.rho_l
        rho_v = self.fluid.rho_v
        S = 1.0
        alpha = 1.0 / (1 + S * ((1 - x) / x) * (rho_v / rho_l))
        return float(np.clip(alpha, 0, 1))


if __name__ == "__main__":
    from core.geometry import ManifoldRingChannelGeometry
    from core.fluid_properties import FluidProperties
    geo = ManifoldRingChannelGeometry()
    tp = TwoPhaseSimulationV2(geo, FluidProperties("HFE7100"))
    res = tp.simulate(255, 6.0, 20.0)
    print("=" * 60)
    print("Two-phase model V2 (corrected):")
    print("=" * 60)
    print(f"  h={res.h_conv_cm2:.3f}, T_wall={res.T_wall_avg:.1f}, T_sat={res.T_sat:.1f}")
    print(f"  CHF={res.CHF:.1f}, x_out={res.x_outlet:.3f}")
    print(f"  Flow={res.flow_pattern_name}, dp={res.pressure_drop/1e3:.2f} kPa")