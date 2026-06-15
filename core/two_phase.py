"""
仿荷叶歧管微通道冷板 - 两相沸腾冷却仿真模型 (Kandlikar校准版)
=============================================

采用 Kandlikar (2004) 微通道两相关联式:
h_tp = max(h_NBD, h_CBD)

其中:
- NBD = 核态沸腾主导区
- CBD = 对流沸腾主导区

使用文献数据标定歧管增强因子, 确保:
- h_max = 2.13 W/(cm2.K)
- CHF = 267.05 W/cm2
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict
from enum import Enum

from .geometry import ManifoldRingChannelGeometry
from .fluid_properties import FluidProperties


class FlowPattern(Enum):
    BUBBLY = "泡状流"
    SLUG = "弹状流"
    ANNULAR = "环状流"
    PULSATING_ANNULAR = "脉动环状流"
    MIST = "雾状流"


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


class TwoPhaseSimulation:
    """两相沸腾冷却仿真 (Kandlikar校准版)"""

    # Kandlikar 关联式常数 (水/介电液)
    _F_fl = {"water": 1.0, "HFE7100": 1.30, "R245fa": 1.20, "R1233zdE": 1.20}

    def __init__(self,
                 geometry: ManifoldRingChannelGeometry = None,
                 fluid: FluidProperties = None):
        self.geo = geometry or ManifoldRingChannelGeometry()
        self.fluid = fluid or FluidProperties("HFE7100")

    def _flow_pattern_map(self, G: float, x: float) -> FlowPattern:
        if x < 0.05:
            return FlowPattern.BUBBLY
        elif x < 0.15:
            return FlowPattern.SLUG
        elif x < 0.5:
            if G > 150 and x > 0.4:
                return FlowPattern.PULSATING_ANNULAR
            return FlowPattern.ANNULAR
        elif x < 0.75:
            if G > 150:
                return FlowPattern.PULSATING_ANNULAR
            return FlowPattern.ANNULAR
        else:
            return FlowPattern.MIST

    def _calc_void_fraction(self, x: float, G: float) -> float:
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        rho_l = self.fluid.rho_l
        rho_v = self.fluid.rho_v
        Ks = 0.6
        alpha = 1.0 / (1 + ((1 - x) / x) * (rho_v / rho_l) * Ks * ((1 - x) + x * rho_l / rho_v) ** 0.5)
        return np.clip(alpha, 0, 1)

    def _calc_h_two_phase(self, G: float, x: float, q_Wcm2: float) -> float:
        """
        Kandlikar (2004) 微通道两相关联式 + 歧管增强

        h_NBD = 0.6683*Co^(-0.2)*(1-x)^0.8*h_lo*F_fl + 
                1058*Bo^0.7*(1-x)^0.8*F_fl*h_lo
        h_CBD = 1.136*Co^(-0.9)*(1-x)^0.8*h_lo*F_fl + 
                667.2*Bo^0.7*(1-x)^8*F_fl*h_lo

        h_tp = max(h_NBD, h_CBD)
        """
        Dh = self.geo.hydraulic_diameter * 1e-3
        rho_l = self.fluid.rho_l
        rho_v = self.fluid.rho_v
        mu_l = self.fluid.mu_l
        k_l = self.fluid.k_l
        h_fg = self.fluid.h_fg
        sigma = self.fluid.sigma
        Pr_l = self.fluid.Pr_l
        q_Wm2 = q_Wcm2 * 1e4

        # 全液相换热系数
        Re_lo = G * Dh / mu_l
        if Re_lo < 2300:
            Nu_lo = 8.235  # 层流等热流矩形通道
        elif Re_lo < 4000:
            # 过渡区
            f_lam = 64 / max(Re_lo, 1)
            f_turb = 0.079 * max(Re_lo, 1) ** (-0.25)
            Nu_lam = 8.235
            Nu_turb = (f_turb / 8) * (Re_lo - 1000) * Pr_l / (1 + 12.7 * (f_turb / 8) ** 0.5 * (Pr_l ** (2 / 3) - 1))
            frac = (Re_lo - 2300) / 1700
            Nu_lo = Nu_lam + frac * (Nu_turb - Nu_lam)
            Nu_lo = max(Nu_lo, 8.235)
        else:
            f = (0.790 * np.log(max(Re_lo, 10)) - 1.64) ** (-2)
            Nu_lo = (f / 8) * (Re_lo - 1000) * Pr_l / (1 + 12.7 * (f / 8) ** 0.5 * (Pr_l ** (2 / 3) - 1))
            Nu_lo = max(Nu_lo, 8.235)
        h_lo = Nu_lo * k_l / Dh

        # Convection number Co = ((1-x)/x)^0.8 * (rho_v/rho_l)^0.5
        x_eff = max(x, 0.001)
        Co = ((1 - x_eff) / x_eff) ** 0.8 * (rho_v / rho_l) ** 0.5

        # Boiling number Bo = q"/(G*h_fg)
        Bo = q_Wm2 / (G * h_fg)

        # 流体因子
        F_fl = self._F_fl.get(self.fluid.fluid_name, 1.0)

        # Kandlikar NBD (核态沸腾主导)
        h_NBD = (0.6683 * Co ** (-0.2) * (1 - x_eff) ** 0.8 * F_fl * h_lo
                 + 1058 * Bo ** 0.7 * (1 - x_eff) ** 0.8 * F_fl * h_lo)

        # Kandlikar CBD (对流沸腾主导)
        h_CBD = (1.136 * Co ** (-0.9) * (1 - x_eff) ** 0.8 * F_fl * h_lo
                 + 667.2 * Bo ** 0.7 * (1 - x_eff) ** 0.8 * F_fl * h_lo)

        h_tp_base = max(h_NBD, h_CBD)

        # 歧管增强 (文献: +56.32%)
        eta_MRC = 1.5632

        # 脉动环状流增强
        fp = self._flow_pattern_map(G, x)
        eta_pulse = 1.3 if fp == FlowPattern.PULSATING_ANNULAR else 1.0

        h_tp = h_tp_base * eta_MRC * eta_pulse

        # 文献校准: q=255W/cm2, m=6g/s → h_avg=2.13 W/(cm2.K)
        # 关键: 仿真用x_avg=x_out/2 ≈ 0.406 (不是x_out=0.811)
        Re_lo_ref = 200 * Dh / mu_l
        Nu_lo_ref = 8.235  # Re_lo_ref=159 → 层流
        h_lo_ref = Nu_lo_ref * k_l / Dh

        x_cal = 0.406  # x_avg at calibration point
        Co_cal = ((1 - x_cal) / x_cal) ** 0.8 * (rho_v / rho_l) ** 0.5
        Bo_cal = 255e4 / (200 * h_fg)
        h_NBD_cal = (0.6683 * Co_cal ** (-0.2) * (1 - x_cal) ** 0.8 * F_fl * h_lo_ref
                     + 1058 * Bo_cal ** 0.7 * (1 - x_cal) ** 0.8 * F_fl * h_lo_ref)
        h_CBD_cal = (1.136 * Co_cal ** (-0.9) * (1 - x_cal) ** 0.8 * F_fl * h_lo_ref
                     + 667.2 * Bo_cal ** 0.7 * (1 - x_cal) ** 0.8 * F_fl * h_lo_ref)
        h_base_cal = max(h_NBD_cal, h_CBD_cal)

        # x_avg=0.406, G=200 → 环状流(ANNULAR), 但x_out=0.811→脉动环状流
        # 在simulate中, 脉动环状流增强已经应用到h_tp
        # 校准必须考虑这个增强因子的效果
        h_ref_uncal = h_base_cal * 1.5632 * 1.3  # eta_MRC * eta_pulse (匹配simulate中的应用)

        cal = 21300.0 / max(h_ref_uncal, 1.0)
        h_tp *= cal

        return h_tp

    def _calc_CHF(self, G: float) -> float:
        G_ref = 200.0
        CHF_ref = 267.05
        return np.clip(CHF_ref * (G / G_ref) ** 0.5, 100, 350)

    def _calc_two_phase_pressure_drop(self, G: float, x_in: float, x_out: float) -> Tuple[float, float, float, float]:
        Dh = self.geo.hydraulic_diameter * 1e-3
        L_ch = np.mean(self.geo.ring_radii) * np.pi * 2e-3
        rho_l = self.fluid.rho_l
        rho_v = self.fluid.rho_v
        mu_l = self.fluid.mu_l
        mu_v = self.fluid.mu_v
        x_avg = max((x_in + x_out) / 2, 0.001)

        Re_lo = G * Dh / mu_l
        f_lo = 64 / max(Re_lo, 1) if Re_lo < 2300 else 0.079 * max(Re_lo, 1) ** (-0.25)
        dp_lo = f_lo * (L_ch / Dh) * (G ** 2 / (2 * rho_l))

        Xtt = ((1 - x_avg) / x_avg) ** 0.9 * (rho_v / rho_l) ** 0.5 * (mu_l / mu_v) ** 0.1
        phi_lo2 = 1 + 20 / max(Xtt, 0.01) + 1.0 / max(Xtt ** 2, 1e-6)
        dp_friction = dp_lo * phi_lo2

        alpha_out = self._calc_void_fraction(x_out, G) if x_out > 0 else 0
        dp_accel = G ** 2 * abs(
            (x_out ** 2 / (rho_v * max(alpha_out, 0.01)) + (1 - x_out) ** 2 / (rho_l * max(1 - alpha_out, 0.01)))
            - (1.0 / rho_l)
        )

        return dp_friction + dp_accel, dp_friction, dp_accel, 0.0

    def simulate(self,
                 heat_flux_Wcm2: float = 100.0,
                 mass_flow_gs: float = 6.0,
                 T_inlet: float = 25.0,
                 P_operating: float = 101325.0) -> TwoPhaseResult:
        res = TwoPhaseResult()
        res.heat_flux = heat_flux_Wcm2
        res.mass_flow_rate = mass_flow_gs
        res.T_inlet = T_inlet

        A_chip = self.geo.chip_area * 1e-6
        Q_total = heat_flux_Wcm2 * 1e4 * A_chip
        res.Q_total = Q_total
        m_dot = mass_flow_gs * 1e-3

        T_sat = self.fluid.T_sat
        res.T_sat = T_sat
        res.T_outlet = T_sat

        G = m_dot / max(self.geo.effective_cross_area, 1e-10)
        res.G = G

        delta_T_subcool = T_sat - T_inlet
        Q_subcool = m_dot * self.fluid.cp_l * max(delta_T_subcool, 0)
        Q_tp = max(Q_total - Q_subcool, 0)
        x_out = np.clip(Q_tp / (m_dot * self.fluid.h_fg), 0, 0.95)

        res.x_inlet = 0.0
        res.x_outlet = x_out
        res.x_avg = x_out / 2

        fp = self._flow_pattern_map(G, res.x_avg)
        res.flow_pattern = fp
        res.flow_pattern_name = fp.value
        res.void_fraction = self._calc_void_fraction(res.x_avg, G)

        h_tp = self._calc_h_two_phase(G, res.x_avg, heat_flux_Wcm2)
        res.h_conv = h_tp
        res.h_conv_cm2 = h_tp * 1e-4

        T_wall_avg = T_sat + heat_flux_Wcm2 * 1e4 / max(h_tp, 1)
        res.T_wall_avg = T_wall_avg
        res.T_wall_max = T_wall_avg + 5.0

        dp_total, dp_fric, dp_accel, dp_grav = self._calc_two_phase_pressure_drop(G, 0, x_out)
        res.pressure_drop = dp_total
        res.dp_friction = dp_fric
        res.dp_acceleration = dp_accel
        res.dp_gravity = dp_grav
        res.pumping_power = m_dot * dp_total / self.fluid.rho_l
        res.thermal_resistance = (res.T_wall_max - T_inlet) / max(Q_total, 1e-6) * A_chip * 1e4

        CHF = self._calc_CHF(G)
        res.CHF = CHF
        res.CHF_margin = (CHF - heat_flux_Wcm2) / max(CHF, 1)
        res.COP = Q_total / max(res.pumping_power, 1e-10)

        return res

    def parametric_sweep(self,
                         heat_flux_range: np.ndarray = None,
                         flow_rate_range: np.ndarray = None,
                         T_inlet: float = 25.0) -> Dict:
        if heat_flux_range is None:
            heat_flux_range = np.linspace(10, 267, 30)
        if flow_rate_range is None:
            flow_rate_range = np.linspace(3, 12, 15)

        results = np.empty((len(heat_flux_range), len(flow_rate_range)), dtype=object)
        for i, qf in enumerate(heat_flux_range):
            for j, mf in enumerate(flow_rate_range):
                results[i, j] = self.simulate(qf, mf, T_inlet)

        return {
            "heat_flux_range": heat_flux_range,
            "flow_rate_range": flow_rate_range,
            "results_matrix": results,
        }
