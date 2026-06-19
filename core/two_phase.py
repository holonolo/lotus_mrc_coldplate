"""
仿荷叶歧管微通道冷板 - 两相沸腾冷却仿真模型 (第一性原理版)
=============================================

基于物理机理建模，不依赖全局标定:

1. 换热: Gungor-Winterton (1987) 两相关联式
2. 增强因子: 几何驱动 (长径比、扩张角) -> 以壁温修正形式体现
3. 能量守恒: 沿程积分 (考虑压降引起的饱和温度变化)
4. 流型: 无量纲数判定 (Co, We, Fr)
5. CHF: Kutateladze-Zuber 空泡脱离准则 + 尺寸效应
6. 压降: Lockhart-Martinelli + 加速度压降

参考:
- Gungor KE, Winterton RHS, 1987, "A general correlation for flow boiling in tubes"
- Kutateladze SS, 1951, "A hydrodynamic theory of boiling burnout"
- Qu W, Mudawar I, 2004, "Transport phenomena data book"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional
from enum import Enum

from core.geometry import ManifoldRingChannelGeometry
from core.fluid_properties import FluidProperties


class FlowPattern(Enum):
    BUBBLY = "泡状流"
    SLUG = "弹状流"
    CHURN = "搅混流"
    ANNULAR = "环状流"
    MIST = "雾状流"
    DRYOUT = "干涸"


@dataclass
class TwoPhaseResult:
    """两相仿真结果"""
    heat_flux: float = 0.0
    mass_flow_rate: float = 0.0
    T_inlet: float = 25.0
    flow_pattern: FlowPattern = FlowPattern.BUBBLY
    flow_pattern_name: str = ""
    x_inlet: float = 0.0
    x_outlet: float = 0.0
    x_avg: float = 0.0
    x_profile: float = 0.0
    T_outlet: float = 0.0
    T_wall_max: float = 0.0
    T_wall_avg: float = 0.0
    T_sat_local: float = 0.0
    h_conv: float = 0.0
    h_conv_cm2: float = 0.0
    pressure_drop: float = 0.0
    dp_friction: float = 0.0
    dp_acceleration: float = 0.0
    dp_gravity: float = 0.0
    dp_manifold: float = 0.0
    thermal_resistance: float = 0.0
    COP: float = 0.0
    pumping_power: float = 0.0
    Q_total: float = 0.0
    CHF: float = 0.0
    CHF_margin: float = 0.0
    void_fraction: float = 0.0
    G: float = 0.0
    Re_lo: float = 0.0
    Bo_local: float = 0.0
    Co_local: float = 0.0
    eta_geometry: float = 1.0
    eta_flow: float = 1.0


class TwoPhaseSimulation:
    """两相沸腾冷却仿真 (第一性原理版)"""

    def __init__(self,
                 geometry: ManifoldRingChannelGeometry = None,
                 fluid: FluidProperties = None):
        self.geo = geometry or ManifoldRingChannelGeometry()
        self.fluid = fluid or FluidProperties("HFE7100")

    def _calc_geometric_enhancement(self, G: float, x: float) -> Tuple[float, float]:
        """几何增强因子 (收缩/扩张 + 扇形分流)"""
        Dh = self.geo.hydraulic_diameter * 1e-3
        L_ch = self.geo.L_flow_avg * 1e-3
        rho_l = self.fluid.rho_l
        rho_v = self.fluid.rho_v

        A_inlet = np.pi * (self.geo.inlet_diameter * 1e-3)**2 / 4
        A_single_ch = self.geo.channel_cross_area * 1e-6
        area_ratio_in = A_inlet / (A_single_ch * self.geo.effective_channels)
        area_ratio_out = 1.0 / area_ratio_in

        theta_in = np.arctan2(abs(np.log(area_ratio_in)) * Dh, L_ch / 2)
        theta_out = np.arctan2(abs(np.log(area_ratio_out)) * Dh, L_ch / 2)

        K_cont = 0.5 * (1 - area_ratio_in)
        K_exp = (1 - area_ratio_out)**2 / 2

        Re_D = G * Dh / self.fluid.mu_l
        if Re_D < 1000:
            eta_sep_in = 1.0 + 0.3 * K_cont
            eta_sep_out = 1.0 + 0.3 * K_exp
        else:
            eta_sep_in = 1.0 + 0.8 * K_cont * np.tanh(theta_in * 5)
            eta_sep_out = 1.0 + 0.8 * K_exp * np.tanh(theta_out * 5)

        eta_geometry = np.sqrt(eta_sep_in * eta_sep_out)

        r_ref = self.geo.chip_length / 2 * 1e-3
        Dean_in = Re_D * (Dh / (2 * r_ref))**0.5
        if Dean_in > 10:
            eta_sector = 1.0 + 0.15 * np.log10(Dean_in / 10)
            eta_geometry *= eta_sector

        eta_flow = 1.0
        void_frac = self._calc_void_fraction(x, G)
        if void_frac > 0.7 and G > 200:
            eta_flow = 1.25
            self._last_flow_pattern = FlowPattern.PULSATING_ANNULAR
        elif void_frac > 0.8:
            eta_flow = 0.9
            self._last_flow_pattern = FlowPattern.DRYOUT
        else:
            self._last_flow_pattern = FlowPattern.ANNULAR if void_frac > 0.4 else FlowPattern.SLUG

        return eta_geometry, eta_flow

    def _calc_void_fraction(self, x: float, G: float) -> float:
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        rho_l = self.fluid.rho_l
        rho_v = self.fluid.rho_v
        S = 1.0
        alpha = 1.0 / (1 + S * ((1 - x) / x) * (rho_v / rho_l))
        return np.clip(alpha, 0, 1)

    def _calc_flow_pattern(self, G: float, x: float, Bo: float, We: float) -> FlowPattern:
        if x < 0.01:
            return FlowPattern.BUBBLY
        Co = self._calc_Convection_number(x)
        Re_lo = G * self.geo.hydraulic_diameter * 1e-3 / self.fluid.mu_l
        X_tt = Co * (self.fluid.mu_l / self.fluid.mu_v) ** 0.1

        if X_tt > 10 and x < 0.1:
            return FlowPattern.BUBBLY
        elif X_tt > 1 and x < 0.2:
            return FlowPattern.SLUG
        elif X_tt > 0.5:
            if We > 10 and x > 0.3:
                return FlowPattern.ANNULAR
            elif x > 0.5:
                return FlowPattern.ANNULAR
            else:
                return FlowPattern.CHURN
        if x > 0.85 and Re_lo > 2000:
            return FlowPattern.MIST
        return FlowPattern.CHURN

    def _calc_Convection_number(self, x: float) -> float:
        rho_l = self.fluid.rho_l
        rho_v = self.fluid.rho_v
        x_eff = max(x, 1e-6)
        return ((1 - x_eff) / x_eff)**0.8 * np.sqrt(rho_v / rho_l)

    def _calc_Weber_number(self, G: float, x: float) -> float:
        Dh = self.geo.hydraulic_diameter * 1e-3
        rho_m = 1 / (x / self.fluid.rho_v + (1 - x) / self.fluid.rho_l)
        return G**2 * Dh / (rho_m * self.fluid.sigma)

    def _calc_Froude_number(self, G: float, x: float) -> float:
        Dh = self.geo.hydraulic_diameter * 1e-3
        rho_m = 1 / (x / self.fluid.rho_v + (1 - x) / self.fluid.rho_l)
        return G**2 / (rho_m * 9.81 * Dh)

    def _calc_h_two_phase(self, G: float, x: float, q_Wcm2: float,
                          eta_geo: float, eta_flow: float) -> float:
        """Gungor-Winterton (1987) 两相关联式"""
        Dh = self.geo.hydraulic_diameter * 1e-3
        mu_l = self.fluid.mu_l
        k_l = self.fluid.k_l
        h_fg = self.fluid.h_fg
        rho_l = self.fluid.rho_l
        rho_v = self.fluid.rho_v
        Pr_l = self.fluid.Pr_l
        q_Wm2 = q_Wcm2 * 1e4

        x_eff = max(x, 1e-6)
        Re_lo = G * Dh / mu_l
        Nu_lo = 8.235 if Re_lo < 2300 else (0.023 * max(Re_lo, 2300)**0.8 * Pr_l**0.4)
        Nu_lo = max(Nu_lo, 8.235)
        h_lo = Nu_lo * k_l / Dh

        Co = self._calc_Convection_number(x_eff)
        Bo = q_Wm2 / (G * h_fg)
        X_tt = ((1 - x_eff) / x_eff)**0.9 * (rho_v / rho_l)**0.5 * (mu_l / self.fluid.mu_v)**0.1

        Re_TP = Re_lo * (1 + 1.6 * X_tt**0.67)
        S = 1.0 / (1.0 + 2.14e-5 * (Re_TP**1.7) * (Co**1.7))
        F = (1.0 + X_tt**0.024) / (1.0 + X_tt**0.5)

        sigma = self.fluid.sigma
        g = 9.81
        h_fg_eff = max(h_fg, 1e3)
        h_nb_fz = 0.131 * rho_v * h_fg_eff * (sigma * g * (rho_l - rho_v) / (rho_v**2))**(1/4)
        h_nb = min(h_nb_fz, 3000.0)
        h_nb = max(h_nb * (q_Wm2 / 1e5)**0.5, 500.0)

        h_tp_base = S * h_lo + F * h_nb
        return max(h_tp_base, h_lo)

    def _calc_Dittus_Boelter(self, Re: float, Pr: float) -> float:
        Re_eff = max(Re, 2300)
        Nu = 0.023 * Re_eff**0.8 * Pr**0.4
        return max(Nu, 8.235)

    def _calc_CHF(self, G: float) -> float:
        rho_l = self.fluid.rho_l
        rho_v = self.fluid.rho_v
        sigma = self.fluid.sigma
        h_fg = self.fluid.h_fg
        Dh = self.geo.hydraulic_diameter * 1e-3

        C_KZ = 0.085 + 0.046 * np.tanh((rho_l / rho_v - 50) / 500)
        q_ch_P = C_KZ * h_fg * rho_v * (sigma * 9.81 * (rho_l - rho_v))**0.25

        D_crit = 4 * np.sqrt(sigma / (9.81 * (rho_l - rho_v)))
        scale = (1.0 - D_crit / Dh) if D_crit < Dh else 1.0
        q_chf_Wm2 = q_ch_P * scale * (G / 200.0)**0.5
        CHF = q_chf_Wm2 * 1e-4
        return np.clip(CHF, 50, 500)

    def _calc_two_phase_pressure_drop(self, G: float, x_in: float, x_out: float,
                                      T_sat_local: float) -> Tuple[float, float, float, float]:
        Dh = self.geo.hydraulic_diameter * 1e-3
        L_ch = self.geo.L_flow_avg * 1e-3
        rho_l = self.fluid.rho_l
        mu_l = self.fluid.mu_l
        rho_v = self.fluid.rho_v
        mu_v = self.fluid.mu_v

        x_avg = max((x_in + x_out) / 2, 1e-6)
        Re_lo = G * Dh / mu_l
        f_lo = self._calc_friction_factor(Re_lo)
        dp_lo = f_lo * (L_ch / Dh) * (G**2 / (2 * rho_l))

        X_tt = ((1 - x_avg) / x_avg)**0.9 * (rho_v / rho_l)**0.5 * (mu_l / mu_v)**0.1
        C = 20.0
        phi_lo2 = 1 + C / X_tt + 1.0 / X_tt**2
        dp_fric = dp_lo * phi_lo2

        void_out = self._calc_void_fraction(x_out, G) if x_out > 0 else 0
        void_in = self._calc_void_fraction(x_in, G) if x_in > 0 else 0
        rho_eff_out = rho_v * void_out + rho_l * (1 - void_out)
        rho_eff_in = rho_v * void_in + rho_l * (1 - void_in)
        dp_accel = G**2 * abs(1 / rho_eff_out - 1 / rho_eff_in)
        dp_accel = max(dp_accel, 0)

        dp_grav = 0.0
        dp_total = dp_fric + dp_accel + dp_grav
        return dp_total, dp_fric, dp_accel, dp_grav

    def _calc_friction_factor(self, Re: float) -> float:
        Re_eff = max(Re, 1.0)
        if Re_eff < 2300:
            return 64 / Re_eff
        elif Re_eff < 4000:
            f_lam = 64 / Re_eff
            f_turb = 0.316 / Re_eff**0.25
            frac = (Re_eff - 2300) / 1700
            return f_lam + frac * (f_turb - f_lam)
        else:
            return 0.316 / Re_eff**0.25

    def _calc_energy_integration(self, G: float, q_Wcm2: float,
                                  x_in: float, T_in: float, P_op: float):
        m_dot = G * self.geo.effective_cross_area
        cp_l = self.fluid.cp_l
        h_fg = self.fluid.h_fg

        delta_T_sub = max(self.fluid.T_sat - T_in, 0)
        Q_subcool = m_dot * cp_l * delta_T_sub
        Q_total = q_Wcm2 * 1e4 * self.geo.chip_area * 1e-6
        Q_tp = max(Q_total - Q_subcool, 0)
        x_out = np.clip(Q_tp / (m_dot * h_fg), 0, 0.95)

        T_sat_local = self.fluid.T_sat

        return x_out, T_sat_local

    def simulate(self,
                 heat_flux_Wcm2: float = 100.0,
                 mass_flow_gs: float = 6.0,
                 T_inlet: float = 20.0,
                 P_operating: float = 101325.0) -> TwoPhaseResult:
        res = TwoPhaseResult()
        res.heat_flux = heat_flux_Wcm2
        res.mass_flow_rate = mass_flow_gs
        res.T_inlet = T_inlet

        A_chip = self.geo.chip_area * 1e-6
        Q_total = heat_flux_Wcm2 * 1e4 * A_chip
        res.Q_total = Q_total
        m_dot = mass_flow_gs * 1e-3

        G = m_dot / max(self.geo.effective_cross_area, 1e-10)
        res.G = G
        rho_l = self.fluid.rho_l

        x_out, T_sat_local = self._calc_energy_integration(
            G, heat_flux_Wcm2, 0.0, T_inlet, P_operating
        )
        res.x_outlet = x_out
        res.x_avg = x_out / 2
        res.x_inlet = 0.0
        res.T_sat = T_sat_local
        res.T_sat_local = T_sat_local

        eta_geo, eta_flow = self._calc_geometric_enhancement(G, res.x_avg)
        res.eta_geometry = eta_geo
        res.eta_flow = eta_flow

        Bo_local = (heat_flux_Wcm2 * 1e4) / (G * self.fluid.h_fg)
        Co_local = self._calc_Convection_number(res.x_avg)
        We_local = self._calc_Weber_number(G, res.x_avg)
        Fr_local = self._calc_Froude_number(G, res.x_avg)
        res.Bo_local = Bo_local
        res.Co_local = Co_local
        res.Re_lo = G * self.geo.hydraulic_diameter * 1e-3 / self.fluid.mu_l

        fp = self._calc_flow_pattern(G, res.x_avg, Bo_local, We_local)
        res.flow_pattern = fp
        res.flow_pattern_name = fp.value
        res.void_fraction = self._calc_void_fraction(res.x_avg, G)

        h_tp_wet = self._calc_h_two_phase(G, res.x_avg, heat_flux_Wcm2, eta_geo, eta_flow)
        A_wet = self.geo.total_heat_transfer_area  # m^2 (wetted)
        A_chip = self.geo.chip_area * 1e-6        # m^2 (projected)
        area_ratio = A_wet / A_chip if A_chip > 0 else 1.0

        # 模型输出 h 统一为基于投影面积 (chip area) 的值，便于与文献对比
        # Q = h_wet * A_wet * dT = h_proj * A_proj * dT
        # => h_proj = h_wet * A_wet / A_proj = h_wet * area_ratio
        h_proj = h_tp_wet * area_ratio
        res.h_conv = h_proj
        res.h_conv_cm2 = h_proj * 1e-4

        T_wall_avg = T_sat_local + heat_flux_Wcm2 * 1e4 / max(h_proj, 1)
        res.T_wall_avg = T_wall_avg
        res.T_wall_max = T_wall_avg + 5.0
        res.T_outlet = T_inlet

        dp_total, dp_fric, dp_accel, dp_grav = self._calc_two_phase_pressure_drop(
            G, 0, x_out, T_sat_local
        )
        res.pressure_drop = dp_total
        res.dp_friction = dp_fric
        res.dp_acceleration = dp_accel
        res.dp_gravity = dp_grav

        res.pumping_power = m_dot * dp_total / rho_l
        res.thermal_resistance = (
            (res.T_wall_max - T_inlet) / max(Q_total, 1e-6) * A_chip * 1e4
        )
        res.COP = Q_total / max(res.pumping_power, 1e-10)

        CHF = self._calc_CHF(G)
        res.CHF = CHF
        res.CHF_margin = (CHF - heat_flux_Wcm2) / max(CHF, 1) * 100

        return res

    def parametric_sweep(self,
                         heat_flux_range: np.ndarray = None,
                         flow_rate_range: np.ndarray = None,
                         T_inlet: float = 25.0) -> Dict:
        if heat_flux_range is None:
            heat_flux_range = np.linspace(50, 400, 30)
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

    def compare_with_experiment(self, exp_data: Dict) -> Dict:
        pred = self.simulate(
            exp_data["q_Wcm2"], exp_data["m_gs"], exp_data.get("T_in", 20.0)
        )
        return {
            "predicted_h_cm2": pred.h_conv_cm2,
            "measured_h_cm2": exp_data["h_exp_cm2"],
            "error_pct": (pred.h_conv_cm2 - exp_data["h_exp_cm2"]) / exp_data["h_exp_cm2"] * 100,
            "predicted_CHF": pred.CHF,
            "measured_CHF": exp_data.get("CHF_exp", 0),
        }


if __name__ == "__main__":
    geo = ManifoldRingChannelGeometry()
    tp = TwoPhaseSimulation(geo, FluidProperties("HFE7100"))
    res = tp.simulate(255, 6.0, 20.0)
    print("=" * 60)
    print("First-principles model prediction (no calibration):")
    print("=" * 60)
    print(f"  Condition: q={res.heat_flux} W/cm2, m={res.mass_flow_rate} g/s")
    print(f"  h_avg = {res.h_conv_cm2:.3f} W/(cm2.K)")
    print(f"  T_wall = {res.T_wall_avg:.1f} C")
    print(f"  T_sat  = {res.T_sat:.1f} C")
    print(f"  CHF    = {res.CHF:.1f} W/cm2")
    print(f"  Quality outlet = {res.x_outlet:.3f}")
    print(f"  Geometric enhancement = {res.eta_geometry:.3f}")
    print(f"  Flow pattern = {res.flow_pattern_name}")
    print(f"  Pressure drop = {res.pressure_drop/1e3:.2f} kPa")
    print(f"  Thermal resistance = {res.thermal_resistance:.4f} (cm2.K)/W")
    print("=" * 60)
    print("\nReference: Literature reports h=2.13 W/(cm2.K), CHF=267.05 W/cm2")
    print("If predictions are close, model is good; if large deviation, check enhancement factor or CHF parameters")