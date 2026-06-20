"""
Gungor-Winterton (1987) 两相沸腾换热关联式
=============================================

h_tp = E * h_l + S * h_pool

其中:
    E = 1 + 24000*Bo^1.16 + 1.37*(1/Co)^0.86   (对流增强因子)
    S = 1 / (1 + 1.15e-6 * E^2 * Re_l^1.17)      (核态沸腾抑制因子)
    h_l = Nu_l * k_l / Dh                          (全液相对流换热系数)
    h_pool = Cooper (1984) 池沸腾关联式

适用范围: 常规尺度管道 (Dh > 2mm), 也适用于部分微通道工况
参考文献: Gungor KE, Winterton RHS. "A general correlation for flow boiling in tubes and annuli",
         Int. J. Heat Mass Transfer, 1987.
"""

import numpy as np
from typing import Dict


class GungorWinterton:
    """Gungor-Winterton (1987) 两相沸腾换热关联式"""

    name = "Gungor-Winterton (1987)"
    suitable_for_microchannel = False  # 原始公式为常规尺度设计

    @staticmethod
    def calc_h(Dh: float, G: float, x: float, q_Wm2: float,
               rho_l: float, rho_v: float, mu_l: float, mu_v: float,
               k_l: float, k_v: float, cp_l: float, h_fg: float,
               sigma: float, Pr_l: float, fluid_name: str = "HFE7100") -> Dict:
        """
        计算两相沸腾换热系数 (基于湿面积)

        Returns: dict with h_tp, h_l, h_pool, E, S, Bo, Co
        """
        x_eff = max(x, 1e-6)
        Re_l = G * Dh / mu_l

        # 全液相对流换热系数
        Nu_l = 8.235 if Re_l < 2300 else (0.023 * max(Re_l, 2300) ** 0.8 * Pr_l ** 0.4)
        Nu_l = max(Nu_l, 8.235)
        h_l = Nu_l * k_l / Dh

        # 无量纲数
        Co = ((1 - x_eff) / x_eff) ** 0.8 * np.sqrt(rho_v / rho_l)  # Convection number
        Bo = q_Wm2 / (G * h_fg)  # Boiling number

        # G-W 增强因子 E (加物理上限)
        E = 1.0 + 24000.0 * Bo ** 1.16 + 1.37 * (1.0 / max(Co, 1e-6)) ** 0.86
        E = min(E, 15.0)

        # G-W 抑制因子 S
        S = 1.0 / (1.0 + 1.15e-6 * E ** 2 * max(Re_l, 1) ** 1.17)

        # Cooper (1984) 池沸腾
        p_crit_map = {"water": 220.6e5, "HFE7100": 21.5e5, "R245fa": 36.5e5, "R1233zdE": 35.7e5}
        p_crit = p_crit_map.get(fluid_name, 22e5)
        p_r = 101325.0 / p_crit
        M_map = {"water": 18, "HFE7100": 250, "R245fa": 134, "R1233zdE": 130.5}
        M = M_map.get(fluid_name, 100)

        h_pool = 55.0 * max(p_r, 1e-6) ** (0.12 - 0.4343 * np.log(max(p_r, 1e-6)))
        h_pool *= (-np.log10(max(p_r, 1e-6))) ** (-0.55)
        h_pool *= M ** (-0.5)
        h_pool *= max(q_Wm2, 100) ** 0.67

        h_tp = E * h_l + S * h_pool
        h_tp = max(h_tp, h_l)

        return {
            "h_tp": h_tp,
            "h_l": h_l,
            "h_pool": h_pool,
            "E": E,
            "S": S,
            "Bo": Bo,
            "Co": Co,
            "Re_l": Re_l,
        }
