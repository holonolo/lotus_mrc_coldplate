"""
Kim-Mudawar (2013) 微通道两相沸腾换热关联式
=============================================

专门为微通道 (Dh < 2mm) 开发, 基于 10,805 个数据点拟合

h_tp = max(h_NBD, h_CBD)

核态沸腾主导区 (NBD):
    h_NBD = [0.023 * Re_l^0.8 * Pr_l^0.4 + 10.6 * Bo^0.76 * (1/x_eff)^0.76] * k_l / Dh

对流沸腾主导区 (CBD):
    h_CBD = [0.023 * Re_l^0.8 * Pr_l^0.4 + 3.5 * (1/x_eff)^0.5 * (rho_v/rho_l)^0.5] * k_l / Dh

适用范围: Dh = 0.16~2.0mm, 多种工质
参考文献: Kim SM, Mudawar I. "Universal approach to predicting two-phase frictional pressure drop
         for adiabatic and condensing mini/micro-channel flows", Int. J. Heat Mass Transfer, 2013.
         以及 Kim SM, Mudawar I. "Universal approach to predicting heat transfer coefficient
         for condensing mini/micro-channel flow", Int. J. Heat Mass Transfer, 2013.
"""

import numpy as np
from typing import Dict


class KimMudawar:
    """Kim-Mudawar (2013) 微通道两相沸腾换热关联式"""

    name = "Kim-Mudawar (2013)"
    suitable_for_microchannel = True

    @staticmethod
    def calc_h(Dh: float, G: float, x: float, q_Wm2: float,
               rho_l: float, rho_v: float, mu_l: float, mu_v: float,
               k_l: float, k_v: float, cp_l: float, h_fg: float,
               sigma: float, Pr_l: float, fluid_name: str = "HFE7100") -> Dict:
        """
        计算两相沸腾换热系数 (基于湿面积)

        Returns: dict with h_tp, h_NBD, h_CBD, Bo, Co, Re_l
        """
        x_eff = max(x, 1e-6)
        Re_l = G * Dh / mu_l

        # 无量纲数
        Bo = q_Wm2 / (G * h_fg)
        Co = ((1 - x_eff) / x_eff) ** 0.8 * np.sqrt(rho_v / rho_l)

        # 全液相 Nu (Dittus-Boelter)
        Nu_l = 0.023 * max(Re_l, 2300) ** 0.8 * Pr_l ** 0.4
        Nu_l = max(Nu_l, 8.235)

        # 核态沸腾主导区 (NBD)
        term_conv = 0.023 * max(Re_l, 2300) ** 0.8 * Pr_l ** 0.4
        # 注意: (1/x_eff)^0.76 在 x→0 时爆炸, 需加上限
        inv_x = min(1.0 / x_eff, 100.0)  # 物理上限: x_min = 0.01
        term_nb = 10.6 * Bo ** 0.76 * inv_x ** 0.76
        Nu_NBD = term_conv + term_nb
        h_NBD = Nu_NBD * k_l / Dh

        # 对流沸腾主导区 (CBD)
        inv_x_cb = min(1.0 / x_eff, 100.0)
        term_cb = 3.5 * inv_x_cb ** 0.5 * np.sqrt(rho_v / rho_l)
        Nu_CBD = term_conv + term_cb
        h_CBD = Nu_CBD * k_l / Dh

        # 取两者最大值
        h_tp = max(h_NBD, h_CBD)
        h_l = Nu_l * k_l / Dh
        h_tp = max(h_tp, h_l)

        return {
            "h_tp": h_tp,
            "h_NBD": h_NBD,
            "h_CBD": h_CBD,
            "h_l": h_l,
            "Bo": Bo,
            "Co": Co,
            "Re_l": Re_l,
        }
