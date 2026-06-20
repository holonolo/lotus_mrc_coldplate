"""
多关联式对比验证脚本
=============================================
对比 Gungor-Winterton (1987) 和 Kim-Mudawar (2013) 在
仿荷叶歧管微通道冷板中的预测表现

文献基准: h=2.13 W/(cm²·K) @ q=255 W/cm², HFE-7100, m=6g/s

运行: python tests/test_correlations.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.geometry import ManifoldRingChannelGeometry
from core.fluid_properties import FluidProperties
from core.correlations import GungorWinterton, KimMudawar


def run_comparison():
    geo = ManifoldRingChannelGeometry()
    fluid = FluidProperties("HFE7100")

    Dh = geo.hydraulic_diameter * 1e-3  # m
    A_eff = geo.effective_cross_area

    # 文献基准工况
    mass_flows = [3, 6, 9, 12]  # g/s
    heat_fluxes = [50, 100, 150, 200, 255]  # W/cm²

    print("=" * 90)
    print("两相沸腾换热关联式对比: G-W (1987) vs Kim-Mudawar (2013)")
    print("=" * 90)
    print(f"冷板: Dh={geo.hydraulic_diameter:.3f} mm, {geo.n_rings}环×{geo.n_sectors}扇区")
    print(f"工质: HFE-7100, T_sat={fluid.T_sat}°C")
    print(f"文献基准: h=2.13 W/(cm²·K) @ q=255 W/cm², m=6 g/s")
    print()

    # 固定流量, 扫描热流密度
    m_gs = 6.0
    m_dot = m_gs * 1e-3
    G = m_dot / A_eff

    print(f"质量流量 = {m_gs} g/s, G = {G:.1f} kg/(m²·s)")
    print("-" * 90)
    header = (f"{'q [W/cm²]':>10} | {'x_out':>6} | "
              f"{'G-W h':>10} | {'KM h':>10} | {'G-W/文献':>8} | {'KM/文献':>8} | "
              f"{'G-W E':>6} | {'G-W S':>6} | {'KM NBD':>10} | {'KM CBD':>10}")
    print(header)
    print("-" * 90)

    for qf in heat_fluxes:
        # 计算出口干度
        Q = qf * 1e4 * geo.chip_area * 1e-6
        Q_sub = m_dot * fluid.cp_l * max(fluid.T_sat - 20.0, 0)
        Q_tp = max(Q - Q_sub, 0)
        x_out = np.clip(Q_tp / (m_dot * fluid.h_fg), 0, 0.95)
        x_avg = x_out / 2

        q_Wm2 = qf * 1e4

        # G-W 关联式
        gw = GungorWinterton.calc_h(
            Dh, G, x_avg, q_Wm2,
            fluid.rho_l, fluid.rho_v, fluid.mu_l, fluid.mu_v,
            fluid.k_l, fluid.k_v, fluid.cp_l, fluid.h_fg,
            fluid.sigma, fluid.Pr_l, "HFE7100"
        )

        # Kim-Mudawar 关联式
        km = KimMudawar.calc_h(
            Dh, G, x_avg, q_Wm2,
            fluid.rho_l, fluid.rho_v, fluid.mu_l, fluid.mu_v,
            fluid.k_l, fluid.k_v, fluid.cp_l, fluid.h_fg,
            fluid.sigma, fluid.Pr_l, "HFE7100"
        )

        # 转换为 W/(cm²·K)
        gw_h_cm2 = gw["h_tp"] * 1e-4
        km_h_cm2 = km["h_tp"] * 1e-4

        # 与文献基准对比 (2.13 W/cm²K)
        lit_h = 2.13
        gw_ratio = gw_h_cm2 / lit_h
        km_ratio = km_h_cm2 / lit_h

        gw_nbd_cm2 = km["h_NBD"] * 1e-4
        km_cbd_cm2 = km["h_CBD"] * 1e-4

        print(f"{qf:>10.0f} | {x_out:>6.3f} | "
              f"{gw_h_cm2:>10.3f} | {km_h_cm2:>10.3f} | "
              f"{gw_ratio:>8.2f} | {km_ratio:>8.2f} | "
              f"{gw['E']:>6.1f} | {gw['S']:>6.4f} | "
              f"{gw_nbd_cm2:>10.3f} | {km_cbd_cm2:>10.3f}")

    print("-" * 90)
    print("\n说明:")
    print("  G-W h   = Gungor-Winterton 预测换热系数 [W/(cm²·K)]")
    print("  KM h    = Kim-Mudawar 预测换热系数 [W/(cm²·K)]")
    print("  G-W/文献 = G-W 预测值 / 文献基准值(2.13)")
    print("  KM/文献  = KM 预测值 / 文献基准值(2.13)")
    print("  G-W E/S = G-W 增强因子/抑制因子")
    print("  KM NBD/CBD = Kim-Mudawar 核态沸腾/对流沸腾分量")
    print()

    # 不同流量对比
    print("=" * 70)
    print("不同流量下的对比 (q=255 W/cm²)")
    print("=" * 70)
    print(f"{'m [g/s]':>8} | {'G':>8} | {'x_out':>6} | {'G-W h':>10} | {'KM h':>10} | {'差值%':>8}")
    print("-" * 70)

    qf = 255
    for m_gs in mass_flows:
        m_dot = m_gs * 1e-3
        G = m_dot / A_eff

        Q = qf * 1e4 * geo.chip_area * 1e-6
        Q_sub = m_dot * fluid.cp_l * max(fluid.T_sat - 20.0, 0)
        Q_tp = max(Q - Q_sub, 0)
        x_out = np.clip(Q_tp / (m_dot * fluid.h_fg), 0, 0.95)
        x_avg = x_out / 2

        gw = GungorWinterton.calc_h(
            Dh, G, x_avg, qf * 1e4,
            fluid.rho_l, fluid.rho_v, fluid.mu_l, fluid.mu_v,
            fluid.k_l, fluid.k_v, fluid.cp_l, fluid.h_fg,
            fluid.sigma, fluid.Pr_l, "HFE7100"
        )
        km = KimMudawar.calc_h(
            Dh, G, x_avg, qf * 1e4,
            fluid.rho_l, fluid.rho_v, fluid.mu_l, fluid.mu_v,
            fluid.k_l, fluid.k_v, fluid.cp_l, fluid.h_fg,
            fluid.sigma, fluid.Pr_l, "HFE7100"
        )

        gw_h = gw["h_tp"] * 1e-4
        km_h = km["h_tp"] * 1e-4
        diff = (gw_h - km_h) / km_h * 100

        print(f"{m_gs:>8.0f} | {G:>8.1f} | {x_out:>6.3f} | {gw_h:>10.3f} | {km_h:>10.3f} | {diff:>8.1f}%")

    print()
    print("结论:")
    print("  1. Kim-Mudawar 专为微通道设计, 对 Dh<0.5mm 工况更稳健")
    print("  2. G-W 的 E 因子在低G高Bo时需加上限(15), 否则爆炸")
    print("  3. 两个关联式的差异随干度增大而变化")
    print("  4. 推荐主模型使用 Kim-Mudawar, G-W 作为交叉验证")


if __name__ == "__main__":
    run_comparison()
