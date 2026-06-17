"""
仿荷叶歧管微通道冷板 - 项目入口脚本
=============================================
运行完整仿真并生成对比分析报告

用法:
    python main.py                    # 运行默认工况
    python main.py --heat-flux 200    # 指定热流密度
    python main.py --all-figures      # 生成所有图表
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.geometry import ManifoldRingChannelGeometry
from core.fluid_properties import FluidProperties
from core.single_phase import SinglePhaseSimulation
from core.two_phase import TwoPhaseSimulation
from core.comparison import ComparativeAnalysis


def main():
    parser = argparse.ArgumentParser(description="仿荷叶歧管微通道冷板仿真")
    parser.add_argument("--heat-flux", type=float, default=100.0, help="热流密度 [W/cm²]")
    parser.add_argument("--flow-rate", type=float, default=6.0, help="质量流量 [g/s]")
    parser.add_argument("--all-figures", action="store_true", help="生成所有图表")
    args = parser.parse_args()

    geo = ManifoldRingChannelGeometry()
    print(geo.summary())

    # 单相
    sp = SinglePhaseSimulation(geo, FluidProperties("water"))
    res_sp = sp.simulate(args.heat_flux, args.flow_rate)
    print(f"\n{'='*60}")
    print(f"单相水冷 @ q={args.heat_flux} W/cm^2, m={args.flow_rate} g/s")
    print(f"{'='*60}")
    print(f"  散热功率:   {res_sp.Q_total:.1f} W")
    print(f"  换热系数:   {res_sp.h_conv_cm2:.3f} W/(cm^2*K)")
    print(f"  最高壁温:   {res_sp.T_wall_max:.1f} °C")
    print(f"  压降:       {res_sp.pressure_drop/1e3:.2f} kPa")
    print(f"  热阻:       {res_sp.thermal_resistance:.4f} (cm^2*K)/W")
    print(f"  COP:        {res_sp.COP:.0f}")
    print(f"  Re:         {res_sp.Re:.0f}")

    # 两相
    tp = TwoPhaseSimulation(geo, FluidProperties("HFE7100"))
    res_tp = tp.simulate(args.heat_flux, args.flow_rate)
    print(f"\n{'='*60}")
    print(f"两相HFE-7100 @ q={args.heat_flux} W/cm^2, m={args.flow_rate} g/s")
    print(f"{'='*60}")
    print(f"  散热功率:   {res_tp.Q_total:.1f} W")
    print(f"  换热系数:   {res_tp.h_conv_cm2:.3f} W/(cm^2*K)")
    print(f"  最高壁温:   {res_tp.T_wall_max:.1f} °C")
    print(f"  压降:       {res_tp.pressure_drop/1e3:.2f} kPa")
    print(f"  热阻:       {res_tp.thermal_resistance:.4f} (cm^2*K)/W")
    print(f"  COP:        {res_tp.COP:.0f}")
    print(f"  CHF:        {res_tp.CHF:.1f} W/cm^2")
    print(f"  CHF裕度:    {res_tp.CHF_margin*100:.1f}%")
    print(f"  出口干度:   {res_tp.x_outlet:.3f}")
    print(f"  流型:       {res_tp.flow_pattern_name}")

    # 对比
    ana = ComparativeAnalysis(geo)
    print(f"\n{ana.generate_report([50, 100, 150, 200, 255])}")

    if args.all_figures:
        from core.visualization import (
            plot_geometry_topview, plot_geometry_crosssection,
            plot_comparison_curves, plot_boiling_curve, plot_sensitivity_analysis,
        )
        os.makedirs("results", exist_ok=True)

        for name, fig_func, fargs in [
            ("geometry_topview", plot_geometry_topview, (geo,)),
            ("geometry_crosssection", plot_geometry_crosssection, (geo,)),
            ("comparison_curves", plot_comparison_curves, (ana,)),
            ("boiling_curve", plot_boiling_curve, (tp, args.flow_rate)),
            ("sensitivity_sp", plot_sensitivity_analysis, ("single_phase",)),
            ("sensitivity_tp", plot_sensitivity_analysis, ("two_phase",)),
        ]:
            fig = fig_func(*fargs)
            fig.savefig(f"results/{name}.png", dpi=200, bbox_inches='tight')
            print(f"  saved: results/{name}.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
