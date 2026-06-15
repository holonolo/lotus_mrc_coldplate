"""
仿荷叶歧管微通道冷板 - 可视化模块
=============================================
matplotlib / plotly 双引擎
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Arc
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from typing import List, Optional, Dict

from .geometry import ManifoldRingChannelGeometry
from .single_phase import SinglePhaseSimulation, SinglePhaseResult
from .two_phase import TwoPhaseSimulation, TwoPhaseResult, FlowPattern
from .comparison import ComparativeAnalysis, ComparisonPoint
from .fluid_properties import FluidProperties

# 中文字体设置 (fallback to English if CJK fonts unavailable)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def plot_geometry_topview(geo: ManifoldRingChannelGeometry,
                          save_path: str = None) -> plt.Figure:
    """绘制冷板俯视图 - 仿荷叶歧管环形微通道"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # 画芯片区域
    chip_half = geo.chip_length / 2
    chip_rect = plt.Rectangle((-chip_half, -chip_half), geo.chip_length, geo.chip_length,
                               linewidth=2, edgecolor='red', facecolor='lightyellow',
                               label='Chip area', zorder=1)
    ax.add_patch(chip_rect)

    # 画冷板外框
    cp_half = geo.coldplate_length / 2
    cp_rect = plt.Rectangle((-cp_half, -cp_half), geo.coldplate_length, geo.coldplate_width,
                             linewidth=2, edgecolor='black', facecolor='none',
                             label='Cold plate', zorder=0)
    ax.add_patch(cp_rect)

    # 画环形微通道
    for i, r in enumerate(geo.ring_radii):
        n_ch = geo.channels_per_ring[i]
        # 环形通道
        circle = plt.Circle((0, 0), r, linewidth=1.5,
                            edgecolor='steelblue', facecolor='none',
                            linestyle='-', zorder=2)
        ax.add_patch(circle)

        # 径向微通道
        angles = np.linspace(0, 2 * np.pi, n_ch, endpoint=False)
        for angle in angles:
            r_in = geo.inlet_diameter / 2
            r_out = r
            x_in, y_in = r_in * np.cos(angle), r_in * np.sin(angle)
            x_out, y_out = r_out * np.cos(angle), r_out * np.sin(angle)
            ax.plot([x_in, x_out], [y_in, y_out], 'steelblue', linewidth=0.5, zorder=2)

    # 中心入口
    inlet = plt.Circle((0, 0), geo.inlet_diameter / 2, linewidth=2,
                        edgecolor='darkblue', facecolor='lightblue',
                        label='Inlet', zorder=3)
    ax.add_patch(inlet)

    # 出口 (4个角)
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        outlet = plt.Circle((dx * (cp_half - 2), dy * (cp_half - 2)), 1.5,
                            linewidth=1.5, edgecolor='darkred', facecolor='lightsalmon',
                            zorder=3)
        ax.add_patch(outlet)
    # 标注一个出口
    ax.annotate('Outlet', xy=(cp_half - 2, cp_half - 2), fontsize=8,
                xytext=(cp_half + 1, cp_half + 1),
                arrowprops=dict(arrowstyle='->', color='darkred'))

    # 中心标注
    ax.annotate('Inlet', xy=(0, 0), fontsize=10, ha='center', va='center',
                color='darkblue', fontweight='bold')

    ax.set_xlim(-cp_half - 3, cp_half + 3)
    ax.set_ylim(-cp_half - 3, cp_half + 3)
    ax.set_aspect('equal')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title('Lotus Leaf-Inspired MRC Cold Plate - Top View')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def plot_geometry_crosssection(geo: ManifoldRingChannelGeometry,
                                save_path: str = None) -> plt.Figure:
    """绘制冷板截面图"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    # 层高度
    y_base = 0
    y_ch_bottom = y_base + geo.base_thickness
    y_ch_top = y_ch_bottom + geo.channel_height
    y_manifold_top = y_ch_top + geo.manifold_height

    # 基板
    ax.fill_between([-15, 15], y_base, y_ch_bottom, color='peru', alpha=0.7, label='Cu base')
    # 微通道层
    ax.fill_between([-15, 15], y_ch_bottom, y_ch_top, color='lightblue', alpha=0.5, label='Microchannel')
    # 歧管层
    ax.fill_between([-15, 15], y_ch_top, y_manifold_top, color='lightgreen', alpha=0.5, label='Manifold')

    # 画微通道 (锯齿状)
    x = -15
    while x < 15:
        ax.fill_between([x, x + geo.channel_width], y_ch_bottom, y_ch_top,
                        color='white', edgecolor='steelblue', linewidth=0.5)
        # 翅片
        ax.fill_between([x + geo.channel_width, x + geo.channel_width + geo.fin_width],
                        y_ch_bottom, y_ch_top, color='peru', alpha=0.7)
        x += geo.channel_width + geo.fin_width

    # 标注
    ax.annotate('', xy=(16, y_base), xytext=(16, y_ch_bottom),
                arrowprops=dict(arrowstyle='<->', color='black'))
    ax.text(17, (y_base + y_ch_bottom) / 2, f'{geo.base_thickness}mm', fontsize=8, va='center')

    ax.annotate('', xy=(16, y_ch_bottom), xytext=(16, y_ch_top),
                arrowprops=dict(arrowstyle='<->', color='black'))
    ax.text(17, (y_ch_bottom + y_ch_top) / 2, f'{geo.channel_height}mm', fontsize=8, va='center')

    ax.annotate('', xy=(16, y_ch_top), xytext=(16, y_manifold_top),
                arrowprops=dict(arrowstyle='<->', color='black'))
    ax.text(17, (y_ch_top + y_manifold_top) / 2, f'{geo.manifold_height}mm', fontsize=8, va='center')

    # 热流箭头
    for x_pos in np.linspace(-10, 10, 5):
        ax.annotate('', xy=(x_pos, y_ch_bottom), xytext=(x_pos, y_base - 1),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(0, y_base - 1.5, 'Heat flux (q")', ha='center', fontsize=10, color='red')

    # 流动箭头
    ax.annotate('Flow in', xy=(-14, y_ch_top + geo.manifold_height / 2), fontsize=9,
                color='blue', fontweight='bold')
    ax.annotate('', xy=(-12, y_ch_top + geo.manifold_height / 2),
                xytext=(-14, y_ch_top + geo.manifold_height / 2),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    ax.set_xlim(-18, 22)
    ax.set_ylim(-2, y_manifold_top + 1)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('Height [mm]')
    ax.set_title('Cross-Section View')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def plot_comparison_curves(analysis: ComparativeAnalysis,
                           save_path: str = None) -> plt.Figure:
    """绘制单相 vs 两相对比曲线"""
    qf_range = np.linspace(20, 250, 30)
    results = analysis.sweep_comparison(qf_range, flow_rate_gs=6.0)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    sp_h = [r.sp_h_conv for r in results]
    tp_h = [r.tp_h_conv for r in results]
    sp_dP = [r.sp_dP for r in results]
    tp_dP = [r.tp_dP for r in results]
    sp_Rth = [r.sp_Rth for r in results]
    tp_Rth = [r.tp_Rth for r in results]
    sp_COP = [r.sp_COP for r in results]
    tp_COP = [r.tp_COP for r in results]
    sp_Tw = [r.sp_T_wall for r in results]
    tp_Tw = [r.tp_T_wall for r in results]
    h_ratio = [r.h_ratio for r in results]

    # 1. 换热系数
    ax = axes[0, 0]
    ax.plot(qf_range, sp_h, 'b-o', markersize=3, label='SP-Water')
    ax.plot(qf_range, tp_h, 'r-s', markersize=3, label='TP-HFE7100')
    ax.set_xlabel('Heat flux [W/cm²]')
    ax.set_ylabel('h [W/(cm²·K)]')
    ax.set_title('Heat Transfer Coefficient')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 压降
    ax = axes[0, 1]
    ax.plot(qf_range, sp_dP, 'b-o', markersize=3, label='SP-Water')
    ax.plot(qf_range, tp_dP, 'r-s', markersize=3, label='TP-HFE7100')
    ax.set_xlabel('Heat flux [W/cm²]')
    ax.set_ylabel('ΔP [kPa]')
    ax.set_title('Pressure Drop')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 热阻
    ax = axes[0, 2]
    ax.plot(qf_range, sp_Rth, 'b-o', markersize=3, label='SP-Water')
    ax.plot(qf_range, tp_Rth, 'r-s', markersize=3, label='TP-HFE7100')
    ax.set_xlabel('Heat flux [W/cm²]')
    ax.set_ylabel('Rth [(cm²·K)/W]')
    ax.set_title('Thermal Resistance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. COP
    ax = axes[1, 0]
    ax.semilogy(qf_range, sp_COP, 'b-o', markersize=3, label='SP-Water')
    ax.semilogy(qf_range, tp_COP, 'r-s', markersize=3, label='TP-HFE7100')
    ax.set_xlabel('Heat flux [W/cm²]')
    ax.set_ylabel('COP')
    ax.set_title('Coefficient of Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. 壁面温度
    ax = axes[1, 1]
    ax.plot(qf_range, sp_Tw, 'b-o', markersize=3, label='SP-Water')
    ax.plot(qf_range, tp_Tw, 'r-s', markersize=3, label='TP-HFE7100')
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='T_max limit')
    ax.set_xlabel('Heat flux [W/cm²]')
    ax.set_ylabel('T_wall [°C]')
    ax.set_title('Wall Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. 换热增强比
    ax = axes[1, 2]
    ax.plot(qf_range, h_ratio, 'g-^', markersize=3)
    ax.set_xlabel('Heat flux [W/cm²]')
    ax.set_ylabel('h_TP / h_SP')
    ax.set_title('Two-Phase Enhancement Ratio')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Lotus MRC Cold Plate: Single-Phase vs Two-Phase Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def plot_boiling_curve(tp_sim: TwoPhaseSimulation,
                       mass_flow_gs: float = 6.0,
                       save_path: str = None) -> plt.Figure:
    """绘制沸腾曲线 (q" vs ΔT_sat)"""
    qf_range = np.linspace(5, 250, 50)
    delta_T_sat = []
    h_list = []
    pattern_colors = []
    pattern_names = []

    for qf in qf_range:
        res = tp_sim.simulate(qf, mass_flow_gs)
        dT = res.T_wall_avg - res.T_sat
        delta_T_sat.append(max(dT, 0.1))
        h_list.append(res.h_conv_cm2)
        pattern_names.append(res.flow_pattern_name)
        # 颜色映射
        fp = res.flow_pattern
        color_map = {
            FlowPattern.BUBBLY: 0,
            FlowPattern.SLUG: 1,
            FlowPattern.ANNULAR: 2,
            FlowPattern.PULSATING_ANNULAR: 3,
            FlowPattern.MIST: 4,
        }
        pattern_colors.append(color_map.get(fp, 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 沸腾曲线
    scatter = ax1.scatter(delta_T_sat, qf_range, c=pattern_colors, cmap='viridis',
                          s=20, edgecolors='none')
    ax1.set_xlabel('ΔT_sat = T_wall - T_sat [°C]')
    ax1.set_ylabel('Heat flux q" [W/cm²]')
    ax1.set_title('Boiling Curve')
    ax1.grid(True, alpha=0.3)

    # 添加流型图例
    cbar = plt.colorbar(scatter, ax=ax1, ticks=[0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels(['Bubbly', 'Slug', 'Annular', 'Pulsating\nAnnular', 'Mist'])

    # 换热系数
    ax2.plot(qf_range, h_list, 'r-', linewidth=2)
    ax2.set_xlabel('Heat flux q" [W/cm²]')
    ax2.set_ylabel('h [W/(cm²·K)]')
    ax2.set_title('Two-Phase HTC vs Heat Flux')
    ax2.grid(True, alpha=0.3)

    # CHF 线
    chf_val = tp_sim.simulate(100, mass_flow_gs).CHF
    ax2.axvline(x=chf_val, color='red', linestyle='--', alpha=0.7, label=f'CHF={chf_val:.0f} W/cm²')
    ax2.legend()

    fig.suptitle(f'Flow Boiling in Lotus MRC (HFE-7100, G={mass_flow_gs} g/s)', fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def plot_sensitivity_analysis(sim_type: str = "single_phase",
                               save_path: str = None) -> plt.Figure:
    """COP 敏感性分析"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    geo = ManifoldRingChannelGeometry()

    if sim_type == "single_phase":
        sim = SinglePhaseSimulation(geo, FluidProperties("water"))
        title_prefix = "Single-Phase Water"
        qf_range = np.linspace(50, 633, 25)
        mf_range = np.linspace(2, 20, 20)

        # 热流密度敏感性
        COP_vs_qf = []
        for qf in qf_range:
            res = sim.simulate(qf, 5.0)
            COP_vs_qf.append(res.COP)
        axes[0].plot(qf_range, COP_vs_qf, 'b-', linewidth=2)
        axes[0].set_xlabel('Heat flux [W/cm²]')
        axes[0].set_ylabel('COP')

        # 流量敏感性
        COP_vs_mf = []
        for mf in mf_range:
            res = sim.simulate(200, mf)
            COP_vs_mf.append(res.COP)
        axes[1].plot(mf_range, COP_vs_mf, 'g-', linewidth=2)
        axes[1].set_xlabel('Mass flow rate [g/s]')
        axes[1].set_ylabel('COP')

        # 热阻 vs 流量
        Rth_vs_mf = []
        for mf in mf_range:
            res = sim.simulate(200, mf)
            Rth_vs_mf.append(res.thermal_resistance)
        axes[2].plot(mf_range, Rth_vs_mf, 'r-', linewidth=2)
        axes[2].set_xlabel('Mass flow rate [g/s]')
        axes[2].set_ylabel('Rth [(cm²·K)/W]')

    else:
        sim = TwoPhaseSimulation(geo, FluidProperties("HFE7100"))
        title_prefix = "Two-Phase HFE-7100"
        qf_range = np.linspace(20, 250, 25)
        mf_range = np.linspace(3, 12, 20)

        COP_vs_qf = []
        for qf in qf_range:
            res = sim.simulate(qf, 6.0)
            COP_vs_qf.append(res.COP)
        axes[0].plot(qf_range, COP_vs_qf, 'b-', linewidth=2)
        axes[0].set_xlabel('Heat flux [W/cm²]')
        axes[0].set_ylabel('COP')
        axes[0].axvline(x=267, color='red', linestyle='--', alpha=0.5, label='CHF')
        axes[0].legend()

        COP_vs_mf = []
        for mf in mf_range:
            res = sim.simulate(100, mf)
            COP_vs_mf.append(res.COP)
        axes[1].plot(mf_range, COP_vs_mf, 'g-', linewidth=2)
        axes[1].set_xlabel('Mass flow rate [g/s]')
        axes[1].set_ylabel('COP')

        Rth_vs_mf = []
        for mf in mf_range:
            res = sim.simulate(100, mf)
            Rth_vs_mf.append(res.thermal_resistance)
        axes[2].plot(mf_range, Rth_vs_mf, 'r-', linewidth=2)
        axes[2].set_xlabel('Mass flow rate [g/s]')
        axes[2].set_ylabel('Rth [(cm²·K)/W]')

    axes[0].set_title('COP vs Heat Flux')
    axes[1].set_title('COP vs Flow Rate')
    axes[2].set_title('Rth vs Flow Rate')

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{title_prefix} - Sensitivity Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig
