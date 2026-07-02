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
from matplotlib.patches import FancyArrowPatch, Circle, Arc, Wedge
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from typing import List, Optional, Dict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.geometry import ManifoldRingChannelGeometry
from core.single_phase import SinglePhaseSimulation, SinglePhaseResult
from core.two_phase import TwoPhaseSimulation, TwoPhaseResult, FlowPattern
from core.comparison import ComparativeAnalysis, ComparisonPoint
from core.fluid_properties import FluidProperties

# 中文字体设置 (使用 SimHei 或 Microsoft YaHei，以支持中文显示)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def plot_geometry_topview(geo: ManifoldRingChannelGeometry,
                          save_path: str = None) -> plt.Figure:
    """绘制冷板俯视图 - 仿荷叶歧管环形微通道

    结构说明:
    - 中心入口 → 进液歧管分流 → 弧形微通道 → 出液歧管汇集 → 4个出口
    - n_sectors个楔形扇区 (进液与出液歧管交替排布)
    - n_rings圈同心圆环形微通道
    - 无挡板结构
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    r_chip = geo.chip_length / 2   # 热源半径 [mm]
    cp_half = geo.coldplate_length / 2
    r_max = max(geo.ring_radii) + geo.ring_spacing  # 最外圈半径

    # === 1. 冷板外框 ===
    rect = plt.Rectangle((-cp_half, -cp_half), geo.coldplate_length, geo.coldplate_width,
                         linewidth=2.5, edgecolor='black', facecolor='#f5f5f5', zorder=0)
    ax.add_patch(rect)

    # === 2. 同心圆环形微通道 (灰色细线, 较低层zorder) ===
    for r in geo.ring_radii:
        circle = plt.Circle((0, 0), r, linewidth=0.5, edgecolor='#888888',
                            facecolor='none', linestyle='-', zorder=2)
        ax.add_patch(circle)

    # === 3. 扇形歧管 (进液/出液交替 + 壁面交替) ===
    dtheta = 2 * np.pi / geo.n_sectors
    r_ref = r_chip
    theta_in_slot = geo.inlet_slot_width / r_ref     # 进液歧管角跨度
    theta_out_slot = geo.outlet_slot_width / r_ref    # 出液歧管角跨度

    # 壁面宽度 (1mm 对应的角跨度)
    w_wall = 1.0  # mm
    theta_wall = w_wall / r_ref

    # 剩余角度按缝宽比例分配给进液/出液歧管
    rem_angle = dtheta - theta_wall
    theta_in_m = rem_angle * (theta_in_slot / (theta_in_slot + theta_out_slot))
    theta_out_m = rem_angle * (theta_out_slot / (theta_in_slot + theta_out_slot))

    # 绘制歧管扇区和壁面
    for i in range(geo.n_sectors):
        a_center = i * dtheta

        if i % 2 == 0:
            # 窄进液歧管 (浅蓝色填充)
            half_w = theta_in_m / 2
            fc, ec = '#a1c9f4', '#1f77b4'
        else:
            # 宽出液歧管 (浅红色填充)
            half_w = theta_out_m / 2
            fc, ec = '#ff9896', '#d62728'

        a_start = a_center - half_w
        a_end = a_center + half_w

        # 歧管扇区 (半透明, zorder=3)
        wedge = Wedge((0, 0), r_max,
                      np.degrees(a_start), np.degrees(a_end),
                      facecolor=fc, edgecolor=ec, linewidth=1.2, alpha=0.45, zorder=3)
        ax.add_patch(wedge)

        # 歧管之间的壁面 (不透明灰色, zorder=4)
        a_wall_start = a_end
        next_i = i + 1
        if next_i % 2 == 0:
            next_half_w = theta_in_m / 2
        else:
            next_half_w = theta_out_m / 2
        a_wall_end = (i + 1) * dtheta - next_half_w

        wedge_w = Wedge((0, 0), r_max,
                        np.degrees(a_wall_start), np.degrees(a_wall_end),
                        facecolor='#cccccc', edgecolor='#999999', linewidth=0.8, alpha=1.0, zorder=4)
        ax.add_patch(wedge_w)

        # === 6. 流动方向箭头 ===
        r_arrow = r_max * 0.72
        r_arrow_start = r_max * 0.38
        if i % 2 == 0:
            # 进液歧管: 从中心向外 (蓝色箭头)
            ax.annotate('',
                        xy=(r_arrow * np.cos(a_center), r_arrow * np.sin(a_center)),
                        xytext=(r_arrow_start * np.cos(a_center), r_arrow_start * np.sin(a_center)),
                        arrowprops=dict(arrowstyle='->', color=ec, lw=1.8), zorder=5)
        else:
            # 出液歧管: 从外向中心 (红色箭头)
            ax.annotate('',
                        xy=(r_arrow_start * np.cos(a_center), r_arrow_start * np.sin(a_center)),
                        xytext=(r_arrow * np.cos(a_center), r_arrow * np.sin(a_center)),
                        arrowprops=dict(arrowstyle='->', color=ec, lw=1.8), zorder=5)

    # === 出液歧管外端收集区域 (红色, 导向4个出口) ===
    # 出液歧管(奇数i)外端用红色扇环表示收集区域
    r_collect_inner = r_max
    r_collect_outer = cp_half * 0.97
    for i in range(geo.n_sectors):
        if i % 2 != 0:
            a_center = i * dtheta
            half_w = theta_out_m / 2
            a_start = a_center - half_w
            a_end = a_center + half_w
            collect = Wedge((0, 0), r_collect_outer,
                           np.degrees(a_start), np.degrees(a_end),
                           width=r_collect_outer - r_collect_inner,
                           facecolor='#ff9896', edgecolor='#d62728',
                           linewidth=0.8, alpha=0.5, zorder=3)
            ax.add_patch(collect)

    # === 4. 中心入口 ===
    inlet = plt.Circle((0, 0), geo.inlet_diameter / 2,
                       edgecolor='darkblue', facecolor='#a1c9f4', alpha=0.85, zorder=6)
    ax.add_patch(inlet)
    ax.annotate('IN', xy=(0, 0), ha='center', va='center', fontsize=11,
                color='darkblue', fontweight='bold', zorder=7)

    # === 5. 热源区域 (芯片红色虚线圆) ===
    chip = plt.Circle((0, 0), r_chip, linewidth=2.5, edgecolor='red',
                      facecolor='none', linestyle='--', zorder=5)
    ax.add_patch(chip)
    ax.annotate(f'Ø{geo.chip_length:.0f}mm', xy=(r_chip * 0.707, r_chip * 0.707),
                fontsize=9, color='red', fontweight='bold', zorder=7)

    # === 7. 4个出口标记 (四边中点) ===
    outlet_positions = [
        (0, cp_half),       # 上方
        (0, -cp_half),      # 下方
        (cp_half, 0),       # 右方
        (-cp_half, 0),      # 左方
    ]
    for (ox, oy) in outlet_positions:
        # 出口圆
        outlet = plt.Circle((ox, oy), 1.8, edgecolor='#8c564b',
                           facecolor='#ff9896', alpha=0.9, zorder=6)
        ax.add_patch(outlet)
        # "OUT" 标签放在圆外侧
        offset_x = 0 if ox == 0 else (2.5 if ox > 0 else -2.5)
        offset_y = 0 if oy == 0 else (2.5 if oy > 0 else -2.5)
        ax.annotate('OUT', xy=(ox + offset_x, oy + offset_y), ha='center', va='center',
                    fontsize=8, color='#8c564b', fontweight='bold', zorder=7)

    # === 8. 图例 ===
    n_inlet = geo.n_sectors // 2
    n_outlet = geo.n_sectors // 2
    ax.plot([], [], color='#1f77b4', linewidth=4, label=f'进液歧管 ({n_inlet}条)')
    ax.plot([], [], color='#d62728', linewidth=4, label=f'出液歧管 ({n_outlet}条)')
    ax.plot([], [], color='#999999', linewidth=3, label='歧管间壁面')
    ax.plot([], [], color='#888888', linewidth=1, label=f'环形微通道 ({geo.n_rings}圈)')
    ax.plot([], [], color='red', linestyle='--', linewidth=2, label=f'热源区域 Ø{geo.chip_length:.0f}mm')
    ax.scatter([], [], color='#ff9896', edgecolors='#8c564b', s=80, label='出口 (4个)')

    ax.set_xlim(-cp_half - 3, cp_half + 3)
    ax.set_ylim(-cp_half - 3, cp_half + 3)
    ax.set_aspect('equal')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title('仿荷叶歧管环形微通道冷板 - 结构俯视图', fontsize=13)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.15)
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

    cp_half = geo.coldplate_length / 2

    # 基板
    ax.fill_between([-cp_half, cp_half], y_base, y_ch_bottom, color='peru', alpha=0.7, label='铜基底')
    # 微通道层
    ax.fill_between([-cp_half, cp_half], y_ch_bottom, y_ch_top, color='lightblue', alpha=0.5, label='微通道层')
    # 歧管层
    ax.fill_between([-cp_half, cp_half], y_ch_top, y_manifold_top, color='lightgreen', alpha=0.5, label='歧管流道层')

    # 画微通道 (锯齿状)
    x = -cp_half
    while x < cp_half:
        ax.fill_between([x, x + geo.channel_width], y_ch_bottom, y_ch_top,
                        color='white', edgecolor='steelblue', linewidth=0.5)
        # 翅片
        ax.fill_between([x + geo.channel_width, x + geo.channel_width + geo.fin_width],
                        y_ch_bottom, y_ch_top, color='peru', alpha=0.7)
        x += geo.channel_width + geo.fin_width

    # 标注
    ax.annotate('', xy=(cp_half + 1, y_base), xytext=(cp_half + 1, y_ch_bottom),
                arrowprops=dict(arrowstyle='<->', color='black'))
    ax.text(cp_half + 2, (y_base + y_ch_bottom) / 2, f'{geo.base_thickness}mm', fontsize=8, va='center')

    ax.annotate('', xy=(cp_half + 1, y_ch_bottom), xytext=(cp_half + 1, y_ch_top),
                arrowprops=dict(arrowstyle='<->', color='black'))
    ax.text(cp_half + 2, (y_ch_bottom + y_ch_top) / 2, f'{geo.channel_height}mm', fontsize=8, va='center')

    ax.annotate('', xy=(cp_half + 1, y_ch_top), xytext=(cp_half + 1, y_manifold_top),
                arrowprops=dict(arrowstyle='<->', color='black'))
    ax.text(cp_half + 2, (y_ch_top + y_manifold_top) / 2, f'{geo.manifold_height}mm', fontsize=8, va='center')

    # 热流箭头
    for x_pos in np.linspace(-cp_half * 0.7, cp_half * 0.7, 5):
        ax.annotate('', xy=(x_pos, y_ch_bottom), xytext=(x_pos, y_base - 1),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(0, y_base - geo.base_thickness - 0.5, '芯片热流密度 (q")', ha='center', fontsize=10, color='red')

    # 流动箭头 (进液歧管 → 微通道, 方向向下)
    x_arrow_label = -cp_half + 1
    ax.annotate('进液歧管', xy=(x_arrow_label, y_ch_top + geo.manifold_height / 2), fontsize=9,
                color='blue', fontweight='bold')
    ax.annotate('', xy=(x_arrow_label + 2, y_ch_bottom),
                xytext=(x_arrow_label + 2, y_manifold_top),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    ax.set_xlim(-cp_half - 3, cp_half + 7)
    ax.set_ylim(y_base - geo.base_thickness - 1.5, y_manifold_top + 1)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('Height [mm]')
    ax.set_title('截面图')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def plot_comparison_curves(analysis: ComparativeAnalysis,
                           sp_flow_gs: float = None,
                           tp_flow_gs: float = None,
                           flow_rate_gs: float = 6.0,
                           save_path: str = None) -> plt.Figure:
    """绘制单相 vs 两相对比曲线"""
    qf_range = np.linspace(20, 250, 30)
    results = analysis.sweep_comparison(qf_range, sp_flow_gs=sp_flow_gs, tp_flow_gs=tp_flow_gs, flow_rate_gs=flow_rate_gs)

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

    tp_fluid_name = analysis.tp_sim.fluid.fluid_name
    tp_label = f"TP-{tp_fluid_name}"

    # 1. 换热系数
    ax = axes[0, 0]
    ax.plot(qf_range, sp_h, 'b-o', markersize=3, label='SP-Water')
    ax.plot(qf_range, tp_h, 'r-s', markersize=3, label=tp_label)
    ax.set_xlabel('Heat flux [W/cm2]')
    ax.set_ylabel('h [W/(cm2.K)]')
    ax.set_title('Heat Transfer Coefficient')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 压降
    ax = axes[0, 1]
    ax.plot(qf_range, sp_dP, 'b-o', markersize=3, label='SP-Water')
    ax.plot(qf_range, tp_dP, 'r-s', markersize=3, label=tp_label)
    ax.set_xlabel('Heat flux [W/cm2]')
    ax.set_ylabel('DP [kPa]')
    ax.set_title('Pressure Drop')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 热阻
    ax = axes[0, 2]
    ax.plot(qf_range, sp_Rth, 'b-o', markersize=3, label='SP-Water')
    ax.plot(qf_range, tp_Rth, 'r-s', markersize=3, label=tp_label)
    ax.set_xlabel('Heat flux [W/cm2]')
    ax.set_ylabel('Rth [(cm2.K)/W]')
    ax.set_title('Thermal Resistance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. COP
    ax = axes[1, 0]
    ax.semilogy(qf_range, sp_COP, 'b-o', markersize=3, label='SP-Water')
    ax.semilogy(qf_range, tp_COP, 'r-s', markersize=3, label=tp_label)
    ax.set_xlabel('Heat flux [W/cm2]')
    ax.set_ylabel('COP')
    ax.set_title('Coefficient of Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. 壁面温度
    ax = axes[1, 1]
    ax.plot(qf_range, sp_Tw, 'b-o', markersize=3, label='SP-Water')
    ax.plot(qf_range, tp_Tw, 'r-s', markersize=3, label=tp_label)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='T_max limit')
    ax.set_xlabel('Heat flux [W/cm2]')
    ax.set_ylabel('T_wall [C]')
    ax.set_title('Wall Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. 换热增强比
    ax = axes[1, 2]
    ax.plot(qf_range, h_ratio, 'g-^', markersize=3)
    ax.set_xlabel('Heat flux [W/cm2]')
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
    """绘制沸腾曲线 (q vs DT_sat)"""
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
        fp = res.flow_pattern
        color_map = {
            FlowPattern.BUBBLY: 0,
            FlowPattern.SLUG: 1,
            FlowPattern.ANNULAR: 2,
            FlowPattern.CHURN: 3,
            FlowPattern.MIST: 4,
        }
        pattern_colors.append(color_map.get(fp, 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    scatter = ax1.scatter(delta_T_sat, qf_range, c=pattern_colors, cmap='viridis',
                          s=20, edgecolors='none')
    ax1.set_xlabel('DT_sat = T_wall - T_sat [C]')
    ax1.set_ylabel('Heat flux q [W/cm2]')
    ax1.set_title('Boiling Curve')
    ax1.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax1, ticks=[0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels(['Bubbly', 'Slug', 'Annular', 'Churn', 'Mist'])

    ax2.plot(qf_range, h_list, 'r-', linewidth=2)
    ax2.set_xlabel('Heat flux q [W/cm2]')
    ax2.set_ylabel('h [W/(cm2.K)]')
    ax2.set_title('Two-Phase HTC vs Heat Flux')
    ax2.grid(True, alpha=0.3)

    chf_val = tp_sim.simulate(100, mass_flow_gs).CHF
    ax2.axvline(x=chf_val, color='red', linestyle='--', alpha=0.7, label=f'CHF={chf_val:.0f} W/cm2')
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

        COP_vs_qf = []
        for qf in qf_range:
            res = sim.simulate(qf, 5.0)
            COP_vs_qf.append(res.COP)
        axes[0].plot(qf_range, COP_vs_qf, 'b-', linewidth=2)
        axes[0].set_xlabel('Heat flux [W/cm2]')
        axes[0].set_ylabel('COP')

        COP_vs_mf = []
        for mf in mf_range:
            res = sim.simulate(200, mf)
            COP_vs_mf.append(res.COP)
        axes[1].plot(mf_range, COP_vs_mf, 'g-', linewidth=2)
        axes[1].set_xlabel('Mass flow rate [g/s]')
        axes[1].set_ylabel('COP')

        Rth_vs_mf = []
        for mf in mf_range:
            res = sim.simulate(200, mf)
            Rth_vs_mf.append(res.thermal_resistance)
        axes[2].plot(mf_range, Rth_vs_mf, 'r-', linewidth=2)
        axes[2].set_xlabel('Mass flow rate [g/s]')
        axes[2].set_ylabel('Rth [(cm2.K)/W]')

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
        axes[0].set_xlabel('Heat flux [W/cm2]')
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
        axes[2].set_ylabel('Rth [(cm2.K)/W]')

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