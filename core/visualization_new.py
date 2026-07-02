"""
仿荷叶歧管微通道冷板 - 可视化模块 (修正版)
=============================================
正确展示流体流动路径：中心入口 → 进液歧管(向下) → 环形微通道 → 出液歧管(向上) → 4个出口
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle, Rectangle, FancyArrow
import matplotlib.patches as mpatches
from typing import List, Optional, Dict

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.geometry import ManifoldRingChannelGeometry
from core.single_phase import SinglePhaseSimulation, SinglePhaseResult
from core.two_phase import TwoPhaseSimulation, TwoPhaseResult, FlowPattern
from core.comparison import ComparativeAnalysis, ComparisonPoint
from core.fluid_properties import FluidProperties

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def plot_geometry_topview(geo: ManifoldRingChannelGeometry,
                          save_path: str = None) -> plt.Figure:
    """绘制冷板俯视图 - 仿荷叶歧管环形微通道
    
    流动路径说明:
    1. 流体从中心入口(IN)流入歧管层
    2. 通过窄进液歧管(蓝色)向下进入微通道层
    3. 在环形微通道中沿弧形流动换热
    4. 加热后流体从宽出液歧管(红色)向上流出
    5. 汇集到4个出口(OUT)流出冷板
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    r_chip = geo.chip_length / 2  # 热源半径 [mm]
    cp_half = geo.coldplate_length / 2
    r_max = max(geo.ring_radii) + geo.ring_spacing if geo.ring_radii else r_chip
    
    # === 冷板外框 ===
    rect = Rectangle((-cp_half, -cp_half), geo.coldplate_length, geo.coldplate_width,
                     linewidth=2, edgecolor='black', facecolor='#fafafa', zorder=0)
    ax.add_patch(rect)
    
    # === 同心圆环形微通道 ===
    for r in geo.ring_radii:
        circle = Circle((0, 0), r, linewidth=0.6, edgecolor='#555555',
                        facecolor='none', linestyle='-', zorder=2)
        ax.add_patch(circle)
    
    # === 扇区角度计算 ===
    dtheta = 2 * np.pi / geo.n_sectors
    r_ref = max(r_chip, 1.0)
    theta_in = geo.inlet_slot_width / r_ref
    theta_out = geo.outlet_slot_width / r_ref
    w_wall = 1.0  # 壁面宽度 mm
    theta_w = w_wall / r_ref
    
    # 剩余角度分配
    rem_angle = 2 * dtheta - 2 * theta_w
    theta_in_m = rem_angle * (theta_in / (theta_in + theta_out))
    theta_out_m = rem_angle * (theta_out / (theta_in + theta_out))
    
    # === 进液/出液歧管交替排布 ===
    for i in range(geo.n_sectors):
        a_center = i * dtheta
        
        if i % 2 == 0:
            # 窄进液歧管 (蓝色) - 流体向下进入微通道
            half_w = theta_in_m / 2
            fc, ec = '#a1c9f4', '#1f77b4'
        else:
            # 宽出液歧管 (红色) - 流体向上从微通道流出
            half_w = theta_out_m / 2
            fc, ec = '#ff9896', '#d62728'
        
        a_start = a_center - half_w
        a_end = a_center + half_w
        
        # 绘制歧管扇区
        wedge = Wedge((0, 0), r_max + 1,
                      np.degrees(a_start), np.degrees(a_end),
                      facecolor=fc, edgecolor=ec, linewidth=1.5, alpha=0.5, zorder=3)
        ax.add_patch(wedge)
        
        # 流动方向箭头
        r_arrow = r_max * 0.75
        if i % 2 == 0:
            # 进液歧管: 从中心向外（表示流体在歧管层向微通道流动）
            ax.annotate('', xy=(r_arrow * np.cos(a_center), r_arrow * np.sin(a_center)),
                        xytext=(r_arrow * 0.4 * np.cos(a_center), r_arrow * 0.4 * np.sin(a_center)),
                        arrowprops=dict(arrowstyle='->', color=ec, lw=2.0), zorder=5)
        else:
            # 出液歧管: 从外向中心（表示流体从微通道向上汇集）
            ax.annotate('', xy=(r_arrow * 0.4 * np.cos(a_center), r_arrow * 0.4 * np.sin(a_center)),
                        xytext=(r_arrow * np.cos(a_center), r_arrow * np.sin(a_center)),
                        arrowprops=dict(arrowstyle='->', color=ec, lw=2.0), zorder=5)
        
        # 绘制壁面
        a_wall_start = a_end
        next_half_w = theta_in_m / 2 if (i + 1) % 2 == 0 else theta_out_m / 2
        a_wall_end = (i + 1) * dtheta - next_half_w
        
        wedge_w = Wedge((0, 0), r_max + 1,
                        np.degrees(a_wall_start), np.degrees(a_wall_end),
                        facecolor='#dddddd', edgecolor='#999999',
                        linewidth=1.0, alpha=1.0, zorder=4)
        ax.add_patch(wedge_w)
    
    # === 中心入口 ===
    inlet = Circle((0, 0), geo.inlet_diameter / 2,
                   edgecolor='darkblue', facecolor='#a1c9f4', alpha=0.8, zorder=5)
    ax.add_patch(inlet)
    ax.annotate('IN', xy=(0, 0), ha='center', va='center', fontsize=11,
                color='darkblue', fontweight='bold', zorder=6)
    ax.annotate('', xy=(0, geo.inlet_diameter / 2 + 1), xytext=(0, geo.inlet_diameter / 2 + 3),
                arrowprops=dict(arrowstyle='->', color='darkblue', lw=2.5), zorder=6)
    
    # === 热源区域 ===
    chip = Circle((0, 0), r_chip, linewidth=2.5, edgecolor='red',
                  facecolor='none', linestyle='--', zorder=5)
    ax.add_patch(chip)
    
    # === 4个出口 ===
    for angle_deg in [45, 135, 225, 315]:
        angle = np.radians(angle_deg)
        x = cp_half * np.cos(angle) * 0.88
        y = cp_half * np.sin(angle) * 0.88
        outlet = Circle((x, y), 2.5, edgecolor='#8c564b', facecolor='#ff9896', zorder=5)
        ax.add_patch(outlet)
        ax.annotate('OUT', xy=(x, y), ha='center', va='center', fontsize=8,
                    color='#8c564b', fontweight='bold', zorder=6)
    
    # === 图例 ===
    ax.plot([], [], color='#1f77b4', linewidth=3, label=f'窄进液歧管 ({geo.n_sectors // 2}条)')
    ax.plot([], [], color='#d62728', linewidth=3, label=f'宽出液歧管 ({geo.n_sectors // 2}条)')
    ax.plot([], [], color='#999999', linewidth=2, label='歧管分流壁面')
    ax.plot([], [], color='#555555', linewidth=1, label=f'环形微通道 ({geo.n_rings}圈)')
    ax.plot([], [], color='red', linestyle='--', linewidth=2, label=f'热源区域 Ø{geo.chip_length}mm')
    ax.scatter([], [], color='#ff9896', edgecolors='#8c564b', s=80, label='出口 (4个)')
    
    ax.set_xlim(-cp_half - 3, cp_half + 3)
    ax.set_ylim(-cp_half - 3, cp_half + 3)
    ax.set_aspect('equal')
    ax.set_xlabel('x [mm]', fontsize=11)
    ax.set_ylabel('y [mm]', fontsize=11)
    ax.set_title('仿荷叶歧管环形微通道冷板 - 俯视图\n'
                 '流动路径: 中心入口 → 进液歧管(蓝) → 微通道 → 出液歧管(红) → 4出口',
                 fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


def plot_geometry_crosssection(geo: ManifoldRingChannelGeometry,
                                save_path: str = None) -> plt.Figure:
    """绘制冷板截面图 - 展示三层结构和流动
    
    展示:
    1. 底层：基板(铜) + 芯片热流
    2. 中层：微通道(白色通道 + 铜翅片)
    3. 顶层：歧管层(进液歧管蓝 + 出液歧管红 + 壁面灰)
    4. 流动箭头：进液向下(蓝) → 微通道流动 → 出液向上(红)
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    
    # 层高度
    y_base = 0
    y_ch_bottom = y_base + geo.base_thickness
    y_ch_top = y_ch_bottom + geo.channel_height
    y_manifold_top = y_ch_top + geo.manifold_height
    
    # 截面展示宽度 (模拟一个歧管段的宽度)
    w_inlet = geo.inlet_slot_width  # 进液歧管宽度
    w_outlet = geo.outlet_slot_width  # 出液歧管宽度
    w_wall = 1.0  # 壁面宽度
    pitch = w_inlet + w_wall + w_outlet + w_wall  # 一个周期
    
    x_start = 0
    x_end = 3 * pitch  # 展示3个周期
    
    # === 基板层 ===
    ax.fill_between([x_start, x_end], y_base, y_ch_bottom,
                    color='peru', alpha=0.8, label='铜基板')
    
    # === 微通道层 ===
    ax.fill_between([x_start, x_end], y_ch_bottom, y_ch_top,
                    color='lightblue', alpha=0.4, label='微通道层')
    
    # 绘制微通道结构 (锯齿状)
    x = x_start
    while x < x_end:
        # 通道 (白色)
        ax.fill_between([x, min(x + geo.channel_width, x_end)],
                        y_ch_bottom, y_ch_top,
                        color='white', edgecolor='steelblue', linewidth=0.5)
        # 翅片 (铜色)
        ax.fill_between([x + geo.channel_width,
                         min(x + geo.channel_width + geo.fin_width, x_end)],
                        y_ch_bottom, y_ch_top,
                        color='peru', alpha=0.7)
        x += geo.channel_width + geo.fin_width
    
    # === 歧管层结构 ===
    # 绘制3个周期: [进液歧管 | 壁面 | 出液歧管 | 壁面]
    x_positions = [0, pitch, 2 * pitch]
    
    for i, x_pos in enumerate(x_positions):
        # 进液歧管 (蓝色)
        ax.fill_between([x_pos, x_pos + w_inlet],
                        y_ch_top, y_manifold_top,
                        color='#a1c9f4', edgecolor='#1f77b4',
                        linewidth=1.5, alpha=0.6)
        ax.text(x_pos + w_inlet/2, y_ch_top + geo.manifold_height/2,
                '进液歧管', ha='center', va='center', fontsize=9,
                color='#1f77b4', fontweight='bold')
        # 流动箭头: 向下
        ax.annotate('', xy=(x_pos + w_inlet/2, y_ch_bottom + 0.2),
                    xytext=(x_pos + w_inlet/2, y_manifold_top - 0.2),
                    arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=2))
        
        # 壁面 (灰色)
        ax.fill_between([x_pos + w_inlet, x_pos + w_inlet + w_wall],
                        y_ch_top, y_manifold_top,
                        color='#dddddd', edgecolor='#999999', linewidth=1)
        
        # 出液歧管 (红色)
        ax.fill_between([x_pos + w_inlet + w_wall, x_pos + w_inlet + w_wall + w_outlet],
                        y_ch_top, y_manifold_top,
                        color='#ff9896', edgecolor='#d62728',
                        linewidth=1.5, alpha=0.6)
        ax.text(x_pos + w_inlet + w_wall + w_outlet/2, y_ch_top + geo.manifold_height/2,
                '出液歧管', ha='center', va='center', fontsize=9,
                color='#d62728', fontweight='bold')
        # 流动箭头: 向上
        ax.annotate('', xy=(x_pos + w_inlet + w_wall + w_outlet/2, y_manifold_top - 0.2),
                    xytext=(x_pos + w_inlet + w_wall + w_outlet/2, y_ch_bottom + 0.2),
                    arrowprops=dict(arrowstyle='->', color='#d62728', lw=2))
        
        # 壁面
        ax.fill_between([x_pos + w_inlet + w_wall + w_outlet,
                         x_pos + pitch],
                        y_ch_top, y_manifold_top,
                        color='#dddddd', edgecolor='#999999', linewidth=1)
        
        # 微通道内流动箭头 (水平)
        ax.annotate('', xy=(x_pos + w_inlet + w_wall + w_outlet/2, y_ch_bottom + geo.channel_height/2),
                    xytext=(x_pos + w_inlet/2, y_ch_bottom + geo.channel_height/2),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))
        ax.text(x_pos + pitch/2, y_ch_bottom + geo.channel_height/2 + 0.3,
                '微通道流动', ha='center', va='bottom', fontsize=8, color='#555555')
    
    # === 尺寸标注 ===
    x_label = x_end + 1
    
    ax.annotate('', xy=(x_label, y_base), xytext=(x_label, y_ch_bottom),
                arrowprops=dict(arrowstyle='<->', color='black'))
    ax.text(x_label + 1.5, (y_base + y_ch_bottom)/2,
            f'{geo.base_thickness}mm', fontsize=9, va='center')
    
    ax.annotate('', xy=(x_label, y_ch_bottom), xytext=(x_label, y_ch_top),
                arrowprops=dict(arrowstyle='<->', color='black'))
    ax.text(x_label + 1.5, (y_ch_bottom + y_ch_top)/2,
            f'{geo.channel_height}mm', fontsize=9, va='center')
    
    ax.annotate('', xy=(x_label, y_ch_top), xytext=(x_label, y_manifold_top),
                arrowprops=dict(arrowstyle='<->', color='black'))
    ax.text(x_label + 1.5, (y_ch_top + y_manifold_top)/2,
            f'{geo.manifold_height}mm', fontsize=9, va='center')
    
    # === 热流箭头 ===
    for x_pos in np.linspace(x_start + 2, x_end - 2, 5):
        ax.annotate('', xy=(x_pos, y_ch_bottom), xytext=(x_pos, y_base - 0.8),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
    ax.text(x_end/2, y_base - geo.base_thickness - 0.5,
            '芯片热流密度 (q") →', ha='center', fontsize=11, color='red', fontweight='bold')
    
    # === 图例与设置 ===
    ax.set_xlim(x_start - 1, x_label + 5)
    ax.set_ylim(y_base - geo.base_thickness - 1.5, y_manifold_top + 0.5)
    ax.set_xlabel('径向位置 [mm]', fontsize=11)
    ax.set_ylabel('高度 [mm]', fontsize=11)
    ax.set_title('仿荷叶歧管微通道冷板 - 截面图\n'
                 '流动: 进液歧管(向下蓝) → 微通道流动 → 出液歧管(向上红)',
                 fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig


# === 保留原有的其他绘图函数 ===
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
    """绘制沸腾曲线"""
    qf_range = np.linspace(5, 250, 50)
    delta_T_sat = []
    h_list = []
    pattern_colors = []

    for qf in qf_range:
        res = tp_sim.simulate(qf, mass_flow_gs)
        dT = res.T_wall_avg - res.T_sat
        delta_T_sat.append(max(dT, 0.1))
        h_list.append(res.h_conv_cm2)
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

    scatter = ax1.scatter(delta_T_sat, qf_range, c=pattern_colors, cmap='viridis', s=20)
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

        COP_vs_qf = [sim.simulate(qf, 5.0).COP for qf in qf_range]
        axes[0].plot(qf_range, COP_vs_qf, 'b-', linewidth=2)
        axes[0].set_xlabel('Heat flux [W/cm2]')
        axes[0].set_ylabel('COP')

        COP_vs_mf = [sim.simulate(200, mf).COP for mf in mf_range]
        axes[1].plot(mf_range, COP_vs_mf, 'g-', linewidth=2)
        axes[1].set_xlabel('Mass flow rate [g/s]')
        axes[1].set_ylabel('COP')

        Rth_vs_mf = [sim.simulate(200, mf).thermal_resistance for mf in mf_range]
        axes[2].plot(mf_range, Rth_vs_mf, 'r-', linewidth=2)
        axes[2].set_xlabel('Mass flow rate [g/s]')
        axes[2].set_ylabel('Rth [(cm2.K)/W]')
    else:
        sim = TwoPhaseSimulation(geo, FluidProperties("HFE7100"))
        title_prefix = "Two-Phase HFE-7100"
        qf_range = np.linspace(20, 250, 25)
        mf_range = np.linspace(3, 12, 20)

        COP_vs_qf = [sim.simulate(qf, 6.0).COP for qf in qf_range]
        axes[0].plot(qf_range, COP_vs_qf, 'b-', linewidth=2)
        axes[0].set_xlabel('Heat flux [W/cm2]')
        axes[0].set_ylabel('COP')
        axes[0].axvline(x=267, color='red', linestyle='--', alpha=0.5, label='CHF')
        axes[0].legend()

        COP_vs_mf = [sim.simulate(100, mf).COP for mf in mf_range]
        axes[1].plot(mf_range, COP_vs_mf, 'g-', linewidth=2)
        axes[1].set_xlabel('Mass flow rate [g/s]')
        axes[1].set_ylabel('COP')

        Rth_vs_mf = [sim.simulate(100, mf).thermal_resistance for mf in mf_range]
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