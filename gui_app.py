"""
仿荷叶歧管微通道冷板 - Streamlit GUI
=============================================
单相水冷 vs 两相沸腾冷却 对比分析工具

运行: streamlit run gui_app.py
"""

import sys
import os
import numpy as np
import io
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import core.geometry
import core.fluid_properties
import core.single_phase
import core.two_phase
import core.comparison
import core.visualization

import streamlit as st
from core.geometry import ManifoldRingChannelGeometry, ChannelShape
from core.fluid_properties import FluidProperties
from core.single_phase import SinglePhaseSimulation
from core.two_phase import TwoPhaseSimulation, FlowPattern
from core.comparison import ComparativeAnalysis
from core.visualization import (
    plot_geometry_topview,
    plot_geometry_crosssection,
    plot_comparison_curves,
    plot_boiling_curve,
    plot_sensitivity_analysis,
    plot_geometry_3d_explosion,
)
from core.validation import ModelValidator
from core.sensitivity import SensitivityAnalyzer

# ===== 页面配置 =====
st.set_page_config(
    page_title="仿荷叶歧管微通道冷板",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🍃 仿荷叶歧管微通道冷板")
st.subheader("单相水冷 vs 两相沸腾冷却 对比分析")
st.caption("基于浙大吴赞课题组: Xin Z, et al. Energy (2025) & Energy Conversion and Management (2026)")

# ===== 侧边栏参数 =====
with st.sidebar:
    st.header("🔧 参数设置")

    with st.form("param_form"):
        # --- 冷板几何 ---
        st.subheader("冷板几何")
        coldplate_diameter = st.number_input("冷板直径 [mm]", value=30.0, min_value=0.0, max_value=500.0)

        input_mode = st.radio("热源输入方式", ["按直径", "按面积"], horizontal=True)
        if input_mode == "按直径":
            chip_length = st.number_input("圆形热源直径 [mm]", value=20.0, min_value=0.0, max_value=100.0)
            chip_area = np.pi * (chip_length / 2) ** 2
            st.caption(f"→ 自动计算面积: {chip_area:.2f} mm²")
        else:
            chip_area = st.number_input("圆形热源面积 [mm²]", value=314.16, min_value=0.5, max_value=10000.0)
            chip_length = 2 * np.sqrt(chip_area / np.pi)
            st.caption(f"→ 自动计算直径: {chip_length:.2f} mm")

        channel_width = st.number_input("微通道宽度 [mm]", value=0.30, min_value=0.0, max_value=1.0, step=0.01, format="%.3f")
        channel_height = st.number_input("微通道深度 [mm]", value=1.4, min_value=0.0, max_value=10.0, step=0.1)
        fin_width = st.number_input("翅片厚度 [mm]", value=0.30, min_value=0.0, max_value=5.0, step=0.01, format="%.3f")
        manifold_height = st.number_input("歧管层高度 [mm]", value=1.8, min_value=0.0, max_value=5.0, step=0.1)
        n_rings = st.number_input("微通道与翅片周期数", value=13, min_value=2, max_value=100)
        n_sectors = st.number_input("歧管分流流道数", value=16, min_value=4, max_value=50, step=2)
        inlet_diameter = st.number_input("中心入口直径 [mm]", value=4.5, min_value=0.5, max_value=200.0)

        # --- 高级几何参数 (折叠) ---
        with st.expander("高级几何参数"):
            coldplate_length = st.number_input("冷板长度 [mm]", value=30.0, min_value=0.5)
            coldplate_width = st.number_input("冷板宽度 [mm]", value=30.0, min_value=0.5)
            total_thickness = st.number_input("冷板总厚度 [mm]", value=4.2, min_value=0.5)
            base_thickness = st.number_input("基板厚度 [mm]", value=0.5, min_value=0.5)
            ring_spacing = st.number_input("环形通道间距 [mm]", value=0.6, min_value=0.05, step=0.01, format="%.3f")
            inlet_slot_width = st.number_input("窄进液歧管缝宽 [mm]", value=0.5, min_value=0.05, step=0.01, format="%.3f")
            outlet_slot_width = st.number_input("宽出液歧管缝宽 [mm]", value=1.5, min_value=0.05, step=0.01, format="%.3f")
            channel_shape_name = st.selectbox(
                "微通道截面形状",
                ["RECTANGULAR", "CIRCULAR", "TRAPEZOIDAL", "TRIANGULAR"],
                index=0,
            )
            channel_diameter = st.number_input("圆形截面直径 [mm]", value=0.3, min_value=0.0, step=0.01, format="%.3f")
            trapezoid_side_angle = st.number_input("梯形侧壁角 [°]", value=54.7, min_value=0.0)

        # --- 材料参数 (折叠) ---
        with st.expander("材料参数"):
            substrate_material = st.text_input("基板材料", value="copper")
            sintering_material = st.text_input("烧结材料", value="silver")

        # --- 运行工况与工质 (合并为一个区域) ---
        st.subheader("运行工况与工质")
        heat_flux = st.number_input("热流密度 [W/cm²]", value=633.0, min_value=1.0, max_value=700.0)
        sp_flow = st.number_input("单相质量流量 [g/s]", value=39.0, min_value=0.5, max_value=200.0)
        tp_flow = st.number_input("两相质量流量 [g/s]", value=6.0, min_value=0.5, max_value=200.0)
        T_inlet = st.number_input("进口温度 [°C]", value=20.0, min_value=0.1, max_value=80.0)
        tp_fluid = st.selectbox("工质类型", ["HFE7100", "R245fa", "R1233zdE", "water"], index=0)

        submitted = st.form_submit_button("应用参数")

    if submitted:
        st.session_state["calculated"] = False

    st.info("文献来源:\n- Xin Z, et al. Energy, 2025\n- Xin Z, et al. ECM, 2026\n- 浙大吴赞课题组")

# ===== 主界面计算按钮 =====
if st.button("🔄 开始计算", type="primary", use_container_width=True):
    st.session_state["calculated"] = True

if not st.session_state.get("calculated", False):
    st.info("请在左侧设置参数后，点击「开始计算」按钮")
    st.stop()

# ===== 创建几何对象 =====
geo = ManifoldRingChannelGeometry(
    chip_area=chip_area,
    chip_length=chip_length,
    coldplate_length=coldplate_length,
    coldplate_width=coldplate_width,
    total_thickness=total_thickness,
    manifold_height=manifold_height,
    inlet_diameter=inlet_diameter,
    n_rings=n_rings,
    n_sectors=n_sectors,
    channel_shape=ChannelShape[channel_shape_name],
    channel_width=channel_width,
    channel_height=channel_height,
    channel_diameter=channel_diameter,
    trapezoid_side_angle=trapezoid_side_angle,
    fin_width=fin_width,
    base_thickness=base_thickness,
    ring_spacing=ring_spacing,
    ring_channel_width=channel_width,
    inlet_slot_width=inlet_slot_width,
    outlet_slot_width=outlet_slot_width,
    substrate_material=substrate_material,
    sintering_material=sintering_material,
)

# ===== 导出辅助函数 =====
def _fig_to_png(fig):
    """将matplotlib图表转为PNG字节数据"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()


def _make_csv(headers, rows):
    """生成CSV字节数据（UTF-8 BOM，兼容Excel中文）"""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    writer.writerows(rows)
    return buf.getvalue().encode('utf-8-sig')


# ===== 预计算仿真结果 =====
sp_sim_main = SinglePhaseSimulation(geo, FluidProperties("water"))
sp_res_main = sp_sim_main.simulate(heat_flux, sp_flow, T_inlet)

tp_sim_main = TwoPhaseSimulation(geo, FluidProperties(tp_fluid))
tp_res_main = tp_sim_main.simulate(heat_flux, tp_flow, T_inlet)

analysis_main = ComparativeAnalysis(geo, tp_fluid=tp_fluid)
cp_main = analysis_main.compare_at_condition(heat_flux, sp_flow, tp_flow, T_inlet)


# ===== 一键导出汇报 =====
if st.button("📤 一键导出完整汇报", type="secondary"):
    import zipfile
    import time

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 几何图
        zf.writestr("01_俯视图.png", _fig_to_png(plot_geometry_topview(geo)))
        zf.writestr("02_截面图.png", _fig_to_png(plot_geometry_crosssection(geo)))
        zf.writestr("03_3D爆炸视图.png", _fig_to_png(plot_geometry_3d_explosion(geo)))

        # 性能曲线
        zf.writestr("04_对比曲线.png", _fig_to_png(plot_comparison_curves(analysis_main, sp_flow, tp_flow)))
        zf.writestr("05_沸腾曲线.png", _fig_to_png(plot_boiling_curve(tp_sim_main, tp_flow)))

        # 雷达图
        fig_r, ax_r = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        cat = ['COP', 'h', '均匀性', '压降(反比)', '热阻(反比)']
        angles_r = np.linspace(0, 2*np.pi, len(cat), endpoint=False).tolist()
        angles_r += angles_r[:1]
        sp_v = [sp_res_main.COP/max(sp_res_main.COP, tp_res_main.COP, 1),
                sp_res_main.h_conv_cm2/max(sp_res_main.h_conv_cm2, tp_res_main.h_conv_cm2, 0.01),
                1, 1, 1]
        tp_v = [tp_res_main.COP/max(sp_res_main.COP, tp_res_main.COP, 1),
                tp_res_main.h_conv_cm2/max(sp_res_main.h_conv_cm2, tp_res_main.h_conv_cm2, 0.01),
                1, 1, 1]
        sp_v += sp_v[:1]; tp_v += tp_v[:1]
        ax_r.plot(angles_r, sp_v, 'b-o', label='单相')
        ax_r.fill(angles_r, sp_v, alpha=0.15, color='blue')
        ax_r.plot(angles_r, tp_v, 'r-s', label='两相')
        ax_r.fill(angles_r, tp_v, alpha=0.15, color='red')
        ax_r.set_xticks(angles_r[:-1])
        ax_r.set_xticklabels(cat)
        ax_r.legend()
        zf.writestr("06_雷达图.png", _fig_to_png(fig_r))

        # 数据表
        geo_data = geo.get_params_dict()
        zf.writestr("07_几何参数.csv", _make_csv(geo_data.keys(), [geo_data.values()]))

        sp_data = {"热流密度": heat_flux, "流量": sp_flow, "壁温": sp_res_main.T_wall_max,
                   "换热系数": sp_res_main.h_conv_cm2, "压降": sp_res_main.pressure_drop,
                   "热阻": sp_res_main.thermal_resistance, "COP": sp_res_main.COP}
        zf.writestr("08_单相结果.csv", _make_csv(sp_data.keys(), [sp_data.values()]))

        tp_data = {"热流密度": heat_flux, "流量": tp_flow, "壁温": tp_res_main.T_wall_max,
                    "换热系数": tp_res_main.h_conv_cm2, "压降": tp_res_main.pressure_drop,
                    "热阻": tp_res_main.thermal_resistance, "COP": tp_res_main.COP,
                    "CHF": tp_res_main.CHF, "流型": tp_res_main.flow_pattern_name}
        zf.writestr("09_两相结果.csv", _make_csv(tp_data.keys(), [tp_data.values()]))

        # 汇报文档
        report_text = f"""仿荷叶歧管微通道冷板仿真汇报
=====================================
生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

一、工况参数
  热流密度: {heat_flux:.1f} W/cm²
  单相流量: {sp_flow:.1f} g/s (水)
  两相流量: {tp_flow:.1f} g/s ({tp_fluid})
  进口温度: {T_inlet:.1f} °C

二、单相水冷结果
  最高壁温: {sp_res_main.T_wall_max:.1f} °C
  换热系数: {sp_res_main.h_conv_cm2:.2f} W/(cm²·K)
  压降: {sp_res_main.pressure_drop/1000:.1f} kPa
  热阻: {sp_res_main.thermal_resistance:.4f} (cm²·K)/W
  COP: {sp_res_main.COP:.0f}

三、两相沸腾结果
  最高壁温: {tp_res_main.T_wall_max:.1f} °C
  换热系数: {tp_res_main.h_conv_cm2:.2f} W/(cm²·K)
  压降: {tp_res_main.pressure_drop/1000:.1f} kPa
  热阻: {tp_res_main.thermal_resistance:.4f} (cm²·K)/W
  COP: {tp_res_main.COP:.0f}
  CHF: {tp_res_main.CHF:.1f} W/cm²
  流型: {tp_res_main.flow_pattern_name}

四、对比结论
  换热系数提升: {(cp_main.h_ratio-1)*100:.1f}%
  热阻降低: {(1-tp_res_main.thermal_resistance/sp_res_main.thermal_resistance)*100:.1f}%
  推荐: {'两相方案更优' if cp_main.h_ratio > 1 else '单相方案更优'}

五、文献参考
  Xin Z, et al. Energy, 2025
  Xin Z, et al. Energy Conversion and Management, 2026
"""
        zf.writestr("00_汇报摘要.txt", report_text.encode('utf-8'))

    zip_buf.seek(0)
    st.download_button("📥 下载完整汇报ZIP", data=zip_buf.getvalue(),
                      file_name=f"coldplate_report_{time.strftime('%Y%m%d_%H%M%S')}.zip",
                      mime="application/zip", key="dl_full_report")
    st.success("汇报包已生成！包含几何图、性能曲线、数据表和汇报摘要。")


# ===== 主界面标签页 =====
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 摘要仪表盘", "📐 几何结构", "💧 单相水冷", "🔥 两相沸腾", "📊 对比分析", "📈 敏感性分析", "🔍 模型验证"
])


# ===== Tab0: 摘要仪表盘 =====
with tab0:
    st.subheader("🔑 关键性能指标")

    # KPI卡片 - 4列
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)

    with col_k1:
        st.metric("COP (单相)", f"{sp_res_main.COP:.0f}",
                  delta=f"{cp_main.sp_COP - cp_main.tp_COP:.0f}" if cp_main.sp_COP > cp_main.tp_COP else f"{cp_main.sp_COP - cp_main.tp_COP:.0f}",
                  delta_color="inverse")
    with col_k2:
        st.metric("COP (两相)", f"{tp_res_main.COP:.0f}",
                  delta=f"{(cp_main.tp_COP/cp_main.sp_COP - 1)*100:.1f}%")
    with col_k3:
        st.metric("热阻 (单相)", f"{sp_res_main.thermal_resistance:.3f}", "cm²·K/W")
    with col_k4:
        st.metric("热阻 (两相)", f"{tp_res_main.thermal_resistance:.3f}", "cm²·K/W")

    st.divider()

    # 第二行KPI
    col_k5, col_k6, col_k7, col_k8 = st.columns(4)
    with col_k5:
        st.metric("壁温 (单相)", f"{sp_res_main.T_wall_max:.1f}°C")
    with col_k6:
        st.metric("壁温 (两相)", f"{tp_res_main.T_wall_max:.1f}°C")
    with col_k7:
        st.metric("压降 (单相)", f"{sp_res_main.pressure_drop/1000:.1f} kPa")
    with col_k8:
        st.metric("压降 (两相)", f"{tp_res_main.pressure_drop/1000:.1f} kPa")

    st.divider()

    # 雷达图和结论
    col_radar, col_conclusion = st.columns([3, 2])

    with col_radar:
        st.subheader("单相 vs 两相 雷达图")
        # 用matplotlib绘制雷达图
        categories = ['COP\n(对数)', '换热系数\nh', '壁温均匀性\n(反比)', '压降\n(反比)', '热阻\n(反比)']
        # 归一化到0-1 (注意: TwoPhaseResult没有delta_T_wall, 用T_wall_max-T_wall_avg近似)
        sp_dtw = max(sp_res_main.delta_T_wall, 0.1)
        tp_dtw = max(tp_res_main.T_wall_max - tp_res_main.T_wall_avg, 0.1)
        cop_max = max(cp_main.sp_COP, cp_main.tp_COP, 1)
        h_max = max(sp_res_main.h_conv_cm2, tp_res_main.h_conv_cm2, 0.001)
        sp_vals = [
            np.log10(max(sp_res_main.COP, 1)) / np.log10(cop_max),
            sp_res_main.h_conv_cm2 / h_max,
            (1.0/sp_dtw) / ((1.0/sp_dtw) + (1.0/tp_dtw)),
            (1.0/max(sp_res_main.pressure_drop, 1)) / ((1.0/max(sp_res_main.pressure_drop, 1)) + (1.0/max(tp_res_main.pressure_drop, 1))),
            (1.0/max(sp_res_main.thermal_resistance, 0.001)) / ((1.0/max(sp_res_main.thermal_resistance, 0.001)) + (1.0/max(tp_res_main.thermal_resistance, 0.001))),
        ]
        tp_vals = [
            np.log10(max(tp_res_main.COP, 1)) / np.log10(cop_max),
            tp_res_main.h_conv_cm2 / h_max,
            (1.0/tp_dtw) / ((1.0/sp_dtw) + (1.0/tp_dtw)),
            (1.0/max(tp_res_main.pressure_drop, 1)) / ((1.0/max(sp_res_main.pressure_drop, 1)) + (1.0/max(tp_res_main.pressure_drop, 1))),
            (1.0/max(tp_res_main.thermal_resistance, 0.001)) / ((1.0/max(sp_res_main.thermal_resistance, 0.001)) + (1.0/max(tp_res_main.thermal_resistance, 0.001))),
        ]

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        sp_vals += sp_vals[:1]
        tp_vals += tp_vals[:1]

        fig_radar, ax_r = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax_r.plot(angles, sp_vals, 'b-o', linewidth=2, label='单相水冷')
        ax_r.fill(angles, sp_vals, alpha=0.15, color='blue')
        ax_r.plot(angles, tp_vals, 'r-s', linewidth=2, label='两相沸腾')
        ax_r.fill(angles, tp_vals, alpha=0.15, color='red')
        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(categories, fontsize=9)
        ax_r.set_ylim(0, 1.1)
        ax_r.set_title('性能对比雷达图 (归一化)', fontsize=12, pad=20)
        ax_r.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        st.pyplot(fig_radar)
        st.download_button("📥 导出雷达图PNG", data=_fig_to_png(fig_radar),
                          file_name="radar_chart.png", mime="image/png", key="dl_radar")

    with col_conclusion:
        st.subheader("📝 结论摘要")
        h_improve = (cp_main.h_ratio - 1) * 100
        rth_improve = (1 - cp_main.Rth_ratio) * 100 if hasattr(cp_main, 'Rth_ratio') else (1 - tp_res_main.thermal_resistance/sp_res_main.thermal_resistance) * 100
        rth_improve = (1 - tp_res_main.thermal_resistance/sp_res_main.thermal_resistance) * 100

        st.success(f"""
        **工况**: q={heat_flux:.0f} W/cm², 单相m={sp_flow:.0f}g/s, 两相m={tp_flow:.0f}g/s

        **关键发现**:
        - 两相换热系数提升 **{h_improve:.1f}%**
        - 两相热阻降低 **{rth_improve:.1f}%**
        - 两相壁温: {tp_res_main.T_wall_max:.1f}°C {"✅" if tp_res_main.T_wall_max < 85 else "⚠️"}
        - CHF裕量: {tp_res_main.CHF_margin*100:.1f}%
        - 流型: {tp_res_main.flow_pattern_name}

        **推荐**: {"两相沸腾冷却方案性能更优" if h_improve > 0 else "单相水冷方案在此工况下更优"}
        """)

    st.divider()

    # 安全工作区标注
    st.subheader("⚠️ 安全工作区")
    qf_range = np.linspace(20, min(300, tp_res_main.CHF * 0.9), 50)
    sp_tw = [sp_sim_main.simulate(q, sp_flow, T_inlet).T_wall_max for q in qf_range]
    tp_tw = [tp_sim_main.simulate(q, tp_flow, T_inlet).T_wall_max for q in qf_range]

    fig_safe, ax_safe = plt.subplots(figsize=(10, 4))
    ax_safe.plot(qf_range, sp_tw, 'b-', linewidth=2, label='单相水冷')
    ax_safe.plot(qf_range, tp_tw, 'r-', linewidth=2, label='两相沸腾')
    ax_safe.axhline(y=85, color='green', linestyle='--', alpha=0.7, label='安全限 85°C')
    ax_safe.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='危险限 100°C')
    ax_safe.axvline(x=tp_res_main.CHF, color='orange', linestyle=':', alpha=0.7, label=f'CHF={tp_res_main.CHF:.0f}')
    # 安全区域绿色背景
    ax_safe.axhspan(0, 85, alpha=0.05, color='green')
    ax_safe.axhspan(85, 100, alpha=0.05, color='yellow')
    ax_safe.axhspan(100, 200, alpha=0.05, color='red')
    ax_safe.set_xlabel('热流密度 [W/cm²]')
    ax_safe.set_ylabel('壁温 [°C]')
    ax_safe.set_title('壁温 vs 热流密度 (含安全区标注)')
    ax_safe.legend(fontsize=8)
    ax_safe.grid(True, alpha=0.3)
    ax_safe.set_ylim(20, max(max(sp_tw), max(tp_tw), 100) + 10)
    plt.tight_layout()
    st.pyplot(fig_safe)
    st.download_button("📥 导出安全工作区图PNG", data=_fig_to_png(fig_safe),
                      file_name="safety_zone.png", mime="image/png", key="dl_safety")

# ===== Tab1: 几何结构 =====
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("俯视图")
        fig_top = plot_geometry_topview(geo)
        st.pyplot(fig_top)
        st.download_button("📥 导出俯视图PNG", data=_fig_to_png(fig_top),
                           file_name="geometry_topview.png", mime="image/png", key="dl_geo_top")

    with col2:
        st.subheader("截面图")
        fig_cross = plot_geometry_crosssection(geo)
        st.pyplot(fig_cross)
        st.download_button("📥 导出截面图PNG", data=_fig_to_png(fig_cross),
                           file_name="geometry_crosssection.png", mime="image/png", key="dl_geo_cross")

    st.subheader("3D 分层爆炸视图")
    fig_3d = plot_geometry_3d_explosion(geo)
    st.pyplot(fig_3d)
    st.download_button("📥 导出3D爆炸视图PNG", data=_fig_to_png(fig_3d),
                       file_name="geometry_3d_explosion.png", mime="image/png", key="dl_geo_3d")

    st.divider()
    st.subheader("几何参数摘要")
    params = geo.get_params_dict()
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("水力直径", f"{params['hydraulic_diameter_mm']} mm")
        st.metric("总通道数 (等效)", f"{params['total_channels']}")
        st.metric("微通道层孔隙率", f"{params['porosity']}")
    with col_b:
        st.metric("加热芯片面积", f"{params['chip_area_mm2']:.2f} mm² (圆形)")
        st.metric("总换热面积 (Awet)", f"{params['heat_transfer_area_m2']:.4f} cm²")
        st.metric("同心圆环数 / 总径向流道数", f"{params['n_rings']} 环 / {params['n_sectors']} 根")
    with col_c:
        st.metric("冷板总厚 / 平均扇区弧长", f"{params['total_thickness_mm']} mm / {params['L_flow_avg_mm']:.3f} mm")
        st.metric("通道截面尺寸", f"{channel_width}×{channel_height} mm")
        st.metric("各环扇区弧长 (mm)", ", ".join([f"{l:.2f}" for l in params['L_flow_rings_mm']]))

    st.divider()
    # 几何参数导出
    geo_csv = _make_csv(
        ["参数", "值", "单位"],
        [
            ["芯片面积", f"{params['chip_area_mm2']:.2f}", "mm²"],
            ["芯片直径", f"{params['chip_length_mm']}", "mm"],
            ["冷板长×宽", f"{params['coldplate_L_W_mm'][0]}×{params['coldplate_L_W_mm'][1]}", "mm"],
            ["冷板总厚", f"{params['total_thickness_mm']}", "mm"],
            ["歧管层高度", f"{params['manifold_height_mm']}", "mm"],
            ["微通道深度", f"{params['channel_height_mm']}", "mm"],
            ["微通道宽度", f"{params['channel_width_mm']}", "mm"],
            ["翅片厚度", f"{params['fin_width_mm']}", "mm"],
            ["基板厚度", f"{params['base_thickness_mm']}", "mm"],
            ["水力直径", f"{params['hydraulic_diameter_mm']}", "mm"],
            ["总通道数", f"{params['total_channels']}", ""],
            ["孔隙率", f"{params['porosity']}", ""],
            ["换热面积", f"{params['heat_transfer_area_m2']:.4f}", "cm²"],
            ["环形歧管数", f"{params['n_rings']}", "环"],
            ["分流扇区数", f"{params['n_sectors']}", "区"],
            ["平均有效流程", f"{params['L_flow_avg_mm']:.3f}", "mm"],
        ]
    )
    st.download_button("📥 导出几何参数CSV", data=geo_csv,
                       file_name="geometry_params.csv", mime="text/csv", key="dl_geo_csv")

# ===== Tab2: 单相水冷 =====
with tab2:
    sp_sim = SinglePhaseSimulation(geo, FluidProperties("water"))
    sp_res = sp_sim.simulate(heat_flux, sp_flow, T_inlet)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("换热系数 h", f"{sp_res.h_conv_cm2:.3f} W/(cm²·K)")
        st.metric("总散热功率", f"{sp_res.Q_total:.1f} W")
    with col2:
        st.metric("最高壁温", f"{sp_res.T_wall_max:.1f} °C")
        st.metric("出口温度", f"{sp_res.T_outlet:.1f} °C")
    with col3:
        st.metric("压降 ΔP", f"{sp_res.pressure_drop/1e3:.2f} kPa")
        st.metric("泵功", f"{sp_res.pumping_power*1e3:.2f} mW")
    with col4:
        st.metric("热阻 Rth", f"{sp_res.thermal_resistance:.4f} (cm²·K)/W")
        st.metric("COP", f"{sp_res.COP:.0f}")
        st.metric("Re", f"{sp_res.Re:.0f}")

    # 单相结果数据导出
    sp_csv = _make_csv(
        ["参数", "值", "单位"],
        [
            ["热流密度", f"{sp_res.heat_flux:.1f}", "W/cm²"],
            ["质量流量", f"{sp_res.mass_flow_rate:.1f}", "g/s"],
            ["进口温度", f"{sp_res.T_inlet:.1f}", "°C"],
            ["出口温度", f"{sp_res.T_outlet:.1f}", "°C"],
            ["最高壁温", f"{sp_res.T_wall_max:.1f}", "°C"],
            ["平均壁温", f"{sp_res.T_wall_avg:.1f}", "°C"],
            ["壁面温差", f"{sp_res.delta_T_wall:.1f}", "°C"],
            ["换热系数h", f"{sp_res.h_conv_cm2:.3f}", "W/(cm²·K)"],
            ["压降ΔP", f"{sp_res.pressure_drop/1e3:.2f}", "kPa"],
            ["热阻Rth", f"{sp_res.thermal_resistance:.4f}", "(cm²·K)/W"],
            ["总散热功率", f"{sp_res.Q_total:.1f}", "W"],
            ["泵功", f"{sp_res.pumping_power*1e3:.2f}", "mW"],
            ["COP", f"{sp_res.COP:.0f}", ""],
            ["Re", f"{sp_res.Re:.0f}", ""],
            ["Nu", f"{sp_res.Nu:.1f}", ""],
            ["Dean数", f"{sp_res.Dean_avg:.1f}", ""],
            ["增强因子", f"{sp_res.enhancement_avg:.2f}", ""],
        ]
    )
    st.download_button("📥 导出单相结果CSV", data=sp_csv,
                       file_name="single_phase_results.csv", mime="text/csv", key="dl_sp_csv")

    st.divider()

    # 单相热流密度扫描
    st.subheader("单相水冷性能随热流密度变化")
    qf_sweep = np.linspace(50, min(heat_flux, 633), 30)
    sp_results = [sp_sim.simulate(qf, sp_flow, T_inlet) for qf in qf_sweep]

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        # 壁温 vs 热流密度
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(qf_sweep, [r.T_wall_max for r in sp_results], 'b-', linewidth=2)
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='T_max=100°C')
        ax.set_xlabel('Heat flux [W/cm²]')
        ax.set_ylabel('T_wall [°C]')
        ax.set_title('Wall Temperature vs Heat Flux (SP-Water)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        st.download_button("📥 导出壁温曲线PNG", data=_fig_to_png(fig),
                           file_name="sp_wall_temp.png", mime="image/png", key="dl_sp_tw")

    with col_chart2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(qf_sweep, [r.COP for r in sp_results], 'g-', linewidth=2)
        ax.set_xlabel('Heat flux [W/cm²]')
        ax.set_ylabel('COP')
        ax.set_title('COP vs Heat Flux (SP-Water)')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        st.download_button("📥 导出COP曲线PNG", data=_fig_to_png(fig),
                           file_name="sp_cop.png", mime="image/png", key="dl_sp_cop")

    # 单相扫描数据导出
    sp_sweep_csv = _make_csv(
        ["热流密度[W/cm²]", "壁温[°C]", "出口温度[°C]", "换热系数[W/(cm²·K)]",
         "压降[kPa]", "热阻[(cm²·K)/W]", "COP", "Re"],
        [[f"{qf:.1f}", f"{r.T_wall_max:.1f}", f"{r.T_outlet:.1f}", f"{r.h_conv_cm2:.3f}",
          f"{r.pressure_drop/1e3:.2f}", f"{r.thermal_resistance:.4f}", f"{r.COP:.0f}", f"{r.Re:.0f}"]
         for qf, r in zip(qf_sweep, sp_results)]
    )
    st.download_button("📥 导出单相扫描数据CSV", data=sp_sweep_csv,
                       file_name="single_phase_sweep.csv", mime="text/csv", key="dl_sp_sweep")

# ===== Tab3: 两相沸腾 =====
with tab3:
    tp_sim = TwoPhaseSimulation(geo, FluidProperties(tp_fluid))
    tp_res = tp_sim.simulate(heat_flux, tp_flow, T_inlet)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("换热系数 h", f"{tp_res.h_conv_cm2:.3f} W/(cm²·K)")
        st.metric("流型", tp_res.flow_pattern_name)
    with col2:
        st.metric("最高壁温", f"{tp_res.T_wall_max:.1f} °C")
        st.metric("出口干度 x", f"{tp_res.x_outlet:.3f}")
    with col3:
        st.metric("压降 ΔP", f"{tp_res.pressure_drop/1e3:.2f} kPa")
        st.metric("CHF", f"{tp_res.CHF:.1f} W/cm²")
    with col4:
        st.metric("热阻 Rth", f"{tp_res.thermal_resistance:.4f} (cm²·K)/W")
        st.metric("COP", f"{tp_res.COP:.0f}")
        st.metric("CHF裕度", f"{tp_res.CHF_margin:.1f}%")

    # 两相结果数据导出
    tp_csv = _make_csv(
        ["参数", "值", "单位"],
        [
            ["热流密度", f"{tp_res.heat_flux:.1f}", "W/cm²"],
            ["质量流量", f"{tp_res.mass_flow_rate:.1f}", "g/s"],
            ["进口温度", f"{tp_res.T_inlet:.1f}", "°C"],
            ["换热系数h", f"{tp_res.h_conv_cm2:.3f}", "W/(cm²·K)"],
            ["最高壁温", f"{tp_res.T_wall_max:.1f}", "°C"],
            ["平均壁温", f"{tp_res.T_wall_avg:.1f}", "°C"],
            ["饱和温度", f"{tp_res.T_sat:.1f}", "°C"],
            ["出口干度x", f"{tp_res.x_outlet:.3f}", ""],
            ["压降ΔP", f"{tp_res.pressure_drop/1e3:.2f}", "kPa"],
            ["摩擦压降", f"{tp_res.dp_friction/1e3:.2f}", "kPa"],
            ["加速压降", f"{tp_res.dp_acceleration/1e3:.2f}", "kPa"],
            ["CHF", f"{tp_res.CHF:.1f}", "W/cm²"],
            ["CHF裕度", f"{tp_res.CHF_margin:.1f}", "%"],
            ["热阻Rth", f"{tp_res.thermal_resistance:.4f}", "(cm²·K)/W"],
            ["COP", f"{tp_res.COP:.0f}", ""],
            ["流型", tp_res.flow_pattern_name, ""],
            ["空隙率", f"{tp_res.void_fraction:.3f}", ""],
        ]
    )
    st.download_button("📥 导出两相结果CSV", data=tp_csv,
                       file_name="two_phase_results.csv", mime="text/csv", key="dl_tp_csv")

    st.divider()

    # 流型图 + 沸腾曲线
    st.subheader("沸腾曲线与流型")
    fig_boil = plot_boiling_curve(tp_sim, tp_flow)
    st.pyplot(fig_boil)
    st.download_button("📥 导出沸腾曲线PNG", data=_fig_to_png(fig_boil),
                       file_name="boiling_curve.png", mime="image/png", key="dl_boil")

    # 沸腾曲线数据导出
    qf_boil = np.linspace(5, 250, 50)
    boil_rows = []
    for qf_b in qf_boil:
        r_b = tp_sim.simulate(qf_b, tp_flow, T_inlet)
        dT_sat = max(r_b.T_wall_avg - r_b.T_sat, 0.1)
        boil_rows.append([f"{qf_b:.1f}", f"{dT_sat:.1f}", f"{r_b.h_conv_cm2:.3f}",
                          f"{r_b.x_outlet:.3f}", r_b.flow_pattern_name])
    boil_csv = _make_csv(
        ["热流密度[W/cm²]", "过热度ΔT_sat[°C]", "换热系数[W/(cm²·K)]",
         "出口干度x", "流型"],
        boil_rows
    )
    st.download_button("📥 导出沸腾曲线数据CSV", data=boil_csv,
                       file_name="boiling_curve_data.csv", mime="text/csv", key="dl_boil_csv")

    # 流型说明
    st.subheader("流型说明 (仿荷叶歧管环形微通道)")
    flow_info = {
        "泡状流": "x < 0.05, 低干度, 分散气泡",
        "弹状流": "0.05 < x < 0.15, Taylor气泡",
        "环状流": "0.15 < x < 0.5, 液膜+气芯",
        "脉动环状流": "0.4 < x < 0.75, 高G, 换热最强",
        "雾状流": "x > 0.75, 液滴弥散, 接近CHF",
    }
    for name, desc in flow_info.items():
        st.markdown(f"- **{name}**: {desc}")

# ===== Tab4: 对比分析 =====
with tab4:
    analysis = ComparativeAnalysis(geo, tp_fluid=tp_fluid)
    cp = analysis.compare_at_condition(heat_flux, sp_flow, tp_flow, T_inlet)

    # 关键指标对比
    st.subheader("关键指标对比")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("换热系数比 (TP/SP)", f"{cp.h_ratio:.2f}×",
                   delta=f"TP: {cp.tp_h_conv:.3f} vs SP: {cp.sp_h_conv:.3f} W/(cm²·K)")
    with col2:
        st.metric("压降比 (TP/SP)", f"{cp.dP_ratio:.2f}×",
                   delta=f"TP: {cp.tp_dP:.2f} vs SP: {cp.sp_dP:.2f} kPa")
    with col3:
        st.metric("热阻比 (TP/SP)", f"{cp.Rth_ratio:.2f}×",
                   delta=f"TP: {cp.tp_Rth:.4f} vs SP: {cp.sp_Rth:.4f}")
    with col4:
        sp_cop_str = f"{cp.sp_COP:.0f}" if cp.sp_COP < 1e6 else f"{cp.sp_COP:.2e}"
        tp_cop_str = f"{cp.tp_COP:.0f}" if cp.tp_COP < 1e6 else f"{cp.tp_COP:.2e}"
        st.metric("COP", f"SP: {sp_cop_str}", delta=f"TP: {tp_cop_str}")

    # 对比指标数据导出
    cmp_csv = _make_csv(
        ["指标", "单相水冷", "两相沸腾", "单位"],
        [
            ["换热系数h", f"{cp.sp_h_conv:.3f}", f"{cp.tp_h_conv:.3f}", "W/(cm²·K)"],
            ["h比(TP/SP)", f"{cp.h_ratio:.2f}", "", ""],
            ["压降ΔP", f"{cp.sp_dP:.2f}", f"{cp.tp_dP:.2f}", "kPa"],
            ["压降比(TP/SP)", f"{cp.dP_ratio:.2f}", "", ""],
            ["热阻Rth", f"{cp.sp_Rth:.4f}", f"{cp.tp_Rth:.4f}", "(cm²·K)/W"],
            ["热阻比(TP/SP)", f"{cp.Rth_ratio:.2f}", "", ""],
            ["COP", f"{cp.sp_COP:.0f}" if cp.sp_COP < 1e6 else f"{cp.sp_COP:.2e}",
             f"{cp.tp_COP:.0f}" if cp.tp_COP < 1e6 else f"{cp.tp_COP:.2e}", ""],
            ["壁温", f"{cp.sp_T_wall:.1f}", f"{cp.tp_T_wall:.1f}", "°C"],
            ["散热功率", f"{cp.sp_Q:.1f}", f"{cp.tp_Q:.1f}", "W"],
            ["CHF", "", f"{cp.tp_CHF:.1f}", "W/cm²"],
            ["流型", "", cp.tp_flow_pattern, ""],
        ]
    )
    st.download_button("📥 导出对比指标CSV", data=cmp_csv,
                       file_name="comparison_results.csv", mime="text/csv", key="dl_cmp_csv")

    st.divider()

    # 对比曲线
    st.subheader("性能对比曲线")
    fig_cmp = plot_comparison_curves(analysis, sp_flow_gs=sp_flow, tp_flow_gs=tp_flow)
    st.pyplot(fig_cmp)
    st.download_button("📥 导出对比曲线PNG", data=_fig_to_png(fig_cmp),
                       file_name="comparison_curves.png", mime="image/png", key="dl_cmp_fig")

    st.divider()

    # 文字报告
    st.subheader("对比分析报告")
    report = analysis.generate_report(
        heat_flux_list=[50, 100, 150, 200, 250],
        sp_flow_gs=sp_flow,
        tp_flow_gs=tp_flow,
    )
    st.text(report)
    st.download_button("📥 导出分析报告TXT", data=report.encode('utf-8'),
                       file_name="comparison_report.txt", mime="text/plain", key="dl_report")

    # 对比扫描数据导出
    qf_cmp_sweep = np.linspace(20, 250, 30)
    cmp_sweep_csv = _make_csv(
        ["热流密度[W/cm²]", "SP-h[W/(cm²·K)]", "TP-h[W/(cm²·K)]", "h比",
         "SP-ΔP[kPa]", "TP-ΔP[kPa]", "SP-Rth", "TP-Rth",
         "SP-COP", "TP-COP", "SP-壁温[°C]", "TP-壁温[°C]", "TP-流型"],
        [[f"{qf:.1f}",
          f"{cp_i.sp_h_conv:.3f}", f"{cp_i.tp_h_conv:.3f}", f"{cp_i.h_ratio:.2f}",
          f"{cp_i.sp_dP:.2f}", f"{cp_i.tp_dP:.2f}",
          f"{cp_i.sp_Rth:.4f}", f"{cp_i.tp_Rth:.4f}",
          f"{cp_i.sp_COP:.0f}" if cp_i.sp_COP < 1e6 else f"{cp_i.sp_COP:.2e}",
          f"{cp_i.tp_COP:.0f}" if cp_i.tp_COP < 1e6 else f"{cp_i.tp_COP:.2e}",
          f"{cp_i.sp_T_wall:.1f}", f"{cp_i.tp_T_wall:.1f}", cp_i.tp_flow_pattern]
         for qf, cp_i in zip(qf_cmp_sweep, analysis.sweep_comparison(qf_cmp_sweep, sp_flow_gs=sp_flow, tp_flow_gs=tp_flow))]
    )
    st.download_button("📥 导出对比扫描数据CSV", data=cmp_sweep_csv,
                       file_name="comparison_sweep.csv", mime="text/csv", key="dl_cmp_sweep")

# ===== Tab5: 敏感性分析 =====
with tab5:
    st.subheader("COP 热阻敏感性分析")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 单相水冷")
        fig_sp = plot_sensitivity_analysis("single_phase")
        st.pyplot(fig_sp)
        st.download_button("📥 导出单相敏感性PNG", data=_fig_to_png(fig_sp),
                           file_name="sensitivity_sp.png", mime="image/png", key="dl_sens_sp")

    with col2:
        st.markdown("### 两相沸腾 (HFE-7100)")
        fig_tp = plot_sensitivity_analysis("two_phase")
        st.pyplot(fig_tp)
        st.download_button("📥 导出两相敏感性PNG", data=_fig_to_png(fig_tp),
                           file_name="sensitivity_tp.png", mime="image/png", key="dl_sens_tp")

    # 敏感性分析数据导出
    from core.fluid_properties import FluidProperties as _FP
    _geo_sens = ManifoldRingChannelGeometry()
    _sp_sens = SinglePhaseSimulation(_geo_sens, _FP("water"))
    _tp_sens = TwoPhaseSimulation(_geo_sens, _FP(tp_fluid))
    qf_sp_sens = np.linspace(50, 633, 25)
    mf_sp_sens = np.linspace(2, 20, 20)
    qf_tp_sens = np.linspace(20, 250, 25)
    mf_tp_sens = np.linspace(3, 12, 20)
    sp_sens_csv = _make_csv(
        ["热流密度[W/cm²]", "COP"],
        [[f"{qf:.1f}", f"{_sp_sens.simulate(qf, 5.0).COP:.0f}"] for qf in qf_sp_sens]
    )
    sp_sens_mf_csv = _make_csv(
        ["流量[g/s]", "COP", "热阻[(cm²·K)/W]"],
        [[f"{mf:.1f}", f"{_sp_sens.simulate(200, mf).COP:.0f}",
          f"{_sp_sens.simulate(200, mf).thermal_resistance:.4f}"] for mf in mf_sp_sens]
    )
    tp_sens_csv = _make_csv(
        ["热流密度[W/cm²]", "COP"],
        [[f"{qf:.1f}", f"{_tp_sens.simulate(qf, 6.0).COP:.0f}"] for qf in qf_tp_sens]
    )
    tp_sens_mf_csv = _make_csv(
        ["流量[g/s]", "COP", "热阻[(cm²·K)/W]"],
        [[f"{mf:.1f}", f"{_tp_sens.simulate(100, mf).COP:.0f}",
          f"{_tp_sens.simulate(100, mf).thermal_resistance:.4f}"] for mf in mf_tp_sens]
    )
    sens_col1, sens_col2 = st.columns(2)
    with sens_col1:
        st.download_button("📥 导出单相敏感性数据CSV", data=sp_sens_csv,
                           file_name="sensitivity_sp_qf.csv", mime="text/csv", key="dl_sens_sp_csv")
    with sens_col2:
        st.download_button("📥 导出两相敏感性数据CSV", data=tp_sens_csv,
                           file_name="sensitivity_tp_qf.csv", mime="text/csv", key="dl_sens_tp_csv")

    st.divider()
    st.subheader("关键发现")

    st.markdown("""
    **单相水冷 (去离子水)**:
    - COP 随热流密度增加而增加 (更多热量被流体带走)
    - 热阻随流量增加而下降 (换热增强)
    - 极限热流密度受壁温约束 (T_max < 100°C)
    - 文献极值: 633 W/cm², COP=1.8×10⁵, ΔP=25.22 kPa

    **两相沸腾 (HFE-7100)**:
    - 换热系数远高于同工质单相对流 (3-8倍), 利用汽化潜热
    - 壁温在两相区几乎恒定 (等温特性)
    - COP 在低热流密度时高, 接近CHF时骤降
    - 脉动环状流区换热最强但压降也最大
    - 文献极值: CHF=267.05 W/cm², h=2.13 W/(cm²·K), COP=18906

    **仿荷叶歧管结构优势**:
    - 中心入口射流冲击 → 高效近端换热
    - 环形分配 → 均匀流量分配
    - 径向短通道 → 低压降 (比传统降低50.72%)
    - 温度均匀性提升43.74%
    """)

    st.divider()
    st.subheader("全局敏感性分析 (Morris方法)")

    sa_type = st.radio("分析类型", ["单相水冷", "两相沸腾"], horizontal=True, key="sa_type_radio")
    sa_sim_type = "single_phase" if sa_type == "单相水冷" else "two_phase"

    if st.button("运行全局敏感性分析", key="run_sa"):
        with st.spinner("正在运行Morris敏感性分析..."):
            analyzer = SensitivityAnalyzer(sa_sim_type)
            analyzer.run_morris(n_trajectories=5)

            col_sa1, col_sa2 = st.columns(2)
            with col_sa1:
                st.markdown("##### 敏感性热力图")
                fig_heat = analyzer.plot_sensitivity_heatmap()
                st.pyplot(fig_heat)
                st.download_button("📥 导出热力图PNG", data=_fig_to_png(fig_heat),
                                  file_name="sensitivity_heatmap.png", mime="image/png", key="dl_sa_heat")

            with col_sa2:
                st.markdown("##### 龙卷风图")
                fig_torn = analyzer.plot_tornado()
                st.pyplot(fig_torn)
                st.download_button("📥 导出龙卷风图PNG", data=_fig_to_png(fig_torn),
                                  file_name="sensitivity_tornado.png", mime="image/png", key="dl_sa_torn")

# ===== Tab6: 模型验证 =====
with tab6:
    st.subheader("模型验证与不确定性量化")
    st.info("基于文献实验数据的多点验证矩阵，含±30%关联式误差带")

    validator = ModelValidator()

    col_val1, col_val2 = st.columns(2)
    with col_val1:
        st.markdown("##### 预测值 vs 文献参考值")
        fig_parity = validator.plot_validation()
        st.pyplot(fig_parity)
        st.download_button("📥 导出验证对比图PNG", data=_fig_to_png(fig_parity),
                          file_name="validation_parity.png", mime="image/png", key="dl_val_parity")

    with col_val2:
        st.markdown("##### 误差分布")
        fig_err = validator.plot_error_distribution()
        st.pyplot(fig_err)
        st.download_button("📥 导出误差分布图PNG", data=_fig_to_png(fig_err),
                          file_name="validation_errors.png", mime="image/png", key="dl_val_err")

    st.divider()

    # 验证统计
    stats = validator.compute_statistics()
    st.markdown("##### 验证统计汇总")
    for key, val in stats.items():
        if isinstance(val, (int, float)):
            st.metric(key, f"{val:.2f}")

    st.divider()

    # 验证报告
    st.markdown("##### 验证报告 (Markdown)")
    report_md = validator.generate_report()
    st.download_button("📥 导出验证报告MD", data=report_md.encode('utf-8'),
                      file_name="validation_report.md", mime="text/markdown", key="dl_val_report")
    with st.expander("查看报告内容"):
        st.markdown(report_md)

# ===== 底部 =====
st.divider()
st.caption("仿荷叶歧管微通道冷板仿真工具 v1.0 | 基于浙大吴赞课题组文献 | Xin Z, et al. Energy (2025) & ECM (2026)")

if __name__ == "__main__":
    try:
        from streamlit.runtime import exists as st_exists
    except ImportError:
        def st_exists(): return False

    if not st_exists():
        import sys
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", __file__, "--browser.gatherUsageStats", "false"]
        sys.exit(stcli.main())
