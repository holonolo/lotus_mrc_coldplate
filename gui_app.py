"""
仿荷叶歧管微通道冷板 - Streamlit GUI
=============================================
单相水冷 vs 两相沸腾冷却 对比分析工具

运行: streamlit run gui_app.py
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib
import core.geometry
import core.fluid_properties
import core.single_phase
import core.two_phase
import core.comparison
import core.visualization

# 强制重载模块以解决 Streamlit 缓存旧代码的问题
importlib.reload(core.geometry)
importlib.reload(core.fluid_properties)
importlib.reload(core.single_phase)
importlib.reload(core.two_phase)
importlib.reload(core.comparison)
importlib.reload(core.visualization)

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
)

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

    # --- 几何参数 ---
    st.subheader("冷板几何")
    coldplate_diameter = st.number_input("冷板直径 [mm]", value=30.0, min_value=0.0, max_value=500.0)
    chip_area = st.number_input("圆形热源面积 [mm²]", value=314.16, min_value=50.0, max_value=10000.0)
    chip_length = st.number_input("圆形热源直径 [mm]", value=20.0, min_value=5.0, max_value=100.0)
    channel_width = st.number_input("微通道宽度 [mm]", value=0.30, min_value=0.05, max_value=1.0, step=0.01, format="%.3f")
    channel_height = st.number_input("微通道深度 [mm]", value=1.4, min_value=0.1, max_value=10.0, step=0.1)
    fin_width = st.number_input("翅片厚度 [mm]", value=0.30, min_value=0.05, max_value=5.0, step=0.01, format="%.3f")
    manifold_height = st.number_input("歧管层高度 [mm]", value=1.8, min_value=0.2, max_value=5.0, step=0.1)
    n_rings = st.number_input("环形微通道数", value=13, min_value=2, max_value=100)
    n_sectors = st.number_input("总歧管分流流道数", value=16, min_value=4, max_value=50, step=2)
    inlet_diameter = st.number_input("中心入口直径 [mm]", value=4.5, min_value=0.0, max_value=200.0)

    st.divider()

    # --- 高级几何参数 ---
    st.subheader("高级几何参数")
    coldplate_length = st.number_input("冷板长度 [mm]", value=coldplate_diameter)
    coldplate_width = st.number_input("冷板宽度 [mm]", value=coldplate_diameter)
    total_thickness = st.number_input("冷板总厚度 [mm]", value=4.2)
    base_thickness = st.number_input("基板厚度 [mm]", value=0.5)
    ring_spacing = st.number_input("环形通道间距 [mm]", value=0.6, step=0.01, format="%.3f")
    inlet_slot_width = st.number_input("窄进液歧管缝宽 [mm]", value=0.5, step=0.01, format="%.3f")
    outlet_slot_width = st.number_input("宽出液歧管缝宽 [mm]", value=1.5, step=0.01, format="%.3f")
    channel_shape_name = st.selectbox(
        "微通道截面形状",
        ["RECTANGULAR", "CIRCULAR", "TRAPEZOIDAL", "TRIANGULAR"],
        index=0,
    )
    channel_diameter = st.number_input("圆形截面直径 [mm]", value=0.3, step=0.01, format="%.3f")
    trapezoid_side_angle = st.number_input("梯形侧壁角 [°]", value=54.7)
    substrate_material = st.text_input("基板材料", value="copper")
    sintering_material = st.text_input("烧结材料", value="silver")

    st.divider()

    # --- 工况参数 ---
    st.subheader("运行工况")
    heat_flux = st.number_input("热流密度 [W/cm²]", value=633.0, min_value=1.0, max_value=700.0)
    sp_flow = st.number_input("单相质量流量 [g/s]", value=39.0, min_value=0.5, max_value=200.0)
    tp_flow = st.number_input("两相质量流量 [g/s]", value=6.0, min_value=1.0, max_value=200.0)
    T_inlet = st.number_input("进口温度 [°C]", value=20.0, min_value=10.0, max_value=80.0)

    st.divider()

    # --- 工质选择 ---
    st.subheader("两相工质")
    tp_fluid = st.selectbox("工质类型", ["HFE7100", "R245fa", "R1233zdE", "water"], index=0)

    st.divider()
    st.info("文献来源:\n- Xin Z, et al. Energy, 2025\n- Xin Z, et al. ECM, 2026\n- 浙大吴赞课题组")

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

# ===== 主界面标签页 =====
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📐 几何结构", "💧 单相水冷", "🔥 两相沸腾", "📊 对比分析", "📈 敏感性分析"
])

# ===== Tab1: 几何结构 =====
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("俯视图")
        fig_top = plot_geometry_topview(geo)
        st.pyplot(fig_top)

    with col2:
        st.subheader("截面图")
        fig_cross = plot_geometry_crosssection(geo)
        st.pyplot(fig_cross)

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

    with col_chart2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(qf_sweep, [r.COP for r in sp_results], 'g-', linewidth=2)
        ax.set_xlabel('Heat flux [W/cm²]')
        ax.set_ylabel('COP')
        ax.set_title('COP vs Heat Flux (SP-Water)')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

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

    st.divider()

    # 流型图 + 沸腾曲线
    st.subheader("沸腾曲线与流型")
    fig_boil = plot_boiling_curve(tp_sim, tp_flow)
    st.pyplot(fig_boil)

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

    st.divider()

    # 对比曲线
    st.subheader("性能对比曲线")
    fig_cmp = plot_comparison_curves(analysis, sp_flow_gs=sp_flow, tp_flow_gs=tp_flow)
    st.pyplot(fig_cmp)

    st.divider()

    # 文字报告
    st.subheader("对比分析报告")
    report = analysis.generate_report(
        heat_flux_list=[50, 100, 150, 200, 250],
        sp_flow_gs=sp_flow,
        tp_flow_gs=tp_flow,
    )
    st.text(report)

# ===== Tab5: 敏感性分析 =====
with tab5:
    st.subheader("COP 热阻敏感性分析")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 单相水冷")
        fig_sp = plot_sensitivity_analysis("single_phase")
        st.pyplot(fig_sp)

    with col2:
        st.markdown("### 两相沸腾 (HFE-7100)")
        fig_tp = plot_sensitivity_analysis("two_phase")
        st.pyplot(fig_tp)

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
