"""仿荷叶歧管环形微通道冷板 - 两相沸腾仿真回归测试
=============================================

测试 TwoPhaseSimulation.simulate():
- 返回 TwoPhaseResult 完整结构 + FlowPattern 枚举
- 物理边界 (T_wall > T_sat, x_outlet∈[0,1], Δp>0, CHF>0, Q 守恒)
- 回归基线锁 (当前第一性原理模型输出, 防止未来重构漂移)
- 物理单调性 (q↑ → h↑, T_wall↑, CHF_margin↓)

注: 仿真仅读取 FluidProperties._TABLES 表值, 不调用 CoolProp,
    故结果跨平台确定, 可用紧容差回归锁。

文献基准工况: HFE-7100, q=255 W/cm², m=6 g/s, T_in=20°C
"""

import pytest

from core.geometry import ManifoldRingChannelGeometry
from core.fluid_properties import FluidProperties
from core.two_phase import TwoPhaseSimulation, TwoPhaseResult, FlowPattern


@pytest.fixture
def sim():
    return TwoPhaseSimulation(
        ManifoldRingChannelGeometry(),
        FluidProperties("HFE7100"),
    )


@pytest.fixture
def res_benchmark(sim):
    """文献基准工况结果 (q=255, m=6, T_in=20)"""
    return sim.simulate(255, 6.0, 20.0)


class TestResultStructure:
    def test_returns_two_phase_result(self, res_benchmark):
        assert isinstance(res_benchmark, TwoPhaseResult)

    def test_flow_pattern_is_enum(self, res_benchmark):
        assert isinstance(res_benchmark.flow_pattern, FlowPattern)
        assert isinstance(res_benchmark.flow_pattern_name, str)
        assert len(res_benchmark.flow_pattern_name) > 0


class TestPhysicalBounds:
    """物理边界 (文献基准工况)"""

    def test_wall_above_saturation(self, res_benchmark):
        assert res_benchmark.T_wall_avg > res_benchmark.T_sat

    def test_outlet_quality_in_unit_interval(self, res_benchmark):
        assert 0.0 <= res_benchmark.x_outlet <= 1.0

    def test_positive_pressure_drop(self, res_benchmark):
        assert res_benchmark.pressure_drop > 0.0

    def test_positive_CHF(self, res_benchmark):
        assert res_benchmark.CHF > 0.0

    def test_positive_heat_transfer_coefficient(self, res_benchmark):
        assert res_benchmark.h_conv_cm2 > 0.0

    def test_total_heat_conservation(self, res_benchmark):
        # Q_total = q [W/cm²] * 1e4 * A_chip [m²]
        expected = 255 * 1e4 * 314.16e-6
        assert res_benchmark.Q_total == pytest.approx(expected, rel=1e-3)

    def test_positive_thermal_resistance(self, res_benchmark):
        assert res_benchmark.thermal_resistance > 0.0

    def test_pressure_drop_components_non_negative(self, res_benchmark):
        # 摩擦/加速度/重力/歧管压降分量不应为负
        for attr in ["dp_friction", "dp_acceleration", "dp_gravity", "dp_manifold"]:
            assert getattr(res_benchmark, attr) >= 0.0


class TestRegressionBaseline:
    """回归基线锁: 当前模型输出 (HEAD @ 9b50956)

    锁定当前第一性原理模型在文献工况的行为。若未来重构改变了数值,
    测试会失败提示开发者确认改动是否预期 —— 预期则更新基线,
    非预期则捕获了回归。
    """

    def test_h_conv(self, res_benchmark):
        assert res_benchmark.h_conv_cm2 == pytest.approx(1.793, rel=0.03)

    def test_CHF(self, res_benchmark):
        assert res_benchmark.CHF == pytest.approx(196.3, rel=0.03)

    def test_T_wall_avg(self, res_benchmark):
        assert res_benchmark.T_wall_avg == pytest.approx(87.5, abs=1.5)

    def test_pressure_drop(self, res_benchmark):
        assert res_benchmark.pressure_drop / 1e3 == pytest.approx(18.69, rel=0.03)

    def test_outlet_quality(self, res_benchmark):
        assert res_benchmark.x_outlet == pytest.approx(0.759, abs=0.02)


class TestMonotonicity:
    """热流密度扫描的物理单调性"""

    def test_h_increases_with_heat_flux(self, sim):
        hs = [sim.simulate(q, 6.0, 20.0).h_conv_cm2 for q in [100, 200, 300]]
        assert hs[1] > hs[0] and hs[2] > hs[1]

    def test_CHF_margin_decreases_with_heat_flux(self, sim):
        margins = [sim.simulate(q, 6.0, 20.0).CHF_margin for q in [100, 200, 300]]
        assert margins[1] < margins[0] and margins[2] < margins[1]

    def test_wall_temp_increases_with_heat_flux(self, sim):
        ts = [sim.simulate(q, 6.0, 20.0).T_wall_avg for q in [100, 200, 300]]
        assert ts[1] > ts[0] and ts[2] > ts[1]
