"""仿荷叶歧管环形微通道冷板 - 单相水冷仿真回归测试
=============================================

测试 SinglePhaseSimulation.simulate():
- 返回 SinglePhaseResult 完整结构
- 物理边界 (T_wall > T_inlet, Δp>0, Re>0, Q 守恒, COP>0)
- 文献基准回归 (ΔP ≈ 25.22 kPa, 容差 5%)
- 回归基线锁 (当前模型默认工况输出)
- Dean 涡二次流增强因子 > 1 + 层流判定

注: 仿真仅读取 FluidProperties._TABLES 表值, 不调用 CoolProp,
    故结果跨平台确定, 可用紧容差回归锁。

文献基准工况: 水, q=100 W/cm², m=39 g/s, T_in=20°C
"""

import pytest

from core.geometry import ManifoldRingChannelGeometry
from core.fluid_properties import FluidProperties
from core.single_phase import SinglePhaseSimulation, SinglePhaseResult


@pytest.fixture
def sim():
    return SinglePhaseSimulation(
        ManifoldRingChannelGeometry(),
        FluidProperties("water"),
    )


@pytest.fixture
def res_default(sim):
    """默认工况结果 (q=100, m=39, T_in=20, 水)"""
    return sim.simulate()


class TestResultStructure:
    def test_returns_single_phase_result(self, res_default):
        assert isinstance(res_default, SinglePhaseResult)


class TestPhysicalBounds:
    """物理边界 (默认工况)"""

    def test_wall_above_inlet(self, res_default):
        assert res_default.T_wall_avg > res_default.T_inlet

    def test_outlet_above_inlet(self, res_default):
        assert res_default.T_outlet > res_default.T_inlet

    def test_wall_max_above_average(self, res_default):
        assert res_default.T_wall_max >= res_default.T_wall_avg

    def test_positive_pressure_drop(self, res_default):
        assert res_default.pressure_drop > 0.0

    def test_positive_reynolds(self, res_default):
        assert res_default.Re > 0.0

    def test_positive_nusselt(self, res_default):
        assert res_default.Nu > 0.0

    def test_total_heat_conservation(self, res_default):
        expected = 100 * 1e4 * 314.16e-6
        assert res_default.Q_total == pytest.approx(expected, rel=1e-3)

    def test_positive_COP(self, res_default):
        assert res_default.COP > 0.0


class TestLiteratureBenchmark:
    """文献基准: ΔP ≈ 25.22 kPa (single_phase.py docstring)"""

    def test_pressure_drop_matches_literature(self, res_default):
        assert res_default.pressure_drop / 1e3 == pytest.approx(25.22, rel=0.05)


class TestRegressionBaseline:
    """回归基线锁: 当前模型默认工况输出 (HEAD @ 9b50956)"""

    def test_h_conv(self, res_default):
        assert res_default.h_conv_cm2 == pytest.approx(4.3539, rel=0.03)

    def test_pressure_drop(self, res_default):
        assert res_default.pressure_drop / 1e3 == pytest.approx(24.84, rel=0.03)

    def test_reynolds(self, res_default):
        assert res_default.Re == pytest.approx(247.85, rel=0.03)

    def test_dean_enhancement(self, res_default):
        # Mori-Nakayama Dean 涡二次流增强因子 > 1 (无增强时 = 1.0)
        assert res_default.enhancement_avg > 1.0
        assert res_default.enhancement_avg == pytest.approx(2.73, rel=0.05)

    def test_laminar_flow_regime(self, res_default):
        # 默认工况 Re < 2300, 为层流
        assert res_default.Re < 2300


class TestMonotonicity:
    """热流密度扫描的物理单调性"""

    def test_wall_temp_increases_with_heat_flux(self, sim):
        ts = [sim.simulate(q, 39.0, 20.0).T_wall_avg for q in [50, 100, 150]]
        assert ts[1] > ts[0] and ts[2] > ts[1]
