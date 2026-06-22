"""仿荷叶歧管环形微通道冷板 - 工质物性库单元测试
=============================================

测试 FluidProperties 对 4 种工质 (water / HFE7100 / R245fa / R1233zdE):
- _TABLES 内置近似物性的 @property 精确锁定 (25°C, 1 atm)
- 物理合理性边界 (液相密度 > 汽相, h_fg > 0, 液相粘度 > 汽相等)
- 未知工质 fallback 与 get() getter 行为

参考值来自 core/fluid_properties.py::_TABLES。
"""

import pytest

from core.fluid_properties import FluidProperties

FLUIDS = ["water", "HFE7100", "R245fa", "R1233zdE"]

# (T_sat, rho_l, rho_v, cp_l, k_l, mu_l, mu_v, h_fg, sigma, Pr_l) — 来自 _TABLES
TABLE = {
    "water":    (100.0, 997.0,  0.6, 4182.0, 0.607, 8.9e-4, 1.2e-5, 2257e3, 0.072,  6.13),
    "HFE7100":  (61.0,  1510.0, 10.0, 1183.0, 0.069, 5.8e-4, 1.0e-5, 112e3,  0.0136, 9.9),
    "R245fa":   (15.1,  1330.0, 8.0,  1300.0, 0.087, 3.7e-4, 1.0e-5, 196e3,  0.015,  5.5),
    "R1233zdE": (18.3,  1250.0, 7.0,  1200.0, 0.082, 3.8e-4, 9.5e-6, 195e3,  0.015,  5.6),
}


@pytest.mark.parametrize("name", FLUIDS)
def test_T_sat(name):
    assert FluidProperties(name).T_sat == pytest.approx(TABLE[name][0], abs=1e-6)


@pytest.mark.parametrize("name", FLUIDS)
def test_rho_l(name):
    assert FluidProperties(name).rho_l == pytest.approx(TABLE[name][1], rel=1e-6)


@pytest.mark.parametrize("name", FLUIDS)
def test_rho_v(name):
    assert FluidProperties(name).rho_v == pytest.approx(TABLE[name][2], rel=1e-6)


@pytest.mark.parametrize("name", FLUIDS)
def test_cp_l(name):
    assert FluidProperties(name).cp_l == pytest.approx(TABLE[name][3], rel=1e-6)


@pytest.mark.parametrize("name", FLUIDS)
def test_k_l(name):
    assert FluidProperties(name).k_l == pytest.approx(TABLE[name][4], rel=1e-6)


@pytest.mark.parametrize("name", FLUIDS)
def test_h_fg(name):
    assert FluidProperties(name).h_fg == pytest.approx(TABLE[name][7], rel=1e-6)


@pytest.mark.parametrize("name", FLUIDS)
def test_Pr_l(name):
    assert FluidProperties(name).Pr_l == pytest.approx(TABLE[name][9], rel=1e-6)


class TestPhysicalBounds:
    """所有工质的物理合理性边界"""

    @pytest.mark.parametrize("name", FLUIDS)
    def test_liquid_denser_than_vapor(self, name):
        fp = FluidProperties(name)
        assert fp.rho_l > fp.rho_v

    @pytest.mark.parametrize("name", FLUIDS)
    def test_positive_properties(self, name):
        fp = FluidProperties(name)
        for v in [fp.cp_l, fp.cp_v, fp.k_l, fp.k_v, fp.h_fg, fp.sigma, fp.Pr_l]:
            assert v > 0

    @pytest.mark.parametrize("name", FLUIDS)
    def test_liquid_more_viscous_than_vapor(self, name):
        fp = FluidProperties(name)
        assert fp.mu_l > fp.mu_v

    @pytest.mark.parametrize("name", FLUIDS)
    def test_T_sat_in_physical_range(self, name):
        # 冷却/制冷工质常压沸点应在合理区间
        assert -60 < FluidProperties(name).T_sat < 150

    @pytest.mark.parametrize("name", FLUIDS)
    def test_water_has_highest_latent_heat(self, name):
        # 水的汽化潜热应显著高于介电/制冷工质
        if name != "water":
            assert FluidProperties("water").h_fg > FluidProperties(name).h_fg


class TestFallbackAndGetter:
    """未知工质 fallback 与 get() 行为"""

    def test_unknown_fluid_falls_back_to_water(self):
        # _TABLES.get(name, water_table): 未知工质回退到水
        fp = FluidProperties("nonexistent_fluid")
        assert fp.T_sat == 100.0
        assert fp.rho_l == 997.0

    def test_get_without_T_returns_table_value(self):
        # 不传 T → 跳过 CoolProp, 直接返回表值
        fp = FluidProperties("water")
        assert fp.get("T_sat") == 100.0
        assert fp.get("rho_l") == 997.0

    def test_property_matches_get_without_T(self):
        # @property 与 get(prop) 在无 T 时应一致
        fp = FluidProperties("HFE7100")
        assert fp.rho_l == fp.get("rho_l")
        assert fp.h_fg == fp.get("h_fg")
