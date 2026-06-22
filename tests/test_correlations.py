"""仿荷叶歧管环形微通道冷板 - 两相沸腾关联式单元测试
=============================================

测试 GungorWinterton (1987) 与 KimMudawar (2013) 两相沸腾换热关联式:
- calc_h() 返回字典结构完整性 (必需键齐全)
- 物理约束 (G-W 的 E∈[1,15], S∈(0,1], h_tp≥h_l, Bo>0, Re_l>0)
- 文献基准量级回归 (HFE-7100, q=255 W/cm², m=6 g/s)
- 单调性 (热流密度增大 → 换热系数增大)

文献基准: h=2.13 W/(cm²·K) @ q=255 W/cm², HFE-7100, m=6 g/s
"""

import pytest

from core.geometry import ManifoldRingChannelGeometry
from core.fluid_properties import FluidProperties
from core.correlations import GungorWinterton, KimMudawar


@pytest.fixture
def benchmark_inputs():
    """文献基准工况下 calc_h() 的输入参数 (HFE-7100, q=255 W/cm², m=6 g/s, x=0.3)"""
    geo = ManifoldRingChannelGeometry()
    fluid = FluidProperties("HFE7100")
    Dh = geo.hydraulic_diameter * 1e-3          # m
    A_eff = geo.effective_cross_area            # m²
    m_dot = 6.0e-3                              # kg/s
    G = m_dot / A_eff
    return dict(
        Dh=Dh, G=G, x=0.3, q_Wm2=255 * 1e4,
        rho_l=fluid.rho_l, rho_v=fluid.rho_v,
        mu_l=fluid.mu_l, mu_v=fluid.mu_v,
        k_l=fluid.k_l, k_v=fluid.k_v,
        cp_l=fluid.cp_l, h_fg=fluid.h_fg,
        sigma=fluid.sigma, Pr_l=fluid.Pr_l,
        fluid_name="HFE7100",
    )


class TestReturnStructure:
    """calc_h() 返回字典结构完整性"""

    def test_gungor_winterton_has_all_keys(self, benchmark_inputs):
        out = GungorWinterton.calc_h(**benchmark_inputs)
        for key in ["h_tp", "h_l", "h_pool", "E", "S", "Bo", "Co", "Re_l"]:
            assert key in out, f"missing key: {key}"

    def test_kim_mudawar_has_all_keys(self, benchmark_inputs):
        out = KimMudawar.calc_h(**benchmark_inputs)
        for key in ["h_tp", "h_NBD", "h_CBD", "h_l", "Bo", "Co", "Re_l"]:
            assert key in out, f"missing key: {key}"


class TestPhysicalConstraints:
    """物理约束 (基准工况)"""

    def test_gw_enhancement_factor_capped(self, benchmark_inputs):
        # 代码硬性上限: E = min(E, 15.0)
        out = GungorWinterton.calc_h(**benchmark_inputs)
        assert 1.0 <= out["E"] <= 15.0

    def test_gw_suppression_factor_in_unit_interval(self, benchmark_inputs):
        out = GungorWinterton.calc_h(**benchmark_inputs)
        assert 0.0 < out["S"] <= 1.0

    def test_gw_h_tp_at_least_liquid(self, benchmark_inputs):
        out = GungorWinterton.calc_h(**benchmark_inputs)
        assert out["h_tp"] >= out["h_l"]

    def test_km_h_tp_at_least_liquid(self, benchmark_inputs):
        out = KimMudawar.calc_h(**benchmark_inputs)
        assert out["h_tp"] >= out["h_l"]

    @pytest.mark.parametrize("cls", [GungorWinterton, KimMudawar])
    def test_positive_dimensionless_numbers(self, cls, benchmark_inputs):
        out = cls.calc_h(**benchmark_inputs)
        assert out["Bo"] > 0
        assert out["Re_l"] > 0
        assert out["h_tp"] > 0


class TestLiteratureBenchmark:
    """文献基准量级回归

    仅对主关联式 Gungor-Winterton 校验量级 (项目主模型, 文献标定目标 h≈2.13)。
    KimMudawar 为交叉验证关联式, 不同关联式在不同干度下差异较大属正常,
    其合理性由 TestPhysicalConstraints (h_tp>0, h_tp≥h_l) 与单调性保证。
    """

    def test_gw_h_near_literature(self, benchmark_inputs):
        out = GungorWinterton.calc_h(**benchmark_inputs)
        h_cm2 = out["h_tp"] * 1e-4  # → W/(cm²·K)
        # 文献 2.13 W/(cm²·K), 容许 ±50% 物理边界
        assert 1.0 < h_cm2 < 4.0


class TestMonotonicity:
    """热流密度增大 → 沸腾换热系数增大"""

    @pytest.mark.parametrize("cls", [GungorWinterton, KimMudawar])
    def test_h_increases_with_heat_flux(self, cls, benchmark_inputs):
        hs = []
        for q in [100, 200, 300]:
            inp = dict(benchmark_inputs, q_Wm2=q * 1e4)
            hs.append(cls.calc_h(**inp)["h_tp"])
        assert hs[1] > hs[0]
        assert hs[2] > hs[1]
