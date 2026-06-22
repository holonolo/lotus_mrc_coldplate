"""仿荷叶歧管环形微通道冷板 - 几何模型单元测试
=============================================

测试 ManifoldRingChannelGeometry 的派生几何参数:
- 确定性物理量精确锁 (水力直径、截面积、并联通道数等可手算的量)
- 物理合理性边界 (孔隙率、换热面积、材料属性)
- 参数化响应 (修改输入参数后派生量正确变化)

文献基准几何 (默认值):
    chip_area=314.16 mm², n_rings=13, n_sectors=16,
    channel_width=0.3 mm, channel_height=1.4 mm,
    inlet_diameter=4.5 mm, ring_spacing=0.6 mm
"""

import pytest

from core.geometry import ManifoldRingChannelGeometry


class TestDerivedGeometryExact:
    """派生几何参数精确锁定 (默认文献基准几何)"""

    @pytest.fixture
    def geo(self):
        return ManifoldRingChannelGeometry()

    def test_hydraulic_diameter(self, geo):
        # Dh = 2*w*h/(w+h) = 2*0.3*1.4/(0.3+1.4) = 0.4941 mm
        assert geo.hydraulic_diameter == pytest.approx(0.4941, abs=1e-4)

    def test_channel_cross_area(self, geo):
        # A = w*h = 0.3*1.4 = 0.42 mm²
        assert geo.channel_cross_area == pytest.approx(0.42, abs=1e-6)

    def test_ring_and_sector_counts(self, geo):
        assert geo.n_rings == 13
        assert geo.n_sectors == 16

    def test_effective_channels(self, geo):
        # 并联流动路径 = n_rings * n_sectors = 13 * 16 = 208
        assert geo.effective_channels == 208

    def test_first_ring_radius(self, geo):
        # r0 = inlet_diameter/2 + 0.5*ring_spacing = 2.25 + 0.30 = 2.55 mm
        assert geo.ring_radii[0] == pytest.approx(2.55, abs=1e-6)

    def test_ring_radii_count_matches_n_rings(self, geo):
        assert len(geo.ring_radii) == geo.n_rings == 13

    def test_ring_radii_strictly_increasing(self, geo):
        for r_prev, r_curr in zip(geo.ring_radii, geo.ring_radii[1:]):
            assert r_curr > r_prev

    def test_material_conductivity(self, geo):
        assert geo.k_substrate == 401.0  # 铜 [W/(m·K)]
        assert geo.k_sinter == 254.0     # 银烧结 [W/(m·K)]


class TestPhysicalBounds:
    """物理合理性边界 (默认几何)"""

    @pytest.fixture
    def geo(self):
        return ManifoldRingChannelGeometry()

    def test_porosity_in_unit_interval(self, geo):
        assert 0.0 < geo.porosity < 1.0

    def test_heat_transfer_area_positive(self, geo):
        assert geo.total_heat_transfer_area > 0.0

    def test_effective_cross_area_positive(self, geo):
        assert geo.effective_cross_area > 0.0

    def test_channels_per_ring_all_positive(self, geo):
        assert all(n > 0 for n in geo.channels_per_ring)

    def test_total_channels_equals_sum(self, geo):
        assert geo.total_channels == sum(geo.channels_per_ring)

    def test_flow_lengths_positive(self, geo):
        assert geo.L_flow_avg > 0.0
        assert all(L > 0 for L in geo.L_flow_rings)


class TestParametricResponse:
    """参数化响应: 修改输入后派生量正确变化"""

    def test_wider_channel_increases_hydraulic_diameter(self):
        geo_default = ManifoldRingChannelGeometry()
        geo_wide = ManifoldRingChannelGeometry(channel_width=0.5)
        assert geo_wide.hydraulic_diameter > geo_default.hydraulic_diameter

    def test_effective_channels_independent_of_width(self):
        # effective_channels = n_rings * n_sectors, 与 channel_width 无关
        geo = ManifoldRingChannelGeometry(channel_width=0.5)
        assert geo.effective_channels == 208

    def test_more_rings_scales_effective_channels(self):
        geo = ManifoldRingChannelGeometry(n_rings=20)
        assert geo.effective_channels == 20 * 16


class TestDictAndSummary:
    """get_params_dict / summary 接口"""

    @pytest.fixture
    def geo(self):
        return ManifoldRingChannelGeometry()

    def test_params_dict_contains_expected_keys(self, geo):
        d = geo.get_params_dict()
        for key in ["chip_area_mm2", "hydraulic_diameter_mm", "total_channels",
                    "porosity", "n_rings", "n_sectors", "ring_radii_mm",
                    "channels_per_ring", "L_flow_avg_mm"]:
            assert key in d

    def test_params_dict_values(self, geo):
        d = geo.get_params_dict()
        assert d["n_rings"] == 13
        assert d["hydraulic_diameter_mm"] == pytest.approx(0.4941, abs=1e-4)

    def test_summary_is_descriptive_string(self, geo):
        s = geo.summary()
        assert isinstance(s, str)
        assert "水力直径" in s
        assert "13" in s  # n_rings
