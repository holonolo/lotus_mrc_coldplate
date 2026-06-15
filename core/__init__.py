"""
仿荷叶歧管微通道冷板 - 核心模块初始化
"""

from .geometry import ManifoldRingChannelGeometry
from .fluid_properties import FluidProperties
from .single_phase import SinglePhaseSimulation, SinglePhaseResult
from .two_phase import TwoPhaseSimulation, TwoPhaseResult, FlowPattern
from .comparison import ComparativeAnalysis, ComparisonPoint
from .visualization import (
    plot_geometry_topview,
    plot_geometry_crosssection,
    plot_comparison_curves,
    plot_boiling_curve,
    plot_sensitivity_analysis,
)

__all__ = [
    "ManifoldRingChannelGeometry",
    "FluidProperties",
    "SinglePhaseSimulation",
    "SinglePhaseResult",
    "TwoPhaseSimulation",
    "TwoPhaseResult",
    "FlowPattern",
    "ComparativeAnalysis",
    "ComparisonPoint",
    "plot_geometry_topview",
    "plot_geometry_crosssection",
    "plot_comparison_curves",
    "plot_boiling_curve",
    "plot_sensitivity_analysis",
]
