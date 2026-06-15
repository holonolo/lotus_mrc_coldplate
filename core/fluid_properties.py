"""
仿荷叶歧管微通道冷板 - 工质物性库
=============================================
支持: 水, HFE-7100, R245fa, R1233zd(E)
基于 CoolProp 计算热物性, 带 fallback 表格
"""

import numpy as np
from typing import Optional

try:
    import CoolProp
    from CoolProp.CoolProp import PropsSI
    HAS_COOLPROP = True
except ImportError:
    HAS_COOLPROP = False
    print("[警告] CoolProp 未安装, 使用内置近似物性表")


class FluidProperties:
    """工质热物性计算"""

    # ===== 内置近似物性 (25°C, 1atm) =====
    _TABLES = {
        "water": {
            "T_sat": 100.0,       # °C @ 1atm
            "rho_l": 997.0,       # kg/m³
            "rho_v": 0.6,        # kg/m³
            "cp_l": 4182.0,      # J/(kg·K)
            "cp_v": 2030.0,      # J/(kg·K)
            "k_l": 0.607,        # W/(m·K)
            "k_v": 0.025,        # W/(m·K)
            "mu_l": 8.9e-4,      # Pa·s
            "mu_v": 1.2e-5,      # Pa·s
            "h_fg": 2257e3,      # J/kg
            "sigma": 0.072,      # N/m
            "Pr_l": 6.13,
        },
        "HFE7100": {
            "T_sat": 61.0,
            "rho_l": 1510.0,
            "rho_v": 10.0,
            "cp_l": 1183.0,
            "cp_v": 850.0,
            "k_l": 0.069,
            "k_v": 0.014,
            "mu_l": 5.8e-4,
            "mu_v": 1.0e-5,
            "h_fg": 112e3,
            "sigma": 0.0136,
            "Pr_l": 9.9,
        },
        "R245fa": {
            "T_sat": 15.1,
            "rho_l": 1330.0,
            "rho_v": 8.0,
            "cp_l": 1300.0,
            "cp_v": 940.0,
            "k_l": 0.087,
            "k_v": 0.014,
            "mu_l": 3.7e-4,
            "mu_v": 1.0e-5,
            "h_fg": 196e3,
            "sigma": 0.015,
            "Pr_l": 5.5,
        },
        "R1233zdE": {
            "T_sat": 18.3,
            "rho_l": 1250.0,
            "rho_v": 7.0,
            "cp_l": 1200.0,
            "cp_v": 880.0,
            "k_l": 0.082,
            "k_v": 0.013,
            "mu_l": 3.8e-4,
            "mu_v": 9.5e-6,
            "h_fg": 195e3,
            "sigma": 0.015,
            "Pr_l": 5.6,
        },
    }

    # CoolProp 工质名映射
    _CP_NAMES = {
        "water": "Water",
        "HFE7100": "HFE7100",
        "R245fa": "R245fa",
        "R1233zdE": "R1233zd(E)",
    }

    def __init__(self, fluid_name: str = "water"):
        self.fluid_name = fluid_name
        self._props = self._TABLES.get(fluid_name, self._TABLES["water"])

    def get(self, prop: str, T: Optional[float] = None, P: Optional[float] = None):
        """获取物性值, 优先 CoolProp, 否则用表格"""
        if HAS_COOLPROP and T is not None:
            try:
                cp_name = self._CP_NAMES.get(self.fluid_name)
                if cp_name is None:
                    return self._props[prop]
                if prop == "T_sat":
                    return PropsSI("T", "P", P or 101325, "Q", 0, cp_name) - 273.15
                T_K = T + 273.15
                P_Pa = P or 101325
                mapping = {
                    "rho_l": ("D", "Q", 0), "rho_v": ("D", "Q", 1),
                    "cp_l": ("C", "Q", 0), "cp_v": ("C", "Q", 1),
                    "k_l": ("L", "Q", 0), "k_v": ("L", "Q", 1),
                    "mu_l": ("V", "Q", 0), "mu_v": ("V", "Q", 1),
                    "h_fg": ("H", "Q", 1),  # 需要差值
                    "sigma": ("I", "Q", 0),  # surface_tension
                }
                if prop in mapping:
                    cp_prop, quality_q, q_val = mapping[prop]
                    if prop == "h_fg":
                        h_l = PropsSI("H", "T", T_K, "Q", 0, cp_name)
                        h_v = PropsSI("H", "T", T_K, "Q", 1, cp_name)
                        return h_v - h_l
                    if prop == "sigma":
                        return PropsSI("surface_tension", "T", T_K, "Q", 0, cp_name)
                    return PropsSI(cp_prop, "T", T_K, "Q", q_val, cp_name)
            except Exception:
                pass
        return self._props[prop]

    @property
    def T_sat(self): return self._props["T_sat"]
    @property
    def rho_l(self): return self._props["rho_l"]
    @property
    def rho_v(self): return self._props["rho_v"]
    @property
    def cp_l(self): return self._props["cp_l"]
    @property
    def cp_v(self): return self._props["cp_v"]
    @property
    def k_l(self): return self._props["k_l"]
    @property
    def k_v(self): return self._props["k_v"]
    @property
    def mu_l(self): return self._props["mu_l"]
    @property
    def mu_v(self): return self._props["mu_v"]
    @property
    def h_fg(self): return self._props["h_fg"]
    @property
    def sigma(self): return self._props["sigma"]
    @property
    def Pr_l(self): return self._props["Pr_l"]


if __name__ == "__main__":
    for name in ["water", "HFE7100", "R245fa", "R1233zdE"]:
        fp = FluidProperties(name)
        print(f"\n--- {name} ---")
        print(f"  T_sat = {fp.T_sat}°C")
        print(f"  rho_l = {fp.rho_l} kg/m³")
        print(f"  h_fg  = {fp.h_fg/1e3:.1f} kJ/kg")
        print(f"  k_l   = {fp.k_l} W/(m·K)")
        print(f"  mu_l  = {fp.mu_l:.2e} Pa·s")
