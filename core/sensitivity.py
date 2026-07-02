"""
仿荷叶歧管微通道冷板 - 全局敏感性分析模块
=============================================

对冷板几何参数进行全局敏感性分析, 评估各参数对关键性能指标的影响:
1. Morris 方法 (简化自实现, 不依赖 SALib)
2. 简化版 Sobol 分析 (蒙特卡洛采样 + 方差分解)
3. 敏感性热力图 / 龙卷风图 / 双参数交互等高线图

待分析几何参数: channel_width, channel_height, fin_width,
                manifold_height, n_rings, inlet_diameter
每个参数在基准值 ±30% 范围内变化 (n_rings 为整数, 取整处理)。

性能指标: COP, thermal_resistance, pressure_drop, T_wall_max

参考:
- Morris MD, 1991, "Factorial sampling plans for preliminary computational experiments"
- Saltelli A, et al., 2010, "Variance based sensitivity analysis of model output"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.geometry import ManifoldRingChannelGeometry
from core.fluid_properties import FluidProperties
from core.single_phase import SinglePhaseSimulation
from core.two_phase import TwoPhaseSimulation

# 中文字体设置 (支持中文显示)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# ===== 参数与指标定义 =====
# 待分析几何参数: (参数名, 基准值, 是否整数)
_PARAM_SPECS = [
    ("channel_width",   0.3, False),
    ("channel_height",  1.4, False),
    ("fin_width",       0.3, False),
    ("manifold_height", 1.8, False),
    ("n_rings",         13,  True),   # 整数参数, 变化时取整
    ("inlet_diameter",  4.5, False),
]

# 性能指标: (结果对象属性名, 显示名)
_METRIC_SPECS = [
    ("COP",                "COP"),
    ("thermal_resistance", "Rth"),
    ("pressure_drop",      "ΔP"),
    ("T_wall_max",         "Tw_max"),
]

_PARAM_RANGE_FRAC = 0.30   # 参数变化范围 ±30%

# Morris 方法参数 (简化版: 每个参数生成多步轨迹)
_MORRIS_LEVELS = 4         # 单条轨迹沿目标参数的水平数 (点数)
_MORRIS_DELTA = 0.2        # 归一化步长 (覆盖整个参数范围的 20%)


@dataclass
class SensitivityResult:
    """敏感性分析结果数据类

    存储单个 参数-指标 对的敏感性分析结果。
    """
    param_name: str                                  # 参数名
    metric_name: str                                 # 性能指标显示名
    sensitivity_index: float = 0.0                   # 敏感性指数 (Morris μ* 或 Sobol 一阶 S_i)
    param_range: Tuple[float, float] = (0.0, 0.0)    # 参数取值范围 (min, max)
    baseline_value: float = 0.0                      # 参数基准值
    baseline_metric: float = 0.0                     # 基准工况下该指标值
    method: str = ""                                 # 分析方法: "morris" / "sobol"
    total_order_index: float = 0.0                   # Sobol 总效应指数 ST_i (仅 Sobol 方法有效)


class SensitivityAnalyzer:
    """全局敏感性分析器

    支持单相/两相仿真, 对冷板几何参数进行 Morris 与简化 Sobol 敏感性分析,
    并提供热力图、龙卷风图、双参数交互等高线图等可视化。

    Args:
        sim_type: 仿真类型, "single_phase" (单相水冷) 或 "two_phase" (两相 HFE-7100)
    """

    def __init__(self, sim_type: str = "single_phase"):
        """初始化敏感性分析器

        Args:
            sim_type: 仿真类型, "single_phase" (单相水冷) 或 "two_phase" (两相HFE-7100)
        """
        if sim_type not in ("single_phase", "two_phase"):
            raise ValueError(f"sim_type 必须为 'single_phase' 或 'two_phase', 得到: {sim_type}")
        self.sim_type = sim_type

        # 参数与指标配置
        self.param_names: List[str] = [s[0] for s in _PARAM_SPECS]
        self.param_baselines: Dict[str, float] = {s[0]: s[1] for s in _PARAM_SPECS}
        self.param_is_int: Dict[str, bool] = {s[0]: s[2] for s in _PARAM_SPECS}
        self.metric_attrs: List[str] = [s[0] for s in _METRIC_SPECS]
        self.metric_names: List[str] = [s[1] for s in _METRIC_SPECS]

        # 计算每个参数的取值范围 [min, max] (基准值 ±30%)
        self.param_ranges: Dict[str, Tuple[float, float]] = {}
        for name, base, _ in _PARAM_SPECS:
            lo = base * (1.0 - _PARAM_RANGE_FRAC)
            hi = base * (1.0 + _PARAM_RANGE_FRAC)
            if self.param_is_int[name]:
                lo = max(int(round(lo)), 1)
                hi = max(int(round(hi)), lo)
            self.param_ranges[name] = (lo, hi)

        # 工况设置 (与项目示例一致)
        if sim_type == "single_phase":
            self.fluid = FluidProperties("water")
            self.heat_flux = 100.0   # W/cm²
            self.mass_flow = 5.0     # g/s
        else:
            self.fluid = FluidProperties("HFE7100")
            self.heat_flux = 100.0
            self.mass_flow = 6.0

        # 基准工况指标值
        self.baseline_metrics = self._eval_metrics(self._baseline_param_dict())

        # 分析结果存储
        self.morris_results: List[SensitivityResult] = []
        self.sobol_results: List[SensitivityResult] = []
        self.morris_matrix: Optional[np.ndarray] = None       # (n_params, n_metrics) μ*
        self.sobol_matrix: Optional[np.ndarray] = None        # (n_params, n_metrics) S_i
        self.sobol_total_matrix: Optional[np.ndarray] = None  # (n_params, n_metrics) ST_i

        # 随机数生成器 (固定种子保证可复现)
        self.rng = np.random.default_rng(42)

    # ==================== 核心求值 ====================

    def _baseline_param_dict(self) -> Dict[str, float]:
        """返回基准参数字典"""
        return {name: self.param_baselines[name] for name in self.param_names}

    def _to_real(self, x_norm: np.ndarray) -> Dict[str, float]:
        """归一化向量 [0,1]^k → 实际参数字典 (整数参数取整)"""
        params: Dict[str, float] = {}
        for i, name in enumerate(self.param_names):
            lo, hi = self.param_ranges[name]
            val = lo + x_norm[i] * (hi - lo)
            if self.param_is_int[name]:
                val = int(round(val))
            params[name] = val
        return params

    def _eval_metrics(self, params: Dict[str, float]) -> np.ndarray:
        """给定几何参数字典, 返回 4 个性能指标 [COP, Rth, ΔP, Tw_max]"""
        geo = ManifoldRingChannelGeometry(**params)
        if self.sim_type == "single_phase":
            sim = SinglePhaseSimulation(geo, self.fluid)
        else:
            sim = TwoPhaseSimulation(geo, self.fluid)
        res = sim.simulate(heat_flux_Wcm2=self.heat_flux, mass_flow_gs=self.mass_flow)
        return np.array([getattr(res, attr) for attr in self.metric_attrs], dtype=float)

    def _safe_eval(self, x_norm: np.ndarray) -> np.ndarray:
        """带异常保护的求值 (归一化输入), 失败返回 NaN"""
        try:
            return self._eval_metrics(self._to_real(x_norm))
        except Exception:
            return np.full(len(self.metric_attrs), np.nan)

    def _eval_params_safe(self, params: Dict[str, float]) -> np.ndarray:
        """带异常保护的求值 (参数字典输入), 失败返回 NaN"""
        try:
            return self._eval_metrics(params)
        except Exception:
            return np.full(len(self.metric_attrs), np.nan)

    # ==================== Morris 方法 ====================

    def run_morris(self, n_trajectories: int = 10) -> List[SensitivityResult]:
        """运行 Morris 方法全局敏感性分析 (简化自实现)

        简化 Morris 实现 (不依赖 SALib):
            1. 每个参数归一化至 [0,1] (对应基准值 ±30% 范围)
            2. 对每个参数生成 n_trajectories 条轨迹, 每条轨迹:
               - 其它参数取随机基准点 (全局采样, 捕捉参数交互)
               - 目标参数从随机起始水平起, 逐步 +δ (基准值 → +δ → +2δ ...)
            3. 计算每次单步变化的基本效应 EE_i = (y_new - y_old) / δ
            4. μ*_i = mean(|EE_i|) 对所有轨迹与步数平均

        Args:
            n_trajectories: 每个参数的轨迹数, 默认 10

        Returns:
            List[SensitivityResult]: 每个参数-指标对的 μ* 结果
        """
        k = len(self.param_names)
        n_steps = _MORRIS_LEVELS - 1  # 单条轨迹的步数
        delta = _MORRIS_DELTA
        start_max = max(1.0 - n_steps * delta, 0.0)  # 起始水平上界 (保证最后不越界)

        # 累积每个 (参数, 指标) 的 |EE|
        ee_abs: Dict[str, Dict[str, List[float]]] = {
            name: {mn: [] for mn in self.metric_names} for name in self.param_names
        }

        for param_i, name in enumerate(self.param_names):
            for _ in range(n_trajectories):
                # 随机基准点 (所有参数在 [0,1] 内随机, 实现全局采样)
                x_norm = self.rng.random(k)
                # 目标参数起始水平限制在 [0, start_max]
                x_norm[param_i] = self.rng.random() * start_max

                y_prev = self._safe_eval(x_norm)

                # 沿目标参数逐步 +δ
                for _step in range(n_steps):
                    x_norm[param_i] += delta
                    y_new = self._safe_eval(x_norm)

                    if not (np.isnan(y_prev).any() or np.isnan(y_new).any()):
                        for m_idx, mn in enumerate(self.metric_names):
                            ee = (y_new[m_idx] - y_prev[m_idx]) / delta
                            ee_abs[name][mn].append(abs(ee))

                    y_prev = y_new

        # 计算 μ* 并构建结果矩阵
        M = len(self.metric_names)
        matrix = np.zeros((k, M))
        results: List[SensitivityResult] = []
        for i, pname in enumerate(self.param_names):
            for j, mn in enumerate(self.metric_names):
                vals = ee_abs[pname][mn]
                mu = float(np.mean(vals)) if len(vals) > 0 else 0.0
                matrix[i, j] = mu
                results.append(SensitivityResult(
                    param_name=pname,
                    metric_name=mn,
                    sensitivity_index=mu,
                    param_range=self.param_ranges[pname],
                    baseline_value=self.param_baselines[pname],
                    baseline_metric=float(self.baseline_metrics[j]),
                    method="morris",
                ))

        self.morris_results = results
        self.morris_matrix = matrix
        return results

    # ==================== 简化 Sobol 分析 ====================

    def run_sobol_simplified(self, n_samples: int = 100) -> List[SensitivityResult]:
        """简化版 Sobol 分析 (蒙特卡洛采样 + 方差分解)

        采用 Saltelli 采样 + Jansen 估计量:
            - 一阶指数  S_i  = (1/N) Σ y_B·(y_ABi - y_A) / V_Y
            - 总效应指数 ST_i = (1/(2N)) Σ (y_A - y_ABi)² / V_Y

        其中 A, B 为两个独立采样矩阵, y_ABi 为将 A 的第 i 列替换为 B 的第 i 列后的输出。
        总求值次数 ≈ N × (k + 2)。

        Args:
            n_samples: 基础样本数 N, 默认 100

        Returns:
            List[SensitivityResult]: 每个参数-指标对的一阶 S_i (含总效应 ST_i)
        """
        k = len(self.param_names)
        N = max(int(n_samples), 2)
        M = len(self.metric_names)

        # 两个独立采样矩阵 (归一化 [0,1], 对应参数 ±30% 范围)
        A = self.rng.random((N, k))
        B = self.rng.random((N, k))

        # 计算 y_A, y_B
        y_A = np.array([self._safe_eval(A[i]) for i in range(N)])  # (N, M)
        y_B = np.array([self._safe_eval(B[i]) for i in range(N)])

        S = np.zeros((k, M))
        ST = np.zeros((k, M))

        for i in range(k):
            # 将 A 的第 i 列替换为 B 的第 i 列
            AB_i = A.copy()
            AB_i[:, i] = B[:, i]
            y_ABi = np.array([self._safe_eval(AB_i[r]) for r in range(N)])  # (N, M)

            for j in range(M):
                yA = y_A[:, j]
                yB = y_B[:, j]
                yAB = y_ABi[:, j]
                mean_y = np.mean(yA)
                V_Y = np.mean(yA ** 2) - mean_y ** 2  # 输出方差
                if V_Y <= 1e-30:
                    continue
                # 一阶指数 (Saltelli 2010)
                S[i, j] = np.mean(yB * (yAB - yA)) / V_Y
                # 总效应指数 (Jansen 1999)
                ST[i, j] = np.mean((yA - yAB) ** 2) / (2.0 * V_Y)

        self.sobol_matrix = S
        self.sobol_total_matrix = ST

        results: List[SensitivityResult] = []
        for i, pname in enumerate(self.param_names):
            for j, mn in enumerate(self.metric_names):
                results.append(SensitivityResult(
                    param_name=pname,
                    metric_name=mn,
                    sensitivity_index=float(S[i, j]),
                    param_range=self.param_ranges[pname],
                    baseline_value=self.param_baselines[pname],
                    baseline_metric=float(self.baseline_metrics[j]),
                    method="sobol",
                    total_order_index=float(ST[i, j]),
                ))
        self.sobol_results = results
        return results

    # ==================== 绘图 ====================

    def _ensure_morris(self) -> np.ndarray:
        """确保 Morris 结果已计算, 返回 μ* 矩阵"""
        if self.morris_matrix is None:
            self.run_morris()
        return self.morris_matrix

    @staticmethod
    def _fmt_val(v: float) -> str:
        """数值格式化 (3 位有效数字, 极端值用科学计数法)"""
        if v == 0:
            return "0"
        return f"{v:.3g}"

    def plot_sensitivity_heatmap(self, save_path: Optional[str] = None) -> plt.Figure:
        """绘制参数 × 性能指标敏感性热力图

        - x 轴: 性能指标 (COP, Rth, ΔP, Tw_max)
        - y 轴: 几何参数 (channel_width, channel_height, fin_width,
                         manifold_height, n_rings, inlet_diameter)
        - 颜色: 归一化敏感性指数 (按列归一化至 0-1)
        - 每个格子标注 μ* 数值
        """
        matrix = self._ensure_morris()
        k, M = matrix.shape

        # 按列归一化 (每个指标列内最大值 = 1, 便于横向比较参数相对重要性)
        col_max = matrix.max(axis=0, keepdims=True)
        col_max_safe = np.where(col_max > 0, col_max, 1.0)
        norm_matrix = matrix / col_max_safe

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(norm_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

        # 坐标轴
        ax.set_xticks(np.arange(M))
        ax.set_yticks(np.arange(k))
        ax.set_xticklabels(self.metric_names, fontsize=11)
        ax.set_yticklabels(self.param_names, fontsize=11)
        ax.set_xlabel("性能指标", fontsize=12)
        ax.set_ylabel("几何参数", fontsize=12)

        # 每个格子标注原始 μ* 数值
        for i in range(k):
            for j in range(M):
                txt = self._fmt_val(matrix[i, j])
                color = "white" if norm_matrix[i, j] > 0.5 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        color=color, fontsize=9, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("归一化敏感性指数 (按列归一化)", fontsize=11)

        ax.set_title("全局敏感性热力图 (Morris μ*)", fontsize=13, fontweight='bold')
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
        return fig

    def plot_tornado(self, save_path: Optional[str] = None) -> plt.Figure:
        """绘制龙卷风图 (参数影响排序)

        对每个性能指标, 以水平条形图展示各参数 μ* 并按影响降序排列。
        """
        matrix = self._ensure_morris()
        k, M = matrix.shape

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        axes_flat = axes.flatten()
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, k))

        for j in range(M):
            ax = axes_flat[j]
            vals = matrix[:, j]
            # 降序排列
            order = np.argsort(vals)[::-1]
            sorted_vals = vals[order]
            sorted_names = [self.param_names[i] for i in order]

            bars = ax.barh(range(k), sorted_vals, color=colors,
                           edgecolor='black', linewidth=0.6)
            ax.set_yticks(range(k))
            ax.set_yticklabels(sorted_names, fontsize=10)
            ax.invert_yaxis()
            ax.set_xlabel("Morris μ*", fontsize=11)
            ax.set_title(f"{self.metric_names[j]} 影响排序", fontsize=12, fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3)

            # 条形末端标注数值
            for bar, v in zip(bars, sorted_vals):
                if v != 0:
                    ax.text(v, bar.get_y() + bar.get_height() / 2,
                            f" {self._fmt_val(v)}", va='center', fontsize=9)

        fig.suptitle("参数影响龙卷风图 (Morris μ*)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
        return fig

    def plot_interaction(self, param1: str, param2: str,
                         save_path: Optional[str] = None) -> plt.Figure:
        """绘制双参数交互等高线图

        对 4 个性能指标分别绘制 param1 × param2 的等高线图,
        其余参数保持基准值。

        Args:
            param1: x 轴参数名
            param2: y 轴参数名
            save_path: 图片保存路径, 为 None 时不保存
        """
        if param1 not in self.param_names:
            raise ValueError(f"param1 必须为 {self.param_names}, 得到: {param1}")
        if param2 not in self.param_names:
            raise ValueError(f"param2 必须为 {self.param_names}, 得到: {param2}")
        if param1 == param2:
            raise ValueError("两个参数不能相同")

        n_grid = 20
        p1_lo, p1_hi = self.param_ranges[param1]
        p2_lo, p2_hi = self.param_ranges[param2]
        p1_vals = np.linspace(p1_lo, p1_hi, n_grid)
        p2_vals = np.linspace(p2_lo, p2_hi, n_grid)
        P1, P2 = np.meshgrid(p1_vals, p2_vals)

        # 计算各指标网格
        M = len(self.metric_names)
        Z = [np.full_like(P1, np.nan, dtype=float) for _ in range(M)]
        for i in range(n_grid):
            for j in range(n_grid):
                params = self._baseline_param_dict()
                params[param1] = P1[i, j]
                params[param2] = P2[i, j]
                # 整数参数取整
                if self.param_is_int[param1]:
                    params[param1] = int(round(params[param1]))
                if self.param_is_int[param2]:
                    params[param2] = int(round(params[param2]))
                y = self._eval_params_safe(params)
                for m in range(M):
                    Z[m][i, j] = y[m]

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        axes_flat = axes.flatten()
        for m in range(M):
            ax = axes_flat[m]
            levels = 20
            cf = ax.contourf(P1, P2, Z[m], levels=levels, cmap='jet')
            ax.contour(P1, P2, Z[m], levels=levels, colors='k',
                       linewidths=0.4, alpha=0.5)
            plt.colorbar(cf, ax=ax, label=self.metric_names[m])
            ax.set_xlabel(f"{param1} [mm]" if not self.param_is_int[param1]
                          else f"{param1}", fontsize=11)
            ax.set_ylabel(f"{param2} [mm]" if not self.param_is_int[param2]
                          else f"{param2}", fontsize=11)
            ax.set_title(f"{self.metric_names[m]}", fontsize=12, fontweight='bold')
            # 标注基准点
            ax.scatter(self.param_baselines[param1], self.param_baselines[param2],
                       marker='x', color='white', s=100, linewidths=2,
                       zorder=5, label='基准')
            ax.legend(loc='upper right', fontsize=9)

        fig.suptitle(f"双参数交互等高线图: {param1} × {param2}",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
        return fig


if __name__ == "__main__":
    # 演示: 单相与两相敏感性分析
    os.makedirs("results", exist_ok=True)

    for st in ["single_phase", "two_phase"]:
        print(f"\n{'=' * 60}")
        print(f"{st} 全局敏感性分析 (Morris)")
        print(f"{'=' * 60}")

        analyzer = SensitivityAnalyzer(st)

        # 基准工况指标
        print(f"基准工况指标: "
              + ", ".join(f"{mn}={analyzer.baseline_metrics[m]:.4g}"
                          for m, mn in enumerate(analyzer.metric_names)))

        # Morris
        analyzer.run_morris(n_trajectories=10)

        # 打印 μ* 矩阵
        print("\nMorris μ* 矩阵 (行=参数, 列=指标):")
        header = f"{'参数':<18}" + "".join(f"{mn:>12}" for mn in analyzer.metric_names)
        print(header)
        for i, pname in enumerate(analyzer.param_names):
            row = f"{pname:<18}"
            for j in range(len(analyzer.metric_names)):
                row += f"{analyzer.morris_matrix[i, j]:>12.4g}"
            print(row)

        # 绘图
        analyzer.plot_sensitivity_heatmap(f"results/sensitivity_heatmap_{st}.png")
        analyzer.plot_tornado(f"results/sensitivity_tornado_{st}.png")
        analyzer.plot_interaction("channel_width", "channel_height",
                                  f"results/sensitivity_interaction_{st}.png")
        print(f"已保存敏感性分析图至 results/sensitivity_*_{st}.png")

        # 简化 Sobol (少量样本演示)
        analyzer.run_sobol_simplified(n_samples=50)
        print(f"\n简化 Sobol 一阶指数 S_i (n_samples=50):")
        print(header)
        for i, pname in enumerate(analyzer.param_names):
            row = f"{pname:<18}"
            for j in range(len(analyzer.metric_names)):
                row += f"{analyzer.sobol_matrix[i, j]:>12.4g}"
            print(row)

    print("\nDone!")
