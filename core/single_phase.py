"""
仿荷叶歧管微通道冷板 - 单相水冷仿真模型
=============================================
1D 分区模型 + 经验关联式
包含: 对流换热、压降、热阻、COP计算

换热模型:
- 层流入口段: Sieder-Tate 关联式
- 弯管二次流增强: Mori-Nakayama (1965) Dean涡修正
- 歧管折返: Borda-Carnot 局部阻力 + 冲击换热(隐含于Dean增强)
- 基板导热: 一维稳态热阻

模型输出 (工况: q=100 W/cm², m_dot=39 g/s, T_in=20°C, 水):
- h_conv ≈ 1.6 W/(cm²·K) (通道内壁局部对流换热系数, 基于湿周)
- h_overall ≈ 8.2 W/(cm²·K) (基于芯片投影面积的整体传热系数)
- Dean涡增强因子 ≈ 1.7x (Mori-Nakayama)
- ΔP ≈ 25 kPa (文献: 25.22 kPa)

注: 文献报告的 h=15.99 W/(cm²·K) 是基于投影面积, 包含二次流+冲击+混合等
多种增强效应的耦合作用, 当前1D模型仅计入Dean涡增强, 其余需CFD验证.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

from .geometry import ManifoldRingChannelGeometry
from .fluid_properties import FluidProperties


@dataclass
class SinglePhaseResult:
    """单相仿真结果数据类
    
    存储单相水冷仿真计算的所有输出指标，包含壁温、压力降、对流换热系数、热阻以及COP等。
    所有的温度单位为 °C，压力单位为 Pa，流量单位为 g/s，热流密度单位为 W/cm²。
    """
    heat_flux: float = 0.0           # 芯片表面受热热流密度 [W/cm²]
    mass_flow_rate: float = 0.0      # 进入冷板的总质量流量 [g/s]
    T_inlet: float = 25.0            # 冷却液入口温度 [°C]

    T_outlet: float = 0.0            # 冷却液出口温度 [°C]
    T_wall_max: float = 0.0          # 冷板底面壁面最高温度 [°C]
    T_wall_avg: float = 0.0          # 冷板底面壁面平均温度 [°C]
    delta_T_wall: float = 0.0        # 壁面最大温差（最高温度 - 最低温度），表征温度均匀性 [°C]
    h_conv: float = 0.0              # 平均对流换热系数 [W/(m²·K)]
    h_conv_cm2: float = 0.0          # 平均对流换热系数（厘米单位） [W/(cm²·K)]
    pressure_drop: float = 0.0       # 冷板总压力降 [Pa]
    thermal_resistance: float = 0.0  # 系统等效热阻 [(cm²·K)/W] (基于壁面最高温度与入口流体温差)
    COP: float = 0.0                 # 性能系数 (Coefficient of Performance), 定义为散热功率除以泵送功耗 [无量纲]
    pumping_power: float = 0.0       # 泵送功耗 [W] (pumping power = m_dot * delta_P / rho)
    Re: float = 0.0                  # 流道内部的平均雷诺数 (Reynolds Number)
    Nu: float = 0.0                  # 流道内部的平均努塞尔数 (Nusselt Number, 含Dean涡增强)
    Dean_avg: float = 0.0            # 平均 Dean 数 (Dean Number, 表征二次流强度)
    enhancement_avg: float = 1.0     # 平均换热增强因子 (Nu_curved / Nu_straight)
    Q_total: float = 0.0             # 总散热量 [W]
    G: float = 0.0                   # 流道内流体的质量流速 [kg/(m²·s)]


class SinglePhaseSimulation:
    """单相水冷仿真计算器
    
    基于一维分区网络模型，针对仿荷叶歧管环形微通道冷板的单相换热与流动特性进行计算。
    本模型结合经验关联式（如 Gnielinski 湍流换热、Shah & London 层流流动摩擦因子等）
    对多圈环形流道进行流量和热量加权求解。
    """

    def __init__(self,
                 geometry: ManifoldRingChannelGeometry = None,
                 fluid: FluidProperties = None):
        """初始化单相仿真计算器
        
        Args:
            geometry: 仿荷叶微通道冷板几何结构参数，默认为系统预设几何
            fluid: 冷却工质热物性模型，默认为水(water)
        """
        self.geo = geometry or ManifoldRingChannelGeometry()
        self.fluid = fluid or FluidProperties("water")

    def _calc_Re(self, G: float) -> float:
        """计算微通道内部雷诺数 (Reynolds Number)
        
        公式: Re = (G * Dh) / mu_l
        其中:
            G: 质量流速 [kg/(m²·s)]
            Dh: 通道水力直径 [m] (从毫米转换为米)
            mu_l: 液体动力粘度 [Pa·s]
            
        Args:
            G: 流道内质量流速 [kg/(m²·s)]
            
        Returns:
            Re: 雷诺数 [无量纲]
        """
        Dh = self.geo.hydraulic_diameter * 1e-3  # mm 转换为 m
        return G * Dh / self.fluid.mu_l

    def _calc_Dean(self, Re: float, r_bend: float) -> float:
        """计算 Dean 数 (Dean Number)
        
        Dean 数表征弯管/曲管中二次流(Dean涡)的强度:
        De = Re * sqrt(Dh / (2 * R_bend))
        
        当 De > 10 时, 二次流效应开始显著;
        当 De > 100 时, 二次流对换热的增强效果明显.
        
        参考文献: Dean (1927, 1928), Mori & Nakayama (1965)
        
        Args:
            Re: 雷诺数 [无量纲]
            r_bend: 弯曲半径 [m]
            
        Returns:
            De: Dean 数 [无量纲]
        """
        Dh = self.geo.hydraulic_diameter * 1e-3  # mm 转换为 m
        return Re * np.sqrt(Dh / max(2 * r_bend, Dh))

    def _calc_Nu_curved(self, Nu_straight: float, De: float, Pr: float) -> float:
        """计算弯管/环形通道二次流增强后的 Nusselt 数

        采用广泛引用的层流弯管换热关联式:
        
        层流 (De < 100):
            Nu_c / Nu_s = 1 + 0.014 · De · Pr^0.5
        该式见于 Mori & Nakayama (1965, 1967) 综述中的实用公式,
        也被 Kalb & Seader (1972), Dravid et al. (1971) 等微通道弯曲流文献引用.

        湍流/高De区 (De > 100):
            采用 Ito (1959) 形式的:
            Nu_c / Nu_s ≈ 1 + 0.11 · De^0.5

        参考文献:
        - Mori Y, Nakayama W. Int. J. Heat Mass Transfer, 8:67-82, 1965
        - Mori Y, Nakayama W. Int. J. Heat Mass Transfer, 10:681-695, 1967
        - Kalb C, Seader J. Int. J. Heat Mass Transfer, 15:801-817, 1972

        Args:
            Nu_straight: 直管(无曲率) Nusselt 数 [无量纲]
            De: Dean 数 [无量纲]
            Pr: 普朗特数 [无量纲]

        Returns:
            Nu_curved: 考虑二次流增强的 Nusselt 数 [无量纲]
        """
        if De < 10:
            return Nu_straight  # Dean 数太小, 二次流可忽略

        if De < 100:
            # 层流弯管: 标准 Mori-Nakayama 实用公式
            enhancement = 1 + 0.014 * De * np.sqrt(Pr)
        else:
            # 较高 Dean 数: Ito (1959) / Austin & Seader (1973) 形式
            enhancement = 1 + 0.11 * np.sqrt(De)

        return Nu_straight * enhancement

    def _calc_Nu(self, Re: float, Pr: float, L_over_Dh: float,
                 De: float = 0.0) -> float:
        """根据流动状态计算局部努塞尔数 (Nusselt Number)
        
        基于雷诺数区间划分为：
        1. 层流区间 (Re < 2300):
           采用等热流边界条件下矩形截面通道的充分发展层流常数 Nusselt = 8.235。
           针对入口段尚未完全发展的区域（若 Re * Pr / Gz_inv > 10），使用 Sieder-Tate 形式的入口段关联式：
           Nu = 1.86 * (Re * Pr / (L/Dh))^(1/3)，并设定下限不低于充分发展值 8.235。
        2. 过渡区间 (2300 <= Re < 4000):
           在层流极限值 (8.235) 与湍流极限值 (Gnielinski 公式算得) 之间按雷诺数进行线性插值。
           插值权重 x = (Re - 2300) / 1700。
        3. 湍流区间 (Re >= 4000):
           采用经典的 Gnielinski 关联式：
           Nu = (f / 8) * (Re - 1000) * Pr / (1 + 12.7 * (f / 8)^0.5 * (Pr^(2/3) - 1))
           其中摩擦因子 f 使用 Petukhov 关联式估算。为保证数值稳定，最终 Nu 设定下限不低于 8.235。
        
        在上述直管 Nu 的基础上, 若 De > 0, 叠加 Mori-Nakayama 弯管二次流增强修正.
           
        Args:
            Re: 雷诺数 [无量纲]
            Pr: 普朗特数 (Prandtl Number) [无量纲]
            L_over_Dh: 相对流动流程长度 (L / Dh) [无量纲]
            De: Dean 数 (Dean Number), 默认 0.0 (无曲率) [无量纲]
            
        Returns:
            Nu: 努塞尔数 [无量纲]
        """
        if Re < 2300:
            Nu_fd = 8.235  # 等热流(constant heat flux)矩形通道充分发展层流限值
            Gz_inv = max(L_over_Dh, 0.1)  # 逆葛拉兹数 (Inverse Graetz Number) 底限保护
            # 判断是否处于热入口段 (Thermal entrance region)
            if Re * Pr / Gz_inv > 10:
                # 考虑热入口段修正的 Sieder-Tate 关联式
                Nu = 1.86 * (Re * Pr / Gz_inv) ** (1 / 3)
                Nu = max(Nu, Nu_fd)  # 确保不低于充分发展值
            else:
                Nu = Nu_fd
        elif Re < 4000:
            # 过渡区: 采用 Gnielinski 公式得到的湍流值与层流值进行线性插值
            f = (0.790 * np.log(max(Re, 10)) - 1.64) ** (-2)  # Petukhov 摩擦因子
            Nu_turb = (f / 8) * (Re - 1000) * Pr / (1 + 12.7 * (f / 8) ** 0.5 * (Pr ** (2 / 3) - 1))
            Nu_lam = 8.235
            x = (Re - 2300) / 1700  # 插值系数 [0, 1]
            Nu = Nu_lam + x * (Nu_turb - Nu_lam)
        else:
            # 湍流区: Gnielinski 关联式，适用于宽广的雷诺数和普朗特数范围 (0.5 < Pr < 2000, 3000 < Re < 5*10^6)
            f = (0.790 * np.log(max(Re, 10)) - 1.64) ** (-2)  # Petukhov 摩擦因子
            Nu = (f / 8) * (Re - 1000) * Pr / (1 + 12.7 * (f / 8) ** 0.5 * (Pr ** (2 / 3) - 1))
            Nu = max(Nu, 8.235)  # 边界安全保护

        # 叠加弯管/环形通道二次流 (Dean涡) 增强修正
        if De > 10:
            Nu = self._calc_Nu_curved(Nu, De, Pr)

        return Nu

    def _calc_friction_factor(self, Re: float) -> float:
        """计算单相流动的达西摩擦因子 (Darcy Friction Factor)
        
        基于雷诺数区间：
        1. 层流区间 (Re < 2300):
           针对非圆形矩形通道，需要基于纵横比 (Aspect Ratio, alpha) 进行 Shah & London 修正。
           纵横比 alpha 定义为 宽/高 (或 高/宽，取较小值，限制在 [0, 1] 区间)。
           f_lam = 24 * (1 - 1.3553*alpha + 1.9467*alpha^2 - 1.7012*alpha^3 + 0.9564*alpha^4 - 0.2537*alpha^5)。
           返回摩擦因子 f = f_lam / Re。
        2. 过渡及湍流区间 (Re >= 2300):
           采用 Petukhov 光滑管经验关联式: f = (0.790 * ln(Re) - 1.64)^(-2)。
           
        Args:
            Re: 雷诺数 [无量纲]
            
        Returns:
            f: 达西摩擦因子 [无量纲]
        """
        if Re < 2300:
            # 矩形截面修正，取通道宽与高之比的倒数，使得纵横比 alpha <= 1.0
            alpha = self.geo.channel_width / self.geo.channel_height
            alpha = min(alpha, 1.0 / alpha) if alpha > 0 else 0.5
            # Shah & London 矩形通道层流阻力多项式系数拟合
            f_lam = 24 * (1 - 1.3553 * alpha + 1.9467 * alpha ** 2
                          - 1.7012 * alpha ** 3 + 0.9564 * alpha ** 4
                          - 0.2537 * alpha ** 5)
            return f_lam / max(Re, 1)  # 避免除以零
        else:
            # Petukhov 湍流摩擦因子关联式
            return (0.790 * np.log(max(Re, 10)) - 1.64) ** (-2)

    def simulate(self,
                 heat_flux_Wcm2: float = 100.0,
                 mass_flow_gs: float = 39.0,
                 T_inlet: float = 20.0) -> SinglePhaseResult:
        """执行仿荷叶冷板的一维单相分区流动与换热仿真
        
        计算逻辑分为以下几个阶段：
        1. 单位转换与热力学基础计算：计算总散热功率 Q_total，工质总体温升等。
        2. 流量与面积配比分配：根据冷板内多环并联的特点，按半径占比分配热流量和换热面积。
        3. 沿程及局部阻力模型：
           - 计算中心入口收缩压降
           - 计算进、出液歧管狭缝(Slots)的质量流速和压降
        4. 环形分区迭代计算循环：对于每一圈环形流道，单独计算：
           - 单路径相对长度 L/Dh 对应的努塞尔数 Nu_i
           - 计算该环的对流换热系数 h_i
           - 分配环的受热功率 Q_i 得到局部工质平均温升，并通过三阻力叠加(对流热阻+铜板导热热阻)算出底面局部壁温
           - 计算该圈通道摩擦压降与折返、分支带来的局部压降损失 (K_local_i = 2.5 + 0.2 * n_sectors)
        5. 全局指标聚合与汇总：进行加权平均计算，返回最终 SinglePhaseResult 结果。
        
        Args:
            heat_flux_Wcm2: 芯片表面受热热流密度 [W/cm²]，默认 100.0 W/cm²
            mass_flow_gs: 进入冷板的总流量 [g/s]，默认 39.0 g/s
            T_inlet: 冷却液入口温度 [°C]，默认 20.0 °C
            
        Returns:
            res: SinglePhaseResult 对象，包含冷板的全部流动与换热热工指标。
        """
        res = SinglePhaseResult()
        res.heat_flux = heat_flux_Wcm2
        res.mass_flow_rate = mass_flow_gs
        res.T_inlet = T_inlet
        
        # 1. 基础物理单位转换
        # 芯片受热面积: mm² 转换为 m²
        A_chip = self.geo.chip_area * 1e-6 
        # 总散热量 [W] = 热流密度 [W/cm²] * 10^4 [cm²/m²] * 芯片面积 [m²]
        Q_total = heat_flux_Wcm2 * 1e4 * A_chip
        res.Q_total = Q_total
        # 质量流量: g/s 转换为 kg/s
        m_dot = mass_flow_gs * 1e-3
        # 通道水力直径: mm 转换为 m
        Dh = self.geo.hydraulic_diameter * 1e-3

        # 流量或热量不合理时提前退出
        if m_dot <= 0 or Q_total <= 0:
            return res

        # 工质理论总出口温升: delta_T = Q / (m_dot * Cp)
        delta_T_fluid = Q_total / (m_dot * self.fluid.cp_l)
        res.T_outlet = T_inlet + delta_T_fluid

        # 获取各环的径向参数和流动流程长度
        ring_radii = np.asarray(self.geo.ring_radii, dtype=float)
        L_flow_rings = np.asarray(self.geo.L_flow_rings, dtype=float) * 1e-3  # mm 转换为 m
        n_rings = max(len(ring_radii), 1)
        n_sectors = max(self.geo.n_sectors, 1)

        # 2. 局部受热量和换热面积分配权重 (正比于环的半径)
        area_weights = ring_radii / max(np.sum(ring_radii), 1e-12)
        Q_rings = Q_total * area_weights
        A_ht_rings = self.geo.total_heat_transfer_area * area_weights
        
        # 并联流道的流量拆分:
        # 每个圈(ring)分配的流量
        m_dot_ring = m_dot / n_rings
        # 环内每个扇形流段(path)分配的流量
        m_dot_path = m_dot_ring / n_sectors

        # 3. 计算单条通道(Path)内的流动参数 (由于通道截面积统一，所有环中单通道内的流速和雷诺数一致)
        # 单流道截面积: mm² 转换为 m²
        A_channel = self.geo.channel_cross_area * 1e-6
        # 流道内流体质量流速 [kg/(m²·s)]
        G_path = m_dot_path / max(A_channel, 1e-12)
        # 计算雷诺数、摩擦因子和流体流动动压 (dyn_p = G^2 / (2*rho))
        Re_path = self._calc_Re(G_path)
        f_path = self._calc_friction_factor(Re_path)
        dyn_p = G_path ** 2 / (2 * self.fluid.rho_l)

        # 4. 歧管与进出口段局部阻力计算 (Borda-Carnot 理论)
        # 中心入口孔截面积 [m²]
        A_inlet = np.pi * (self.geo.inlet_diameter * 1e-3) ** 2 / 4
        # 歧管缝隙数量(默认扇区的一半为进水口)
        n_inlet_slots = max(self.geo.n_sectors // 2, 1)
        # 窄进液歧管缝总截面积 [m²]
        A_inlet_slots = n_inlet_slots * self.geo.inlet_slot_width * self.geo.manifold_height * 1e-6
        # 宽出液歧管缝总截面积 [m²]
        A_outlet_slots = n_inlet_slots * self.geo.outlet_slot_width * self.geo.manifold_height * 1e-6

        # 对应流速计算
        G_inlet = m_dot / max(A_inlet, 1e-12)
        G_inlet_slots = m_dot / max(A_inlet_slots, 1e-12)
        G_outlet_slots = m_dot / max(A_outlet_slots, 1e-12)

        # 阻力损失计算 (Borda-Carnot 突缩/突扩 + 转弯/分流损失):
        # 突缩: K_cont = 0.5 * (1 - sigma), sigma = A2/A1 (小/大)
        # 突扩: K_exp = (1 - sigma)^2, sigma = A1/A2 (小/大)
        # 90°弯头: K_bend = 1.1 (Idelchik)
        # 分配损失: K_branch = 0.5 (流体分配到多通道)

        # 中心入口 → 歧管分配: 突缩 + 90°转弯 + 分流
        sigma_inlet = min(A_inlet_slots / max(A_inlet, 1e-12), 1.0)
        K_inlet = 0.5 * (1 - sigma_inlet) + 1.1 + 0.5  # 突缩 + 转弯 + 分流
        delta_P_inlet = K_inlet * G_inlet ** 2 / (2 * self.fluid.rho_l)

        # 进液歧管缝 → 微通道: 突缩 + 90°转弯
        A_micro_total = self.geo.effective_cross_area
        sigma_slot_in = min(A_micro_total / max(A_inlet_slots, 1e-12), 1.0)
        K_slot_in = 0.5 * (1 - sigma_slot_in) + 1.1  # 突缩 + 转弯

        # 微通道 → 出液歧管缝: 突扩 + 90°转弯
        sigma_slot_out = min(A_outlet_slots / max(A_micro_total, 1e-12), 1.0)
        K_slot_out = (1 - sigma_slot_out) ** 2 + 1.1  # 突扩 + 转弯

        delta_P_slots = (
            K_slot_in * G_inlet_slots ** 2 / (2 * self.fluid.rho_l)
            + K_slot_out * G_outlet_slots ** 2 / (2 * self.fluid.rho_l)
        )

        T_wall_rings = []
        h_rings = []
        Nu_rings = []
        Nu_straight_rings = []  # 无Dean增强的直管Nu, 用于计算增强因子
        Dean_rings = []
        pressure_rings = []
        T_out_rings = []

        # 固体导热计算参数
        t_base = self.geo.base_thickness * 1e-3  # 基板厚度 mm 转换为 m
        k_base = self.geo.k_substrate            # 基材(铜)导热系数 [W/(m·K)]

        # 5. 循环迭代每一圈环形流道
        for idx, (Q_i, A_i, L_i) in enumerate(zip(Q_rings, A_ht_rings, L_flow_rings)):
            # 无量纲流动长度
            L_over_Dh = L_i / max(Dh, 1e-12)
            # 该环的弯曲半径 (环形通道中心线半径) [m]
            r_bend_i = ring_radii[idx] * 1e-3
            # Dean 数: 表征环形通道曲率引起的二次流强度
            De_i = self._calc_Dean(Re_path, r_bend_i)
            # 计算该环局部 Nu 数 (含入口段 + Dean涡增强) 与对流换热系数 h_i
            Nu_i = self._calc_Nu(Re_path, self.fluid.Pr_l, L_over_Dh, De=De_i)
            h_i = Nu_i * self.fluid.k_l / max(Dh, 1e-12)
            # 同时计算无增强的直管Nu, 用于输出增强因子
            Nu_straight_i = self._calc_Nu(Re_path, self.fluid.Pr_l, L_over_Dh, De=0.0)

            # 该环工质流体的局部温升 dT_i (Q_i / (m_dot_ring * Cp))
            dT_i = Q_i / max(m_dot_ring * self.fluid.cp_l, 1e-12)
            # 局部流体平均温度取进出口中点温度
            T_fluid_avg_i = T_inlet + 0.5 * dT_i

            # 壁面热流密度及基板导热热阻计算
            q_wall_i = Q_i / max(A_i, 1e-12)
            R_cond_i = t_base / max(k_base * A_i, 1e-12)
            # 底面壁温由三部分组成：工质平均温 + 对流温差 (q / h) + 铜板导热温差 (Q * R_cond)
            T_wall_i = T_fluid_avg_i + q_wall_i / max(h_i, 1e-12) + Q_i * R_cond_i

            # 流动压降：
            # 沿程摩擦阻力损失
            delta_P_friction_i = f_path * L_over_Dh * dyn_p
            # 分流折返局部阻力 (Borda-Carnot + 90°弯头理论)
            # 每条路径有2次90°弯头(进液转弯+出液转弯)
            # K_bend = 1.1 per 90° bend (Idelchik)
            # + 收缩/扩张损失已在 delta_P_slots 中计入
            K_bend = 2 * 1.1  # 2次90°弯头
            # Dean 数修正 (曲率效应): 弯头处二次流增强混合
            if De_i > 10:
                K_bend *= (1 + 0.15 * np.log10(De_i / 10))
            delta_P_local_i = K_bend * dyn_p

            # 存储该环计算数据
            T_wall_rings.append(T_wall_i)
            h_rings.append(h_i)
            Nu_rings.append(Nu_i)
            Nu_straight_rings.append(Nu_straight_i)
            Dean_rings.append(De_i)
            pressure_rings.append(delta_P_friction_i + delta_P_local_i)
            T_out_rings.append(T_inlet + dT_i)

        # 转换为 NumPy 数组方便统计
        T_wall_rings = np.asarray(T_wall_rings)
        h_rings = np.asarray(h_rings)
        Nu_rings = np.asarray(Nu_rings)
        Nu_straight_rings = np.asarray(Nu_straight_rings)
        Dean_rings = np.asarray(Dean_rings)
        pressure_rings = np.asarray(pressure_rings)
        T_out_rings = np.asarray(T_out_rings)

        # 6. 全局物理量聚合汇总 (将局部环的结果按照受热面积/流量进行加权平均)
        res.G = G_path
        res.Re = Re_path
        res.Nu = float(np.average(Nu_rings, weights=area_weights))
        res.Dean_avg = float(np.average(Dean_rings, weights=area_weights))
        Nu_straight_avg = float(np.average(Nu_straight_rings, weights=area_weights))
        res.enhancement_avg = res.Nu / max(Nu_straight_avg, 1e-6)
        res.h_conv = float(np.average(h_rings, weights=area_weights))
        res.h_conv_cm2 = res.h_conv * 1e-4  # 转换为 W/(cm²·K)
        # 流体实际混合出口温度
        res.T_outlet = float(np.average(T_out_rings, weights=np.full(n_rings, m_dot_ring)))
        res.T_wall_avg = float(np.average(T_wall_rings, weights=area_weights))
        res.T_wall_max = float(np.max(T_wall_rings))
        # 最大壁面温差（评估温度均匀性）
        res.delta_T_wall = float(res.T_wall_max - np.min(T_wall_rings))
        
        # 冷板总压降 = 平均多环压降 + 入口压降 + 歧管缝隙压降
        res.pressure_drop = float(np.average(pressure_rings, weights=area_weights)
                                  + delta_P_inlet + delta_P_slots)
        # 泵送功率 W_pump = (m_dot * delta_P) / rho
        res.pumping_power = m_dot * res.pressure_drop / self.fluid.rho_l

        # 等效热阻 R_th = (T_wall_max - T_inlet) / Q_total [K/W]
        # 再乘以 A_chip * 10^4 转换为工程常用单位 [(cm²·K)/W]
        R_total = (res.T_wall_max - T_inlet) / max(Q_total, 1e-6)
        res.thermal_resistance = R_total * A_chip * 1e4
        
        # 性能系数 COP = 总换热功率 / 泵送消耗功率
        res.COP = Q_total / max(res.pumping_power, 1e-10)

        return res

    def parametric_sweep(self,
                         heat_flux_range: np.ndarray = None,
                         flow_rate_range: np.ndarray = None,
                         T_inlet: float = 25.0) -> Dict:
        """对热流密度和流量进行双参数扫描
        
        Args:
            heat_flux_range: 一维热流密度扫描数组 [W/cm²]，默认在 [50, 633] 内生成 30 个点
            flow_rate_range: 一维质量流量扫描数组 [g/s]，默认在 [1, 20] 内生成 20 个点
            T_inlet: 入口冷却液温度 [°C]，默认 25.0 °C
            
        Returns:
            Dict: 包含 "heat_flux_range", "flow_rate_range" 和 "results_matrix" (二维仿真结果矩阵) 的字典
        """
        if heat_flux_range is None:
            heat_flux_range = np.linspace(50, 633, 30)
        if flow_rate_range is None:
            flow_rate_range = np.linspace(1, 20, 20)

        results = np.empty((len(heat_flux_range), len(flow_rate_range)), dtype=object)
        for i, qf in enumerate(heat_flux_range):
            for j, mf in enumerate(flow_rate_range):
                results[i, j] = self.simulate(qf, mf, T_inlet)

        return {
            "heat_flux_range": heat_flux_range,
            "flow_rate_range": flow_rate_range,
            "results_matrix": results,
        }
