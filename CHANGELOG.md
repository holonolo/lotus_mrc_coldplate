# 更新日志 (Changelog)

所有该项目的版本变动及优化记录都将记录在此文件中。

## [Unreleased] - 2026-06-19

### 重大重构 (Major Refactoring)
- **两相模型重写**：[two_phase.py](core/two_phase.py) 完整重写为基于物理机理建模的**第一性原理版**，彻底移除全局标定系统。
  - 核心换热公式从 Kandlikar (2004) 替换为 **Gungor-Winterton (1987)** 关联式，对低导热介电流体（HFE-7100）更稳健。
  - 几何增强因子 η_geometry 从歧管面积比、Borda-Carnot 收缩/扩张损失及 Dean 二次流理论推导，替代原硬编码常数 1.5632。
  - 流型判定使用无量纲数（Co, We, X_tt）动态映射，替代固定干度阈值（0.05, 0.15, 0.5）。
  - CHF 模型基于 Kutateladze-Zuber 空泡脱离准则 + Yu 微通道尺寸修正。

### 新增 (Added)
- 新增 [test_new_model.py](test_new_model.py) 新模型验证测试脚本。
- 新增 [core/two_phase_v2.py](core/two_phase_v2.py) 修正实验版（增强因子以壁温修正形式独立处理）。

### 优化 (Improved)
- 换热面积基准修正：h 输出同时提供基于**投影面积**和**湿面积**两个基准的值，明确物理定义。
- **README.md** 更新：补充两相模型物理模型说明、基准说明和对比验证数据。
- **文档完善**：更新项目结构描述以反映新增文件。

### 修复 (Fixed)
- 修复 h_proj 公式方向错误（原为 h_proj = h_wet / area_ratio，修正为 h_proj = h_wet × area_ratio），使其符合 Q = h_wet × A_wet × ΔT = h_proj × A_proj × ΔT 的物理守恒关系。
- 确认 two_phase_v2.py 中 Reynold 数 Re_D 的计算使用 mu_l（动力粘度）而非 rho_l（密度）。

### 验证 (Validation)
- 在文献基准工况（q=255 W/cm², m=6 g/s, HFE-7100）下完成对标：
  - 湿面积基准换热系数：预测 2.18 vs 文献 2.13 W/(cm²·K)，误差 +2.5%
  - 壁温-饱和温差 21.8K，满足 < 60K 要求
  - 均无需全局标定，纯物理预测

## [Unreleased] - 2026-06-18

### 新增 (Added)
- 新增详细的单相水冷模型分区仿真物理计算逻辑与温升压降阻力网络分析文档。

### 优化 (Improved)
- **代码可读性提升**：对 [single_phase.py](core/single_phase.py) 进行了重构和代码整理，添加了符合规范的 Docstrings 和物理注释。
- **关联式详注**：详细注释了 Re、Nu、摩擦因子 f 的定义与计算区间划分。
- **数据结构规范化**：为 SinglePhaseResult 数据类中的各物理量补充了标准公制单位注释。