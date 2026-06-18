# 更新日志 (Changelog)

所有该项目的版本变动及优化记录都将记录在此文件中。

## [Unreleased] - 2026-06-18

### 新增 (Added)
- 新增详细的单相水冷模型分区仿真物理计算逻辑与温升压降阻力网络分析文档：[walkthrough.md](file:///C:/Users/20163751/.gemini/antigravity/brain/72b44ea8-34e2-433f-8339-d1843f5afdf7/walkthrough.md)。

### 优化 (Improved)
- **代码可读性提升**：对单相水冷仿真计算器 [single_phase.py](file:///d:/Python/lotus_mrc_coldplate/core/single_phase.py) 进行了重构和代码整理，添加了完全符合规范的中文 Docstrings 和详实的计算公式物理注释。
- **关联式详注**：
  - 详细注释了雷诺数 $Re$ 的定义与计算。
  - 详细注释了 Nusselt 数 $Nu$ 的区间划分计算机制，包括层流常数限值与入口段 Sieder-Tate 修正、湍流段 Gnielinski 公式和过渡段线性插值。
  - 详细注释了摩擦因子 $f$ 在层流区 Shah & London 矩形管道多项式修正和湍流区 Petukhov 公式。
  - 阐明了基于三热阻叠加模型（工质温升＋对流热阻＋底板导热）的局部壁面最高温度求解逻辑。
- **数据结构规范化**：为 [SinglePhaseResult](file:///d:/Python/lotus_mrc_coldplate/core/single_phase.py#L25) 数据类中的各物理量补充了标准公制单位（如 W/cm²、g/s、Pa、K/W）注释。
