# 🍃 仿荷叶歧管微通道冷板仿真与对比分析工具

本项目是一个用于设计和评估**仿荷叶歧管环形微通道冷板**（Manifold Ring Channel Cold Plate）热工水力学性能的仿真计算工具。程序提供单相水冷与两相沸腾冷却（工质包括 HFE-7100、R245fa、R1233zdE 及水）的对比分析，并支持图形化交互和命令行两种运行模式。

该模型基于**浙江大学吴赞课题组**的相关学术文献进行标定与开发。

---

## 📖 学术背景与参考文献
本冷板模型和仿真算法参考自以下学术成果：
1. **单相水冷模型**：Xin Z, et al. *Energy*, 2025.
2. **两相沸腾模型**：Xin Z, et al. *Energy Conversion and Management*, 2026.
3. **结构优势**：中心入口射流冲击 + 环形通道径向多通道分流，相比传统平行微通道可降低 **50.72%** 的压降，壁温均匀性提升 **43.74%**。

---

## ✨ 核心功能
* **📐 几何结构参数化建模**：可配置芯片面积、通道宽深、翅片厚度、歧管高度和环形歧管层数，支持实时绘制冷板俯视图和截面结构图。
* **💧 单相水冷模拟**：模拟不同热流密度与质量流量下的单相对流换热系数、泵功、最高壁温、热阻和 COP，支持参数敏感性扫描。
* **🔥 两相沸腾模拟（Gungor-Winterton 第一性原理模型）**：
  - 采用 **Gungor-Winterton (1987)** 两相关联式（替代原先的标定版 Kandlikar 模型）
  - **无全局标定系数**：所有物理量由流体物性和几何参数推导得出
  - **几何增强因子**：基于歧管面积比、扩张角、Dean 二次流理论计算，无硬编码常数
  - **流型判定**：使用 Lockhart-Martinelli 参数 X_tt 和对流数 Co 动态映射，替代固定干度阈值
  - **CHF 预测**：Kutateladze-Zuber 空泡脱离准则 + Yu 微通道尺寸修正
  - **压降计算**：Lockhart-Martinelli 模型（摩擦 + 加速度 + 重力分量）
* **📊 对比分析报告**：提供单相水冷与两相沸腾冷却的关键指标（换热系数、压降、热阻、COP）对比，自动生成文本报告与对比曲线。
* **💻 友好运行方式**：
  * **命令行接口（CLI）**：适合快速计算和批量保存图片。
  * **网页交互界面（GUI）**：基于 Streamlit 构建，参数调节直观，图表自适应缩放。

---

## 🛠️ 环境依赖
本项目完全支持**无网/内网环境**本地离线运行。所需依赖包如下：
* Python 3.7+
* `numpy`
* `scipy`
* `matplotlib`
* `CoolProp` (用于高精度流体物性查询)
* `streamlit` (用于 GUI 界面)

---

## 🚀 快速开始

### 方式一：运行网页交互界面 (GUI)

#### 1. 一键直接运行（VS Code 友好）
如果您在 VS Code 中打开本项目，可以直接点击编辑器右上角的 **"运行"按钮 (Play 按钮)** 运行 `gui_app.py`。
> **原理**：我们在 `gui_app.py` 底部集成了自动引导程序（Bootstrapper）并关闭了外网遥测机制，它会自动调用本地 Streamlit 服务拉起浏览器。

#### 2. 通过命令行运行
您也可以在终端中执行以下标准命令启动：
```bash
streamlit run gui_app.py
```
启动后，浏览器会自动打开 `http://localhost:8501` 展示交互界面。

---

### 方式二：运行命令行仿真 (CLI)
通过运行 `main.py` 进行快速计算，并在控制台直接输出对比分析报告。

* **运行默认工况**：
  ```bash
  python main.py
  ```
* **指定热流密度与流量运行**：
  ```bash
  python main.py --heat-flux 200 --flow-rate 8.0
  ```
* **生成并保存所有仿真曲线图表**（图像将输出到项目根目录下的 `results/` 文件夹）：
  ```bash
  python main.py --all-figures
  ```

---

## 📁 项目结构
```text
lotus_mrc_coldplate/
├── core/                         # 仿真核心计算模块
│   ├── __init__.py
│   ├── geometry.py               # 仿生冷板几何参数与计算类
│   ├── fluid_properties.py       # 流体物性查询（基于CoolProp）
│   ├── single_phase.py           # 单相水冷分区计算模型
│   ├── two_phase.py              # 两相沸腾计算模型（Gungor-Winterton 第一性原理版）
│   ├── comparison.py             # 单/两相数据对比及报告生成
│   └── visualization.py          # 图形绘制与可视化功能
├── results/                      # 命令行模式保存的图表输出目录
├── gui_app.py                    # Streamlit GUI 交互应用主入口
├── main.py                       # CLI 命令行仿真主入口
├── test_new_model.py             # 新模型验证测试脚本
├── CHANGELOG.md                  # 版本变动记录
└── README.md                     # 本说明文档
```

---

## 💡 物理模型说明

### 两相模型（第一性原理版）
新版 `two_phase.py` 完全基于物理机理建模，**不依赖全局标定系数**：

1. **换热系数**：Gungor-Winterton (1987) 关联式
   - `h_tp = S × h_lo + F × h_nb`
   - S：核态沸腾抑制因子（基于两相雷诺数和 Co 数）
   - F：对流增强因子（基于 X_tt 参数）
   - h_nb：Forster-Zuber 核态沸腾（基于流体物性）
2. **几何增强**：从面积比、扩张角、Dean 数等几何参数推导，无硬编码常数
3. **CHF**：Kutateladze-Zuber 空泡脱离准则 + 微通道尺寸修正
4. **流型映射**：无量纲数（Co, We, X_tt）驱动，非固定干度阈值

### 基准说明
- 换热系数 h 在代码中同时提供**基于投影面积**（chip area）和**基于湿面积**（wetted area）两个基准的值，便于与文献对标
- 当前模型在文献基准工况（q=255 W/cm², m=6 g/s, HFE-7100）下预测的湿面积基准换热系数 h_wet ≈ 2.18 W/(cm²·K)，与文献报告的 2.13 W/(cm²·K) 高度吻合（误差 +2.5%）

---

## 📝 更新历史
关于本项目的最新改进记录，请参阅：
* [CHANGELOG.md](file:///d:/Python/lotus_mrc_coldplate/CHANGELOG.md) — 记录各版本的代码重构与功能变动历史。