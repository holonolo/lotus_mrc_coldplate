"""
测试新两相模型
"""
import sys
sys.path.insert(0, r'd:\Python\lotus_mrc_coldplate')

from core.geometry import ManifoldRingChannelGeometry
from core.fluid_properties import FluidProperties
from core.two_phase import TwoPhaseSimulation

# 测试工况: 文献基准点
geo = ManifoldRingChannelGeometry()
tp = TwoPhaseSimulation(geo, FluidProperties("HFE7100"))

print("=" * 60)
print("First-principles model prediction (no calibration):")
print("=" * 60)
res = tp.simulate(heat_flux_Wcm2=255, mass_flow_gs=6.0, T_inlet=20.0)
print(f"  Condition: q={res.heat_flux} W/cm2, m={res.mass_flow_rate} g/s")
print(f"  h_avg = {res.h_conv_cm2:.3f} W/(cm2.K)")
print(f"  T_wall = {res.T_wall_avg:.1f} C")
print(f"  T_sat  = {res.T_sat:.1f} C")
print(f"  CHF    = {res.CHF:.1f} W/cm2")
print(f"  Quality outlet = {res.x_outlet:.3f}")
print(f"  Geometric enhancement = {res.eta_geometry:.3f}")
print(f"  Flow pattern = {res.flow_pattern_name}")
print(f"  Pressure drop = {res.pressure_drop/1e3:.2f} kPa")
print(f"  Thermal resistance = {res.thermal_resistance:.4f} (cm2.K)/W")
print("=" * 60)
print("\nReference: Literature reports h=2.13 W/(cm2.K), CHF=267.05 W/cm2")
print("If predictions are close, model is good; if large deviation, check enhancement factor or CHF parameters")

# Compare with old model (calibrated)
print("\n" + "=" * 60)
print("Comparison with old model (calibrated):")
print("=" * 60)
print("Old model calibrated to: h=2.13, CHF=267.05")
print(f"New model predicts: h={res.h_conv_cm2:.3f}, CHF={res.CHF:.1f}")
error_h = (res.h_conv_cm2 - 2.13) / 2.13 * 100
error_chf = (res.CHF - 267.05) / 267.05 * 100
print(f"Relative error: h={error_h:.1f}%, CHF={error_chf:.1f}%")

if abs(error_h) < 20 and abs(error_chf) < 20:
    print("PASS: Error within acceptable range (<20%)")
else:
    print("WARNING: Large deviation, consider tuning enhancement factor parameters")

# Parameter sweep test
print("\n" + "=" * 60)
print("Parameter sweep (heat flux effect):")
print("=" * 60)
for q in [100, 150, 200, 255, 300]:
    r = tp.simulate(q, 6.0, 20.0)
    print(f"  q={q:3d} W/cm2 -> h={r.h_conv_cm2:.3f} W/(cm2.K), "
          f"T_wall={r.T_wall_avg:5.1f}C, "
          f"CHF_margin={r.CHF_margin:5.1f}%")
