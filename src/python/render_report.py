"""Render a concise, evidence-linked report for stages 1 and 2."""

from __future__ import annotations

import json
from pathlib import Path


def render_report(stage1_path: Path, stage2_path: Path, output_path: Path) -> None:
    stage1 = json.loads(stage1_path.read_text(encoding="utf-8"))
    stage2 = json.loads(stage2_path.read_text(encoding="utf-8"))
    candidate1 = stage1["candidate"]
    candidates2 = stage2["candidates"]
    metrics2 = stage2["metrics"]
    reduction = 1 - metrics2["post_sic_residual_power"] / metrics2["pre_sic_power"]

    text = f"""# 6G 多用户 PSS/SSS 联合检测：阶段一至二本地结果

## 执行范围

本次仅完成单用户标准兼容基线与双用户 PBCH 辅助 SIC。PSS/SSS 使用 NR 定义序列；Polar 编码只通过 PBCH 的 BCH Polar+CRC 恢复参与候选确认。

## 阶段一：单用户 NR SS/PBCH 基线

固定 AWGN 场景（PCI={stage1['truth']['pci']}、定时偏移={stage1['truth']['timing_offset_samples']} 样点、CFO={stage1['truth']['cfo_hz']} Hz、SNR={stage1['config']['snr_db']} dB）执行成功。

- PCI：估计 {candidate1['pci']}，正确。
- 定时误差：{stage1['metrics']['timing_error_samples']} 样点。
- CFO：估计 {candidate1['cfo_hz']:.2f} Hz，误差 {stage1['metrics']['cfo_error_hz']:.2f} Hz。
- PBCH Polar+CRC：`{candidate1['pbch_crc_ok']}`。

## 阶段二：双用户重叠与 SIC

场景包括强用户 PCI {stage2['truth']['users'][0]['pci']}（{stage2['truth']['users'][0]['power_db']} dB）和弱用户 PCI {stage2['truth']['users'][1]['pci']}（{stage2['truth']['users'][1]['power_db']} dB），相对定时偏移 {stage2['truth']['users'][1]['timing_offset_samples'] - stage2['truth']['users'][0]['timing_offset_samples']} 样点。

- 无 SIC 单次检测：确认 PCI {stage2['no_sic']['detected_pci']}，PBCH CRC=`{stage2['no_sic']['pbch_crc_ok']}`。
- SIC 第 1 次检测/重构：PCI {candidates2[0]['pci']}，PBCH CRC=`{candidates2[0]['pbch_crc_ok']}`。
- SIC 残差二次检测：PCI {candidates2[1]['pci']}，PBCH CRC=`{candidates2[1]['pbch_crc_ok']}`。
- 接收功率：消除前 `{metrics2['pre_sic_power']:.8g}`，消除后 `{metrics2['post_sic_residual_power']:.8g}`，下降 {reduction:.1%}。

## 结论与边界

该固定、可复现双用户场景证明了“候选检测 → PBCH 解码/DM-RS 估计 → 标准波形重构 → 相减 → 残差检测”的链路可行。它**不代表**在任意 SNR、功率差、多径、残余频偏或三用户以上场景下的统计性能；这些需要后续完整实验矩阵、置信区间与失败样本统计。

本次没有实现 AI 排序、门限自适应、FPGA 资源/时延评估或非标准 PSS/SSS Polar 方案。

## 依据与可追溯来源

- 3GPP TS 38.211：NR 同步信号、SS/PBCH block 与物理资源定义。
  https://www.3gpp.org/ftp/Specs/archive/38_series/38.211/
- 3GPP TS 38.212：PBCH/BCH Polar 编码、速率匹配和 CRC 相关定义。
  https://www.3gpp.org/ftp/Specs/archive/38_series/38.212/

## 复现

在本目录运行：

```powershell
matlab -batch "addpath('src/matlab'); run_single_user_baseline; run_two_user_sic"
python -m pytest tests -v
```
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    render_report(
        root / "outputs" / "stage1_single_user.json",
        root / "outputs" / "stage2_two_user_sic.json",
        root / "outputs" / "stage1_stage2_report.md",
    )
