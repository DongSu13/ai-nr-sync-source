# AI 辅助 NR 同步

[中文](README.md) | [English](README.en.md)

本仓库提供 NR 同步的仿真代码，包括 PSS/CFO 前端同步、多发射源（小区）PBCH 接收、串行干扰消除（SIC）和 AI 候选排序。输入信号由代码合成生成，无需下载外部数据集。

研究问题分为两类，指标不能混用：

1. 在同一波形和 PSS 互相关实现下，单次 AI CFO 提议能否相对单次经典自相关 CFO 提议提高 PSS 正确峰命中率并降低 CFO RMSE。
2. 在相同 PBCH 解码预算下，AI 候选排序能否帮助 SIC 链恢复更多真实发射源。

这里没有 FPGA 部署、真实射频采集或网络接入实验。信道与数据均由代码合成生成，近似多径实现用于研究比较，不等同于完整 3GPP 一致性测试。

## 方法概览

```text
接收采样 -> CFO 估计/补偿 -> PSS 互相关峰与 N_ID2 -> SSS 与 PCI
         -> PBCH DM-RS 信道估计 -> Polar 译码 + CRC 确认
         -> 重建已确认用户并 SIC -> 在残差上重复搜索
```

前端 AI 给出 CFO 提议，最终 PSS 搜索仍使用相同的相关器。多用户 AI 用于候选排序；PBCH CRC 与 SIC 的物理链路不因 AI 而省略。仓库也包含经典网格、经典自相关、CNN/CNN-LSTM、扩展模型族和统计审计代码，用于比较不同预算下的同步与恢复结果。

## 代码结构

| 路径 | 内容 |
| --- | --- |
| `src/python/` | 合成场景、PSS/CFO 模型、训练、候选排序、统计审计和运行入口 |
| `src/matlab/` | SS/PBCH 接收、多用户 SIC、MATLAB 链路与 5G Toolbox 探测脚本 |
| `configs/` | 小规模基线与候选生成配置 |
| `tests/` | 生成器、模型接口、审计逻辑、MATLAB 启动器和核心算法测试 |

默认输出目录是 `outputs/`。模型权重、生成数据、波形、日志和报告均被 `.gitignore` 排除。

## 安装与测试

依赖范围对应 Python 3.12 环境。以下命令中的 `python` 必须指向已选定的 Python 3.12 解释器；创建环境前先用 `python --version` 确认：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests -q
```

CUDA 训练测试在无 GPU 环境会跳过；MATLAB 链路需要 MATLAB R2022a 或兼容版本以及 5G Toolbox。MATLAB 可执行文件通过 `MATLAB_BIN` 或系统 `PATH` 发现：

把 MATLAB 加入 `PATH`，或设置 `MATLAB_BIN` 指向本机 `matlab` 可执行文件后运行：

```powershell
python src/python/matlab_launcher.py v4 development
```

`development` 先生成训练和开发数据，完整仿真规模较大。完成后运行 `v4 final` 进行模型选择与独立评估。也可用 `v4 all` 连续执行两个阶段。

以下训练烟测需要 CUDA GPU，输出目录必须尚不存在：

```powershell
python src/python/run_paper_aligned.py --output outputs/paper_aligned_smoke --smoke
python src/python/run_paper_repro.py --output outputs/paper_repro_smoke --train 16 --val 8 --test 8 --epochs 1
```

这些命令生成新的合成数据与模型。独立 CFO 实验默认从随机初始化训练；可通过 `--initial-weights` 指定自己生成的兼容权重。扩展模型比较在 `develop` 阶段要求 `--reference-weight`，可使用上面四模型训练输出的 `weights/cross_cnn.pt`。

## 引用与边界

本项目参考公开 NR 同步与 AI 辅助同步方向的方法，并实现了自己的仿真、接收与比较流程。相关背景可参考：

- [Wang 等，Signal Synchronization for 5G NR Under Large CFOs Based on Convolutional Neural Network Combined with Long Short-Term Memory，Electronics 2026, 15(8), 1566](https://www.mdpi.com/2079-9292/15/8/1566)
- [3GPP TS 38.211](https://www.3gpp.org/ftp/Specs/archive/38_series/38.211/)
- [3GPP TS 38.212](https://www.3gpp.org/ftp/Specs/archive/38_series/38.212/)

## 许可证

本仓库代码采用 [MIT License](LICENSE)，允许使用、修改和再分发，须保留版权及许可声明。MATLAB、5G Toolbox 和其他第三方依赖遵循各自许可证。
