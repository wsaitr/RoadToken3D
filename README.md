# RoadToken3D — current model for Huawei Ascend

私有预发布仓库。仅上传当前论文模型所需代码、一个合成检查和 **3条案例轨迹**；完整样本待录用后另行发布。

## 当前版本

模型算法以当前415窗口论文的 `joint-timegate6-anchor-20261023-v6` 为准，包含：

- 保留原始XYZ预测的可学习笛卡尔分支；
- 共享纵向进度的三维道路解码、水平20米/道路Z修正2米限幅；
- 物理通行时延消息和最早到达约束；
- 学习式时间占用门（logit增益10、1米监督死区、类别平衡）；
- 只加到Z的学习式高度锚及其监督；
- 原6项损失加高度门/时间门两项辅助损失，按全局128条样本汇总分子、分母。

这次提供的是上述当前模型的昇腾适配入口，**不是旧的端点稳定版改名**。旧训练调度器、云端发布工具、诊断历史、原始大数据、检查点和其他加速器实现均未上传。父模型依赖文件中的旧版本编号属于继承关系，不代表主模型回退。

**验证边界：算法同步和小型CPU合成检查已完成；昇腾实机验收尚未完成。** 不据此声称已有昇腾训练结果或可逐位复现其他设备的历史数值。前缀累加采用保持零填充尾部一致的昇腾设备内参考实现；其数值差异和速度仍须实机验证。

## 使用

先按[华为官方兼容说明](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)准备匹配的CANN、PyTorch与`torch_npu`，再运行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m pip install -e . --no-deps
python -m roadtoken3d.ascend --help
```

准备配置后，在实际昇腾环境中显式运行：

```bash
python -m roadtoken3d.ascend train --config configs/current.json --output runs/new-ascend
python -m roadtoken3d.ascend predict --config configs/current.json --weights checkpoints/full.pt --output runs/new-prediction
```

这些命令不会由仓库自动执行；需要另行准备冻结父模型、道路几何和完整输入。配置模板为 [configs/current.example.json](configs/current.example.json)，输入字段和检查点规则见 [INTERFACE.md](INTERFACE.md)。模型接口与设备入口分开，CPU仅用于小型实现检查；训练/预测入口强制使用真实NPU。

检查已提供样本无需模型或加速器：

```bash
python tools/inspect_samples.py
```

已有PyTorch的环境可运行唯一合成检查：

```bash
PYTHONPATH=src python tests/smoke_current.py
```

3条CSV保留论文已有案例的相对XYZ目标轨迹，分别为长程延续、高度变化和水平失败例。不含车辆标识、绝对定位、模型预测、完整历史或道路图；它们用于展示格式与轨迹，不能代替完整训练输入或全样本性能评价。

## 最小代码导航

| 文件 | 用途 |
|---|---|
| `src/roadtoken3d/current_model.py` | 当前v6模型、时间门、高度锚及配对控制配置 |
| `src/roadtoken3d/current_loss.py`、`optimization.py` | 当前损失、门控监督及汇总更新 |
| `src/roadtoken3d/residual.py`、`endpoint.py` | 道路相对残差和端点稳定解码 |
| `src/roadtoken3d/ascend.py`、`data.py` | 昇腾入口及准备好的窗口输入 |
| `src/roadtoken3d/prefix.py` | 昇腾前缀累加适配，不回退主机张量计算 |
| 其余模型文件 | 当前父模型的直接依赖，已去掉无用旧定义 |
| `SOURCE_SYNC.json` | 同步来源、文件指纹与实机验证状态 |

## English

This private preview contains the current paper's v6 model, an Ascend-only FP32 runtime and three illustrative target trajectories. Learned time occupancy, the Z-only height anchor, the learned XYZ branch, bounded road residual and all eight training terms are included. Historical deployment/diagnostic code and alternative accelerator implementations are excluded.

CPU synthetic checks verify implementation properties; they are not Ascend device qualification or paper performance. NPU execution, numerical compatibility and speed remain unverified on actual Ascend hardware. Full data and checkpoints are not included. See the interface document before running training.
