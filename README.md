# QAVS-Net

基于 PyTorch 的可见光与红外图像配准及融合实验项目。模型通过注意力引导的多尺度特征融合和项目内实现的 VMamba 风格二维选择扫描模块，预测单应性变换并生成融合结果。训练程序会记录验证指标、保存检查点和可视化结果。

## 项目文件

| 文件 | 用途 |
| --- | --- |
| `config.py` | 数据路径、训练超参数、模型参数和质量评分配置 |
| `dataset.py` | 多模态图像读取、预处理和随机单应性扰动 |
| `model.py` | 配准与融合网络结构 |
| `train.py` | 数据集划分、训练、验证、测试及结果导出 |

## 环境

- Python 3.10 或更高版本
- PyTorch 及与其版本匹配的 torchvision
- NVIDIA GPU（可选；没有 CUDA 时程序会自动使用 CPU）

建议先根据操作系统、CUDA 版本和 PyTorch 官方安装说明安装 `torch` 与 `torchvision`，再安装其余依赖：

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

需要 CUDA 时，请确保 PyTorch 和 torchvision 来自相匹配的安装渠道/版本。

## 数据集

默认数据根目录是 `QAVS-Net` 的上一级目录下的 `dataset/`，即在当前项目结构中为 `PythonProject2/dataset/`。需要准备以下四个子目录：

```text
PythonProject2/
├── dataset/
│   ├── crop_HR_visible/  # 可见光参考图像
│   ├── crop_LR_visible/  # 可见光待配准/低分辨率图像
│   ├── infrared/        # 红外参考图像
│   └── cropinfrared/    # 红外待配准图像
└── QAVS-Net/
    ├── config.py
    └── train.py
```

图像支持 `.jpg`、`.jpeg`、`.png` 和 `.bmp`。程序以 `crop_HR_visible/` 中的文件名为基准，要求可见光对应图像使用相同文件名；红外对应图像按相同主文件名查找，扩展名可以不同。图像会被缩放到 `256 × 256`。训练、验证和测试集按配置中的比例随机划分。

如需更改目录、划分比例、输入尺寸或模型参数，请编辑 `config.py` 中的 `Config`。数据路径由 `config.py` 根据项目位置自动计算。

## 训练、验证和测试

在项目目录中运行：

```bash
python train.py
```

可选命令行参数：

```bash
python train.py --epochs 100 --batch-size 4 --lr 0.0001
python train.py --max-files 100 --save-dir ./my_experiment
```

| 参数 | 说明 |
| --- | --- |
| `--epochs` | 覆盖训练轮数 |
| `--batch-size` | 覆盖批大小 |
| `--lr` | 覆盖学习率 |
| `--max-files` | 限制参与实验的配对样本数，适合快速试运行 |
| `--save-dir` | 覆盖检查点及结果的输出目录 |

训练结束后，输出目录包含 `latest_model.pth`、`best_quality_model.pth`、损失曲线、测试对比图和 `test_results_summary.txt`。默认输出目录是 `checkpoints666_vmamba/`。项目目录中也可能保留其他实验的输出目录；运行时实际使用的路径以控制台的 `Output` 行为准。

## 说明

- `config.py` 中默认批大小为 4、训练轮数为 200；显存不足时可降低批大小。
- 模型检查点由 PyTorch 保存，体积较大；重新训练会在输出目录中创建或覆盖同名检查点和结果文件。
- 代码对 SIQE 指标使用项目内的代理评分进行质量选择和汇总，并非独立的官方 SIQE 实现。

