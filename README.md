

# TicTacToe 项目说明

🧠 **项目目标**  
本项目旨在构建一个能够精通井字棋（TicTacToe）游戏的AI模型，通过训练和优化算法，实现智能对弈和预测功能。

## 📁 主要组件说明

以下是项目的核心模块和功能：

- `model.py`: 定义用于训练的神经网络模型。
- `datasets.py`: 提供用于数据加载和处理的数据集类。
- `train.py`: 实现模型训练的完整流程。
- `utils.py`: 包含数据生成和游戏逻辑相关工具方法。
- `main.py`: 项目入口点，用于启动训练或预测。
- `train_API.py`: 提供训练相关的API接口。
- `eval.py`: 包含用于模型预测和展示棋盘的功能。
- `predict.py`: 实现AI代理和预测功能。
- `alpha_beta.py`: 提供井字棋游戏的决策算法。

## 数据文件

- `datas/tic_tac_toe_data.npy`: 存储训练数据。
- `datas/tic_tac_toe_labels.npy`: 存储对应的标签数据。

## 🛠️ 安装与使用

### 安装依赖

确保安装了以下依赖：

- Python 3.x
- PaddlePaddle
- NumPy
- 其他标准库（如os, random等）

### 数据生成

使用 `utils.py` 中的 `TicTacToeDataGenerator` 类生成训练数据：

```python
from utils import TicTacToeDataGenerator

generator = TicTacToeDataGenerator()
generator.generate_and_save(num_samples=10000, save_dir='datas')
```

### 数据集加载

使用 `datasets.py` 加载训练数据：

```python
from datasets import get_data_loaders

train_loader, test_loader = get_data_loaders(data_dir='datas', batch_size=64)
```

### 模型训练

使用 `train.py` 进行模型训练：

```python
from model import TicTacToeModel
from train import TicTacToeTrainer

model = TicTacToeModel()
trainer = TicTacToeTrainer(model, train_loader, test_loader)
trainer.train(epochs=100)
```

## 📌 贡献指南

欢迎贡献！请确保提交的代码遵循项目结构和风格，并提供必要的测试和文档。

## 📄 许可证

本项目采用 MIT 许可证，请遵守相关条款。

## 📬 联系方式

有关项目问题或合作，请联系：thzsen@example.com