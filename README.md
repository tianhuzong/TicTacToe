

该项目是一个用于训练和运行井字棋（Tic-Tac-Toe）游戏的人工智能模型的代码库。它结合了深度学习和强化学习技术，旨在通过自我对弈生成数据并训练模型以提高游戏水平。

---

## 🧠 项目目标

本项目的目标是构建一个能够自主学习并掌握井字棋游戏策略的 AI 模型。通过使用神经网络模型和强化学习算法（如 Alpha-Beta 剪枝），项目能够生成高质量的训练数据，训练模型并评估其性能。

---

## 📁 主要组件说明

### `model.py` / `tic_tac_toe/model.py`
- 实现了神经网络模型 `TicTacToeModel`。
- 模型输入形状为 `[batch_size, 2, 3, 3]`，用于表示棋盘状态（玩家和对手的位置）。
- 提供前向传播逻辑。

### `datasets.py` / `tic_tac_toe/datasets.py`
- 定义了 `TicTacToeDataset` 类，用于生成和加载井字棋游戏数据。
- 包括数据集的生成、编码、加载等功能。
- 支持训练和测试数据的划分。

### `train.py` / `tic_tac_toe/train.py`
- 包含 `TicTacToeTrainer` 类，管理模型训练和评估流程。
- 支持保存和加载模型检查点，模型训练过程中可设置优化器、学习率调度器、早停机制等。
- 提供完整的训练和评估接口。

### `utils.py`
- 提供了 `TicTacToeDataGenerator` 类，用于生成井字棋游戏样本。
- 包含判断胜负、棋盘状态评估、Alpha-Beta 剪枝算法、最佳落子策略等核心功能。
- 支持数据生成并保存至指定目录。

### `main.py`
- 程序入口文件，可能用于初始化训练流程或运行模型推理。

### `train_API.py` / `tic_tac_toe/train_API.py`
- 可能包含用于训练的 API 接口，便于与其他模块或外部系统集成。

### 数据文件
- `datas/tic_tac_toe_data.npy`: 存储游戏状态数据。
- `datas/tic_tac_toe_labels.npy`: 存储对应标签（例如最佳落子位置）。

---

## 🛠️ 安装与使用

### 安装依赖
```bash
pip install paddlepaddle numpy scikit-learn
```

### 数据生成
使用 `utils.py` 生成并保存数据集：
```python
from utils import TicTacToeDataGenerator

generator = TicTacToeDataGenerator()
generator.generate_and_save(num_samples=10000, save_dir='data')
```

### 数据集加载
使用 `tic_tac_toe/datasets.py` 加载数据：
```python
from tic_tac_toe.datasets import get_data_loaders

train_loader, test_loader = get_data_loaders(data_dir='data', batch_size=64)
```

### 模型训练
使用 `tic_tac_toe/train.py` 中的 `TicTacToeTrainer` 启动训练：
```python
from tic_tac_toe.train import TicTacToeTrainer

trainer = TicTacToeTrainer(model, optimizer, loss_fn, train_loader, test_loader)
trainer.train(epochs=100)
```

---

## 📌 贡献指南

欢迎为本项目提交 issue 或 pull request。以下是一些贡献方向：
- 优化模型结构以提高准确率
- 改进数据生成逻辑，增加样本多样性
- 完善训练日志和可视化功能
- 增加支持更多游戏策略或 AI 算法

---

## 📄 许可证

本项目使用标准开源许可证，具体请查看仓库中的 `LICENSE` 文件。

---

## 📬 联系方式

如果您有任何问题或建议，请通过 Gitee 的 issue 系统或私信联系我们！