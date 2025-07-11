

# TicTacToe 项目说明

**项目目标**  
本项目旨在构建一个能够精通井字棋（TicTacToe）游戏的AI模型，通过训练和优化算法，实现智能对弈和预测功能。

## 📁 主要组件说明

以下是项目的核心模块和功能：
- `tic_tac_toe/`: 这个目录下存放着模型的训练器。
- `tic_tac_toe/model.py`: 定义用于训练的神经网络模型。
- `tic_tac_toe/datasets.py`: 提供用于数据加载和处理的数据集类。
- `tic_tac_toe/train.py`: 实现模型训练的完整流程。
- `utils.py`: 包含数据生成相关工具方法。
- `main.py`: 项目入口点，用于启动训练。
- `eval.py`: AI的自对弈,包含用于模型预测和展示棋盘的功能。
- `alpha_beta.py`: 提供井字棋游戏的决策算法。

## 数据文件

- `datas/tic_tac_toe_data.npy`: 存储训练数据。
- `datas/tic_tac_toe_labels.npy`: 存储对应的标签数据。

## 安装与使用

### 安装依赖

确保安装了以下依赖：

- Python 3.x
- PaddlePaddle
- NumPy
- onnxruntime

### 数据生成

运行
```bash
python utils.py
```
会自动生成5000个样本，并保存到`data`目录下。
如果你想修改数据,请修改utils.py

### 数据集加载

在main.py中就通过类似于下列的方法来加载

使用 `datasets.py` 加载训练数据：

```python
from tic_tac_toe.datasets import get_data_loaders

train_loader, test_loader = get_data_loaders(data_dir='datas', batch_size=64)
```

### 模型训练

使用 `train.py` 进行模型训练：

```python
from tic_tac_toe.model import TicTacToeModel
from tic_tac_toe.train import TicTacToeTrainer
from tic_tac_toe.datasets import get_data_loaders

model = TicTacToeModel(dropout_prob=0.3)
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.001)
loss_fn = paddle.nn.CrossEntropyLoss()

train_loader, test_loader = get_data_loaders(
    data_dir='data',      # 数据目录
    batch_size=64,        # 批量大小
    test_size=0.2,        # 测试集比例(20%)
    random_state=42       # 随机种子(确保可重复性)
)

trainer = TicTacToeTrainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    train_loader=train_loader,
    test_loader=test_loader,
    checkpoint_dir='checkpoints',
    device='gpu' if paddle.device.is_compiled_with_cuda() else 'cpu',
    lr_scheduler=lr_scheduler
)

history = trainer.train(epochs=1000)

```
在checkpoints目录下会checkpoints文件,加载模型的方法:
```python
import paddle
from tic_tac_toe.model import TicTacToeModel

model = TicTacToeModel()
model.set_state_dict(paddle.load('checkpoints/best_model.pdparams')['model_state_dict'])
```
训练结束后,在`models`目录下会保存着`tictactoe.pdparams`文件,想要加载他请直接:
```python
import paddle
from tic_tac_toe.model import TicTacToeModel

model = TicTacToeModel()
model.set_state_dict(paddle.load('models/tictactoe.pdparams'))
```

### 断点续训
当模型训练被意外暂停时,可以通过checkpoints目录下的文件继续训练:
```python
from tic_tac_toe.model import TicTacToeModel
from tic_tac_toe.train import TicTacToeTrainer
from tic_tac_toe.datasets import get_data_loaders

model = TicTacToeModel(dropout_prob=0.3)
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.001)
loss_fn = paddle.nn.CrossEntropyLoss()

train_loader, test_loader = get_data_loaders(
    data_dir='data',      # 数据目录
    batch_size=64,        # 批量大小
    test_size=0.2,        # 测试集比例(20%)
    random_state=42       # 随机种子(确保可重复性)
)

trainer = TicTacToeTrainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    train_loader=train_loader,
    test_loader=test_loader,
    checkpoint_dir='checkpoints',
    device='gpu' if paddle.device.is_compiled_with_cuda() else 'cpu',
    lr_scheduler=lr_scheduler
)

last_epoch = trainer.load_checkpoint('checkpoints/checkpoints_epoch_100.pdparams')
history = trainer.train(epochs=1000, start_epoch=last_epoch)

```

## 📌 贡献指南

欢迎贡献！请确保提交的代码遵循项目结构和风格，并提供必要的测试和文档。

## 📄 许可证

本项目采用 BSD 3-Clause 许可证，请遵守相关条款。

## 赞助
为爱发电,请到我的[爱发电](https://afdian.com/a/thzsen)支持我

[![Afdian](https://pic1.afdiancdn.com/static/img/welcome/button-sponsorme.png)](https://afdian.com/a/thzsen)
