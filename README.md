

# TicTacToe 项目说明

**项目目标**  
本项目旨在构建一个能够精通井字棋（TicTacToe）游戏的AI模型，通过训练和优化算法，实现智能对弈和预测功能。

## 赞助
为爱发电,请到我的[爱发电](https://afdian.com/a/thzsen)支持我

[![Afdian](https://pic1.afdiancdn.com/static/img/welcome/button-sponsorme.png)](https://afdian.com/a/thzsen)

## 📁 主要组件说明

以下是项目的核心模块和功能：
- `tic_tac_toe/`: 这个目录下存放着模型的训练器。
- `tic_tac_toe/model.py`: 定义用于训练的神经网络模型。
- `tic_tac_toe/datasets.py`: 提供用于数据加载和处理的数据集类。
- `tic_tac_toe/train.py`: 实现模型训练的完整流程。
- `tic_tac_toe/generate_datas.py`: 包含数据生成相关工具方法。
- `tic_tac_toe/eval.py`: AI的自对弈,包含用于模型预测和展示棋盘的功能。
- `main.py`: 项目入口点，用于启动训练。
- `alpha_beta.py`: 提供井字棋游戏的决策算法。


## 安装与使用

### 安装依赖

确保安装了以下依赖：

- Python 3.x
- PaddlePaddle
- NumPy
- onnxruntime

1. **生成数据集**：
```bash
python cli.py generate_data --num_samples 5000 --save_dir data
```
注: 生成的数据集将保存在`data/`目录下。
- `num_samples`: 生成的数据样本数量
- `save_dir`: 保存数据集的目录
2. **训练模型**：
```bash
python cli.py train --epochs 1000 --dropout_prob 0.3 --learning_rate 0.001 --batch_size 64 --test_size 0.2 --random_state 42
```
训练模型过程中checkpoints将会保存在`checkpoints/`目录下。
训练完成后,`models`目录下会有一个最终的模型`tictactoe.pdparams`。注意:那并不是最优模型,最优模型在`checkpoints/best_model.pdparams`.
- `epochs`: 训练的轮数
- `dropout_prob`: Dropout概率(用于防止过拟合)
- `learning_rate`: 学习率
- `batch_size`: 批量大小
- `test_size`: 测试集比例
- `random_state`: 随机种子(确保可重复性)

3. **断点续训**：
```bash
python cli.py resume_train --checkpoint_path checkpoints/checkpoint_epoch_100.pdparams --epochs 1000 --dropout_prob 0.3 --learning_rate 0.001 --batch_size 64 --test_size 0.2 --random_state 42
```
断点续训会从指定的checkpoint开始继续训练，直到达到设定的epochs数。
- `checkpoint_path`: 断点续训的起始checkpoint
- `epochs`: 训练的轮数
- `dropout_prob`: Dropout概率(用于防止过拟合)
- `learning_rate`: 学习率
- `batch_size`: 批量大小
- `test_size`: 测试集比例
- `random_state`: 随机种子(确保可重复性)

4. **推理**：
```bash
python cli.py predict --board 'X  ' --board ' O ' --board '  O' --player X --model_path checkpoints/best_model.pdparams --difficulty 3
```
注:传入棋盘的方式为 `--board 'X  ' --board ' O ' --board '  O'` 
这个表示:
```python
[
    ['X','',''],
    ['','O',''],
    ['','','O']
]
```

`--player` 参数表示玩家是X还是O 只能填写X或O

5. **导出到onnx**：
使用方法
你可以使用以下命令将训练好的模型转换为 ONNX 格式：
```bash
python cli.py export_onnx --model_path models/tictactoe.pdparams --output_path models/tictactoe.onnx
```
这个命令会将 models/tictactoe.pdparams 中的模型转换为 ONNX 格式，并保存到 models/tictactoe.onnx。你可以根据需要调整 model_path 和 output_path 参数。
**注意:传入的model_path不可以是checkpoints,必须是最后的model!!!**

6.**将checkpoint导出到model**
```bash
python cli.py export_model_from_checkpoint --checkpoint_path checkpoints/checkpoint_epoch_100.pdparams --output_path models/tictactoe.pdparams
```
这个命令会将 checkpoints/checkpoint_epoch_100.pdparams 中的 model_state_dict 提取出来，并保存到 models/tictactoe.pdparams。你可以根据需要调整 checkpoint_path 和 output_path 参数。

**下列内容是代码的内容可以不看**

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

