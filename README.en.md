# TicTacToe Project Instructions

**Project Objectives**  
This project aims to build an AI model proficient in the Tic-Tac-Toe game. Through training and optimizing algorithms, it can achieve intelligent gameplay and prediction functions.

## Sponsorship
If you'd like to support this project, please visit my [Afdian page](https://afdian.com/a/thzsen).

[![Afdian](https://pic1.afdiancdn.com/static/img/welcome/button-sponsorme.png)](https://afdian.com/a/thzsen)

## 📁 Key Component Descriptions

The following are the core modules and functions of the project:
- `tic_tac_toe/`: This directory stores the model trainer.
- `tic_tac_toe/model.py`: Defines the neural network model for training.
- `tic_tac_toe/datasets.py`: Provides dataset classes for data loading and processing.
- `tic_tac_toe/train.py`: Implements the complete process of model training.
- `tic_tac_toe/generate_datas.py`: Contains utility methods related to data generation.
- `tic_tac_toe/eval.py`: Enables self-play of the AI, including functions for model prediction and board display.
- `main.py`: The entry point of the project, used to start training.
- `alpha_beta.py`: Provides the decision-making algorithm for the Tic-Tac-Toe game.

## Installation and Usage

### Install Dependencies
Ensure that the following dependencies are installed:
- Python 3.x
- PaddlePaddle
- NumPy
- onnxruntime

1. **Generate the Dataset**:
```bash
python cli.py generate_data --num_samples 5000 --save_dir data
```
Note: The generated dataset will be saved in the `data/` directory.
- `num_samples`: The number of data samples to generate.
- `save_dir`: The directory to save the dataset.

2. **Train the Model**:
```bash
python cli.py train --epochs 1000 --dropout_prob 0.3 --learning_rate 0.001 --batch_size 64 --test_size 0.2 --random_state 42
```
During the model training process, checkpoints will be saved in the `checkpoints/` directory.
After training is completed, there will be a final model `tictactoe.pdparams` in the `models` directory. Note: This is not the optimal model. The optimal model is at `checkpoints/best_model.pdparams`.
- `epochs`: The number of training epochs.
- `dropout_prob`: The Dropout probability (used to prevent overfitting).
- `learning_rate`: The learning rate.
- `batch_size`: The batch size.
- `test_size`: The proportion of the test set.
- `random_state`: The random seed (to ensure reproducibility).

3. **Resume Training from a Checkpoint**:
```bash
python cli.py resume_train --checkpoint_path checkpoints/checkpoint_epoch_100.pdparams --epochs 1000 --dropout_prob 0.3 --learning_rate 0.001 --batch_size 64 --test_size 0.2 --random_state 42
```
Resuming training will start from the specified checkpoint and continue until the set number of epochs is reached.
- `checkpoint_path`: The starting checkpoint for resuming training.
- `epochs`: The number of training epochs.
- `dropout_prob`: The Dropout probability (used to prevent overfitting).
- `learning_rate`: The learning rate.
- `batch_size`: The batch size.
- `test_size`: The proportion of the test set.
- `random_state`: The random seed (to ensure reproducibility).

4. **Inference**:
```bash
python cli.py predict --board 'X  ' --board ' O ' --board '  O' --player X --model_path checkpoints/best_model.pdparams --difficulty 3
```
Note: The way to pass the board is `--board 'X  ' --board ' O ' --board '  O'`, which represents:
```python
[
    ['X', '', ''],
    ['', 'O', ''],
    ['', '', 'O']
]
```
The `--player` parameter indicates whether the player is 'X' or 'O'. You can only enter 'X' or 'O'.

5. **Export to ONNX**:
You can use the following command to convert the trained model to ONNX format:
```bash
python cli.py export_onnx --model_path models/tictactoe.pdparams --output_path models/tictactoe.onnx
```
This command will convert the model in `models/tictactoe.pdparams` to ONNX format and save it as `models/tictactoe.onnx`. You can adjust the `model_path` and `output_path` parameters as needed.
**Note: The `model_path` passed in cannot be a checkpoint. It must be the final model!!!**

6. **Export a Checkpoint to a Model**:
```bash
python cli.py export_model_from_checkpoint --checkpoint_path checkpoints/checkpoint_epoch_100.pdparams --output_path models/tictactoe.pdparams
```
This command will extract the `model_state_dict` from `checkpoints/checkpoint_epoch_100.pdparams` and save it as `models/tictactoe.pdparams`. You can adjust the `checkpoint_path` and `output_path` parameters as needed.

**The following content is the code and can be ignored.**

### Dataset Loading
In `main.py`, you can load the data in a similar way as follows:
Use `datasets.py` to load the training data:
```python
from tic_tac_toe.datasets import get_data_loaders

train_loader, test_loader = get_data_loaders(data_dir='datas', batch_size=64)
```

### Model Training
Use `train.py` to train the model:
```python
from tic_tac_toe.model import TicTacToeModel
from tic_tac_toe.train import TicTacToeTrainer
from tic_tac_toe.datasets import get_data_loaders

model = TicTacToeModel(dropout_prob=0.3)
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.001)
loss_fn = paddle.nn.CrossEntropyLoss()

train_loader, test_loader = get_data_loaders(
    data_dir='data',      # Data directory
    batch_size=64,        # Batch size
    test_size=0.2,        # Proportion of the test set (20%)
    random_state=42       # Random seed (to ensure reproducibility)
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
Checkpoint files will be saved in the `checkpoints` directory. To load the model, use the following code:
```python
import paddle
from tic_tac_toe.model import TicTacToeModel

model = TicTacToeModel()
model.set_state_dict(paddle.load('checkpoints/best_model.pdparams')['model_state_dict'])
```
After training is completed, the `tictactoe.pdparams` file will be saved in the `models` directory. To load it, use the following code:
```python
import paddle
from tic_tac_toe.model import TicTacToeModel

model = TicTacToeModel()
model.set_state_dict(paddle.load('models/tictactoe.pdparams'))
```

### Resume Training
When the model training is unexpectedly interrupted, you can continue training using the files in the `checkpoints` directory:
```python
from tic_tac_toe.model import TicTacToeModel
from tic_tac_toe.train import TicTacToeTrainer
from tic_tac_toe.datasets import get_data_loaders

model = TicTacToeModel(dropout_prob=0.3)
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.001)
loss_fn = paddle.nn.CrossEntropyLoss()

train_loader, test_loader = get_data_loaders(
    data_dir='data',      # Data directory
    batch_size=64,        # Batch size
    test_size=0.2,        # Proportion of the test set (20%)
    random_state=42       # Random seed (to ensure reproducibility)
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

## 📌 Contribution Guidelines
We welcome contributions! Please ensure that the submitted code follows the project structure and style, and provide necessary tests and documentation.

## 📄 License
This project uses the BSD 3-Clause License. Please comply with the relevant terms.