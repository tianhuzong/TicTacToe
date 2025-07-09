This project is a code repository for training and running an artificial intelligence model for the Tic-Tac-Toe game. It combines deep learning and reinforcement learning techniques, aiming to generate data through self-play and train the model to improve its gameplay.

---

## 🧠 Project Objectives

The objective of this project is to build an AI model capable of autonomously learning and mastering Tic-Tac-Toe strategies. By utilizing neural network models and reinforcement learning algorithms (such as Alpha-Beta pruning), the project generates high-quality training data, trains the model, and evaluates its performance.

---

## 📁 Key Component Descriptions

### `model.py` / `tic_tac_toe/model.py`
- Implements the neural network model `TicTacToeModel`.
- The model input shape is `[batch_size, 2, 3, 3]`, representing the board state (positions of the player and opponent).
- Provides forward propagation logic.

### `datasets.py` / `tic_tac_toe/datasets.py`
- Defines the `TicTacToeDataset` class for generating and loading Tic-Tac-Toe game data.
- Includes functionalities for dataset generation, encoding, and loading.
- Supports splitting data into training and testing sets.

### `train.py` / `tic_tac_toe/train.py`
- Contains the `TicTacToeTrainer` class, which manages the model training and evaluation process.
- Supports saving and loading model checkpoints; during training, optimizers, learning rate schedulers, and early stopping mechanisms can be configured.
- Provides complete training and evaluation interfaces.

### `utils.py`
- Provides the `TicTacToeDataGenerator` class for generating Tic-Tac-Toe game samples.
- Includes core functionalities such as determining win/loss conditions, board state evaluation, Alpha-Beta pruning algorithm, and optimal move strategy.
- Supports data generation and saving to specified directories.

### `main.py`
- Entry point of the program, potentially used to initialize the training process or run model inference.

### `train_API.py` / `tic_tac_toe/train_API.py`
- May contain API interfaces for training, facilitating integration with other modules or external systems.

### Data Files
- `datas/tic_tac_toe_data.npy`: Stores game state data.
- `datas/tic_tac_toe_labels.npy`: Stores corresponding labels (e.g., optimal move positions).

---

## 🛠️ Installation and Usage

### Install Dependencies
```bash
pip install paddlepaddle numpy scikit-learn
```

### Data Generation
Use `utils.py` to generate and save datasets:
```python
from utils import TicTacToeDataGenerator

generator = TicTacToeDataGenerator()
generator.generate_and_save(num_samples=10000, save_dir='data')
```

### Dataset Loading
Use `tic_tac_toe/datasets.py` to load the data:
```python
from tic_tac_toe.datasets import get_data_loaders

train_loader, test_loader = get_data_loaders(data_dir='data', batch_size=64)
```

### Model Training
Use `TicTacToeTrainer` from `tic_tac_toe/train.py` to start training:
```python
from tic_tac_toe.train import TicTacToeTrainer

trainer = TicTacToeTrainer(model, optimizer, loss_fn, train_loader, test_loader)
trainer.train(epochs=100)
```

---

## 📌 Contribution Guidelines

We welcome issues and pull requests for this project. Here are some suggested areas for contribution:
- Optimize the model architecture to improve accuracy
- Improve data generation logic to increase sample diversity
- Enhance training logging and visualization features
- Add support for additional game strategies or AI algorithms

---

## 📄 License

This project uses a standard open-source license. Please refer to the `LICENSE` file in the repository for details.

---

## 📬 Contact

If you have any questions or suggestions, please contact us via the issue system or private message on Gitee!