import click
import paddle
from tic_tac_toe.model import TicTacToeModel
from tic_tac_toe.train import TicTacToeTrainer
from tic_tac_toe.datasets import get_data_loaders
from tic_tac_toe.generate_datas import TicTacToeDataGenerator
from predict import TicTacToeAgent


@click.group()
def cli():
    pass


@cli.command()
@click.option('--num_samples', default=5000, help='Number of samples to generate.')
@click.option('--save_dir', default='data', help='Directory to save the generated data.')
def generate_data(num_samples, save_dir):
    """Generate and save TicTacToe dataset."""
    generator = TicTacToeDataGenerator()
    generator.generate_and_save(num_samples=num_samples, save_dir=save_dir)


@cli.command()
@click.option('--epochs', default=1000, help='Total number of training epochs.')
@click.option('--dropout_prob', default=0.3, help='Dropout probability for the model.')
@click.option('--learning_rate', default=0.001, help='Learning rate for the optimizer.')
@click.option('--batch_size', default=64, help='Batch size for training.')
@click.option('--test_size', default=0.2, help='Proportion of the test set.')
@click.option('--random_state', default=42, help='Random seed for data splitting.')
def train(epochs, dropout_prob, learning_rate, batch_size, test_size, random_state):
    """Train the TicTacToe model from scratch."""
    model = TicTacToeModel(dropout_prob=dropout_prob)
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=learning_rate)
    loss_fn = paddle.nn.CrossEntropyLoss()

    train_loader, test_loader = get_data_loaders(
        data_dir='data',
        batch_size=batch_size,
        test_size=test_size,
        random_state=random_state
    )

    lr_scheduler = paddle.optimizer.lr.ReduceOnPlateau(
        learning_rate=learning_rate,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
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

    history = trainer.train(epochs=epochs)


@cli.command()
@click.option('--checkpoint_path', default='checkpoints/checkpoint_epoch_100.pdparams',
              help='Path to the checkpoint file for resuming training.')
@click.option('--epochs', default=1000, help='Total number of training epochs.')
@click.option('--dropout_prob', default=0.3, help='Dropout probability for the model.')
@click.option('--learning_rate', default=0.001, help='Learning rate for the optimizer.')
@click.option('--batch_size', default=64, help='Batch size for training.')
@click.option('--test_size', default=0.2, help='Proportion of the test set.')
@click.option('--random_state', default=42, help='Random seed for data splitting.')
def resume_train(checkpoint_path, epochs, dropout_prob, learning_rate, batch_size, test_size, random_state):
    """Resume training the TicTacToe model from a checkpoint."""
    model = TicTacToeModel(dropout_prob=dropout_prob)
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=learning_rate)
    loss_fn = paddle.nn.CrossEntropyLoss()

    train_loader, test_loader = get_data_loaders(
        data_dir='data',
        batch_size=batch_size,
        test_size=test_size,
        random_state=random_state
    )

    lr_scheduler = paddle.optimizer.lr.ReduceOnPlateau(
        learning_rate=learning_rate,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
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

    last_epoch = trainer.load_checkpoint(checkpoint_path)
    history = trainer.train(epochs=epochs, start_epoch=last_epoch)


@cli.command()
@click.option('--board', multiple=True, help='The current TicTacToe board as a 2D array.')
@click.option('--player', type=click.Choice(['X', 'O'], case_sensitive=False), help='The current player (X or O).')
@click.option('--model_path', default='checkpoints/best_model.pdparams', help='Path to the trained model.')
@click.option('--difficulty', default=3, help='Difficulty level (1-5).')
def predict(board, player, model_path, difficulty):
    """Predict the next move for the given board and player."""
    player = player.upper()
    board = [list(row) for row in board]

    agent = TicTacToeAgent(model_path=model_path)
    move = agent.predict_move(board, player, difficulty=difficulty)
    print(f"Predicted move: Row {move[0] + 1}, Column {move[1] + 1}")

@cli.command()
@click.option('--model_path', default='checkpoints/best_model.pdparams', help='Path to the trained model.')
@click.option('--output_path', default='models/tictactoe.onnx', help='Path to save the ONNX model.')
def export_onnx(model_path, output_path):
    """Export the trained model to ONNX format."""
    model = TicTacToeModel()
    model.set_state_dict(paddle.load(model_path))
    x_spec = paddle.static.InputSpec(shape=[None, 2, 3, 3], dtype='float32', name="input")
    paddle.onnx.export(model, output_path, [x_spec], opset_version=11)
    print(f"Model exported to {output_path}")
    
@cli.command()
@click.option('--checkpoint_path', default='checkpoints/checkpoint_epoch_100.pdparams',
              help='Path to the checkpoint file.')
@click.option('--output_path', default='models/tictactoe.pdparams',
              help='Path to save the extracted model state dict.')
def export_model_from_checkpoint(checkpoint_path, output_path):
    """Export the model state dict from the checkpoint."""
    checkpoint = paddle.load(checkpoint_path)
    model_state_dict = checkpoint["model_state_dict"]
    paddle.save(model_state_dict, output_path)
    print(f"Model state dict exported to {output_path}")

if __name__ == '__main__':
    cli()