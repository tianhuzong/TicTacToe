import paddle
from .model import TicTacToeModel
def export(model_path, output_path):
    model = TicTacToeModel()
    model.set_state_dict(paddle.load(model_path)["model_state_dict"])
    x_spec = paddle.static.InputSpec(shape=[None, 2, 3, 3], dtype='float32', name="input")
    paddle.onnx.export(model, output_path, [x_spec], opset_version=11)