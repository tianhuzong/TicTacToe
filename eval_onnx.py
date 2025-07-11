import onnxruntime as ort
import numpy as np

def __predict(onnx_session, input_args):
    return onnx_session.run(None, {"input": input_args})

def process_data(board, player):
    x_channel = np.array([[1 if board[r][c] == 'X' else 0 for c in range(3)] for r in range(3)], dtype=np.float32)
    o_channel = np.array([[1 if board[r][c] == 'O' else 0 for c in range(3)] for r in range(3)], dtype=np.float32)
    if player == 'X':
        input_data = np.stack([x_channel, o_channel], axis=0)  # 通道0=X，通道1=O
    else:
        input_data = np.stack([o_channel, x_channel], axis=0)  # 通道0=O，通道1=X
    input_data = input_data[np.newaxis, ...]
    return input_data

def best_move(board, player):
    model_path = "./models/tictactoe.onnx"
    onnx_session = ort.InferenceSession(model_path)
    input_data = process_data(board, player)
    output_data = __predict(onnx_session, input_data)[0]
    max_index = np.argmax(output_data)
    row, col = divmod(max_index, 3)
    return (row, col)

if __name__ == "__main__":
    print("Best move:", best_move([['O', '', ''], ['', 'X', ''], ['O', '', '']], 'X'))