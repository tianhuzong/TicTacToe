import paddle
import paddle.nn.functional as F
import numpy as np
from tic_tac_toe import TicTacToeModel
from alpha_beta import check_winner, is_board_full, find_best_move

model = TicTacToeModel()
model.set_state_dict(paddle.load("checkpoints/best_model.pdparams")["model_state_dict"])

def predict_move(model, board, player):
    """
    使用训练好的模型预测下一步最佳移动
    
    参数:
        model: 训练好的模型
        board: 当前棋盘状态 (3x3的列表，包含'X', 'O'或' ')
        player: 当前玩家 ('X'或'O')
    
    返回:
        (row, col): 最佳移动的位置
    """
    # 准备输入数据
    x_channel = np.array([[1 if board[r][c] == 'X' else 0 for c in range(3)] for r in range(3)], dtype=np.float32)
    o_channel = np.array([[1 if board[r][c] == 'O' else 0 for c in range(3)] for r in range(3)], dtype=np.float32)
    if player == 'X':
        input_data = np.stack([x_channel, o_channel], axis=0)  # 通道0=X，通道1=O
    else:
        input_data = np.stack([o_channel, x_channel], axis=0)  # 通道0=O，通道1=X
    input_data = paddle.to_tensor(input_data[np.newaxis, ...], dtype='float32')  # 添加batch维度
    
    # 预测
    model.eval()
    with paddle.no_grad():
        outputs = model(input_data)
    
    # 获取所有可能的移动的概率
    probs = F.softmax(outputs[0]).numpy()
    
    # 排除已经被占据的位置
    for r in range(3):
        for c in range(3):
            if board[r][c] != ' ':
                probs[r*3 + c] = 0
    
    # 选择概率最高的移动
    best_move_idx = np.argmax(probs)
    row, col = best_move_idx // 3, best_move_idx % 3
    
    return row, col



def display_board(board):
    """打印棋盘"""
    for i in range(3):
        print(" | ".join(board[i]))
        if i < 2:
            print("---------")
        


if __name__ == "__main__":
    board = [[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']]
    board2 = [['','',''],['','',''],['','','']]
    player = 'O'
    board[1][1] = "X"
    while True:
        #r,w = input('(x,y):').split(',')
        
        if check_winner(board) is not None or is_board_full(board):
            break
        
        #move = predict_move(model, board, player)
        move = find_best_move(board, player)
        print(move)
        board[move[0]][move[1]] = player
        print('Current Board:')
        display_board(board)
        player = 'X' if player == 'O' else 'O'
        
        
