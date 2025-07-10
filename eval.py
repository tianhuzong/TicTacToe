import paddle
import paddle.nn.functional as F
import numpy as np
from tic_tac_toe import TicTacToeModel

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
        
def check_winner(board):
    """检查是否有玩家获胜"""
    # 检查行
    for row in board:
        if row[0] == row[1] == row[2] != " ":
            return row[0]
    
    # 检查列
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != " ":
            return board[0][col]
    
    # 检查对角线
    if board[0][0] == board[1][1] == board[2][2] != " ":
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != " ":
        return board[0][2]
    
    return None  # 没有获胜者

def is_board_full(board):
    """检查棋盘是否已满"""
    for row in board:
        if " " in row:
            return False
    return True

def evaluate(board):
    """评估当前棋盘状态"""
    winner = check_winner(board)
    if winner == "X":
        return 1
    elif winner == "O":
        return -1
    else:
        return 0  # 平局或未结束

def alpha_beta(board, depth, alpha, beta, maximizing_player):
    """Alpha-Beta 剪枝算法实现"""
    if check_winner(board) or is_board_full(board):
        return evaluate(board)
    
    if maximizing_player:
        max_eval = float('-inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = "X"
                    eval = alpha_beta(board, depth+1, alpha, beta, False)
                    board[i][j] = " "
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
        return max_eval
    else:
        min_eval = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = "O"
                    eval = alpha_beta(board, depth+1, alpha, beta, True)
                    board[i][j] = " "
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
        return min_eval

def find_best_move(board, player):
    """找到最佳移动"""
    best_move = (-1, -1)
    if player == "X":
        best_eval = float('-inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = "X"
                    eval = alpha_beta(board, 0, float('-inf'), float('inf'), False)
                    board[i][j] = " "
                    if eval > best_eval:
                        best_eval = eval
                        best_move = (i, j)
    else:  # player == "O"
        best_eval = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = "O"
                    eval = alpha_beta(board, 0, float('-inf'), float('inf'), True)
                    board[i][j] = " "
                    if eval < best_eval:
                        best_eval = eval
                        best_move = (i, j)
    return best_move

if __name__ == "__main__":
    board = [[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']]
    board2 = [['','',''],['','',''],['','','']]
    player = 'O'
    while True:
        r,w = input('(x,y):').split(',')
        board[int(r)][int(w)] = "X"
        if check_winner(board) is not None or is_board_full(board):
            break
        
        move = predict_move(model, board, player)
        #move = find_best_move(board, "O")
        print(move)
        board[move[0]][move[1]] = "O"
        print('Current Board:')
        display_board(board)
        
        
