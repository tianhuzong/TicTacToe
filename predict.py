import paddle
import paddle.nn.functional as F
import numpy as np
from tic_tac_toe import TicTacToeModel
class TicTacToeAgent:
    def __init__(self, model_path=None, dropout_prob=0.3):
        # 初始化模型
        self.model = TicTacToeModel(dropout_prob)
        if model_path:
            # 加载预训练模型
            self.model.set_state_dict(paddle.load(model_path)["model_state_dict"])
        self.model.eval()  # 推理模式
        
        # 难度参数映射 (难度级别 -> 温度参数)
        # 温度越高，输出越随机（难度越低）；温度越低，输出越确定（难度越高）
        self.difficulty_mapping = {
            1: 2.0,   # 简单：高随机性
            2: 1.0,   # 较易：中等随机性
            3: 0.5,   # 中等：低随机性
            4: 0.2,   # 困难：极低随机性
            5: 0.0   # 专家：几乎无随机性（贪婪选择）
        }
        
    def preprocess_board(self, board, current_player):
        """
        将棋盘转换为模型输入格式
        
        参数:
        board (list[list[str]]): 3x3棋盘，空位置为''，玩家为'X'/'O'
        current_player (str): 当前要落子的玩家 ('X'或'O')
        
        返回:
        paddle.Tensor: 形状为[1, 2, 3, 3]的输入张量
        """
        # 初始化两个通道 (X和O)
        x_channel = paddle.zeros((1, 3, 3), dtype='float32')
        o_channel = paddle.zeros((1, 3, 3), dtype='float32')
        
        # 填充通道数据
        for i in range(3):
            for j in range(3):
                if board[i][j] == 'X':
                    x_channel[0, i, j] = 1.0
                elif board[i][j] == 'O':
                    o_channel[0, i, j] = 1.0
        
        # 组合通道 (X通道在前，O通道在后)
        if current_player == 'X':
            input_tensor = paddle.concat([x_channel, o_channel], axis=0)  # 通道0=X，通道1=O
        else:
            input_tensor = paddle.concat([o_channel, x_channel], axis=0)  # 通道0=O，通道1=X
        #input_tensor = paddle.concat([x_channel, o_channel], axis=0)
        # 增加批次维度
        input_tensor = paddle.unsqueeze(input_tensor, axis=0)
        
        return input_tensor
    
    def predict_move(self, board, current_player, difficulty=3):
        """
        预测下一步落子位置，支持难度控制
        
        参数:
        board (list[list[str]]): 3x3棋盘
        current_player (str): 当前玩家 ('X'或'O')
        difficulty (int): 难度级别 (1-5)
        
        返回:
        tuple: 最佳落子位置 (row, col)
        """
        # 检查难度级别有效性
        if difficulty not in self.difficulty_mapping:
            raise ValueError(f"难度级别必须为1-5，当前为{difficulty}")
        
        # 获取温度参数
        temperature = self.difficulty_mapping[difficulty]
        
        # 检查棋盘是否有可用位置
        available_moves = []
        for i in range(3):
            for j in range(3):
                if board[i][j] == '':
                    available_moves.append((i, j))
        
        if not available_moves:
            raise ValueError("棋盘已满，无法落子")
        
        # 预处理棋盘
        input_tensor = self.preprocess_board(board, current_player)
        
        # 模型预测
        with paddle.no_grad():
            logits = self.model(input_tensor)  # 形状: [1, 9]
        
        for i in range(3):
            for j in range(3):
                if board[i][j] != '':
                    idx = i * 3 + j
                    logits[0][idx] = -float('inf')  # 设为负无穷，确保不会被选中
            
        # 温度=0时直接选择logits最大的位置
        if temperature == 0:
            selected_idx = paddle.argmax(logits[0]).item()
        else:
            # 将logits转换为概率分布，并应用温度调整
            probs = F.softmax(logits / temperature, axis=1).numpy()[0]  # 应用温度
            
            # 过滤非法位置（已落子的位置概率设为0）
            for i in range(3):
                for j in range(3):
                    if board[i][j] != '':
                        idx = i * 3 + j
                        probs[idx] = 0.0
            
            # 归一化概率（确保合法位置概率和为1）
            probs = probs / np.sum(probs)
            
            # 根据概率分布采样落子位置
            move_indices = np.arange(9)
            selected_idx = np.random.choice(move_indices, p=probs)
        
        # 转换为行列坐标
        row = selected_idx // 3
        col = selected_idx % 3
        
        return (row, col)
    
def alpha_beta_best_move(board, player):
    """
    使用Alpha-Beta剪枝算法计算井字棋最佳移动
    
    参数:
        board: 3x3列表，包含 'X', 'O' 或 ' ' (空格)
        player: 当前玩家 ('X' 或 'O')
    
    返回:
        (row, col): 最佳移动位置
    """
    def evaluate(b):
        """评估棋盘状态：X胜返回1，O胜返回-1，平局返回0"""
        # 检查行/列/对角线
        lines = [
            # 行
            [b[0][0], b[0][1], b[0][2]],
            [b[1][0], b[1][1], b[1][2]],
            [b[2][0], b[2][1], b[2][2]],
            # 列
            [b[0][0], b[1][0], b[2][0]],
            [b[0][1], b[1][1], b[2][1]],
            [b[0][2], b[1][2], b[2][2]],
            # 对角线
            [b[0][0], b[1][1], b[2][2]],
            [b[0][2], b[1][1], b[2][0]]
        ]
        for line in lines:
            if line.count('X') == 3:
                return 1
            if line.count('O') == 3:
                return -1
        return 0

    def is_full(b):
        """检查棋盘是否已满"""
        return all(cell != ' ' for row in b for cell in row)

    def alphabeta(b, depth, alpha, beta, maximizing_player):
        score = evaluate(b)
        
        # 终止条件
        if score != 0:  # 有玩家获胜
            return score
        if is_full(b):   # 平局
            return 0
            
        if maximizing_player:
            max_eval = -float('inf')
            for i in range(3):
                for j in range(3):
                    if b[i][j] == '':
                        b[i][j] = 'X'
                        eval = alphabeta(b, depth+1, alpha, beta, False)
                        b[i][j] = ''
                        max_eval = max(max_eval, eval)
                        alpha = max(alpha, eval)
                        if beta <= alpha:
                            break
            return max_eval
        else:
            min_eval = float('inf')
            for i in range(3):
                for j in range(3):
                    if b[i][j] == '':
                        b[i][j] = 'O'
                        eval = alphabeta(b, depth+1, alpha, beta, True)
                        b[i][j] = ''
                        min_eval = min(min_eval, eval)
                        beta = min(beta, eval)
                        if beta <= alpha:
                            break
            return min_eval

    # 寻找最佳移动
    best_val = -float('inf') if player == 'X' else float('inf')
    best_move = (-1, -1)
    
    for i in range(3):
        for j in range(3):
            if board[i][j] == '':
                board[i][j] = player
                move_val = alphabeta(board, 0, -float('inf'), float('inf'), player == 'O')
                board[i][j] = ' '
                
                if player == 'X' and move_val > best_val:
                    best_val = move_val
                    best_move = (i, j)
                elif player == 'O' and move_val < best_val:
                    best_val = move_val
                    best_move = (i, j)
    
    return best_move

if __name__ == "__main__":
    # 初始化AI代理（假设已训练好模型）
    agent = TicTacToeAgent(model_path='checkpoints/best_model.pdparams')
    
    # 示例棋盘
    board = [
        ['X', '', 'X'],
        ['', 'O', 'O'],
        ['', '', '']
    ]
    
    # 不同难度的预测结果
    for difficulty in [1, 3, 4, 5]:
        move = agent.predict_move(board, 'O', difficulty=difficulty)
        print(f"难度级别{difficulty}的预测落子: 行{move[0]+1}, 列{move[1]+1}")
    best_move = alpha_beta_best_move(board, 'O')
    print(f"best_move: 行{best_move[0]+1}, 列{best_move[1]+1}")