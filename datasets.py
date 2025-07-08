import random
import numpy as np
import paddle
from paddle.io import Dataset
from collections import defaultdict

class TicTacToeDataset(Dataset):
    def __init__(self, num_samples=10000):
        super(TicTacToeDataset, self).__init__()
        self.num_samples = num_samples
        self.states, self.actions = self._generate_dataset()
        
    def _generate_dataset(self):
        """生成训练数据集"""
        dataset = defaultdict(list)
        
        while len(dataset['states']) < self.num_samples:
            # 初始化空棋盘
            board = [[' ' for _ in range(3)] for _ in range(3)]
            current_player = 'X' if random.random() > 0.5 else 'O'
            
            # 随机生成游戏状态（不完整对局）
            for _ in range(random.randint(0, 8)):
                try:
                    # 使用alpha-beta获取最佳落子
                    row, col = self.alpha_beta_tic_tac_toe(board, current_player)
                    board[row][col] = current_player
                    # 保存状态和对应的最佳落子
                    dataset['states'].append(self._encode_state(board, current_player))
                    dataset['actions'].append(row * 3 + col)  # 展平为0-8
                    current_player = 'O' if current_player == 'X' else 'X'
                except RuntimeError:  # 游戏结束
                    break
        
        return np.array(dataset['states']), np.array(dataset['actions'])
    
    def _encode_state(self, board, player):
        """将棋盘状态编码为数值矩阵"""
        state = np.zeros((3, 3, 3), dtype=np.float32)
        
        for i in range(3):
            for j in range(3):
                if board[i][j] == 'X':
                    state[i, j, 0] = 1
                elif board[i][j] == 'O':
                    state[i, j, 1] = 1
        
        # 第三通道表示当前玩家
        state[:, :, 2] = 1 if player == 'X' else 0
        return state
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        # 转换为paddle tensor
        state = paddle.to_tensor(self.states[idx], dtype='float32')
        action = paddle.to_tensor(self.actions[idx], dtype='int64')
        return state, action
