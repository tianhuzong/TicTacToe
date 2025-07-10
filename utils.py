import numpy as np
import random
import os
from tqdm import tqdm

class TicTacToeDataGenerator:
    def __init__(self):
        pass
    
    def _is_winner(self, board, player):
        """检查是否有玩家获胜"""
        # 检查行
        for row in range(3):
            if all([board[row][col] == player for col in range(3)]):
                return True
        # 检查列
        for col in range(3):
            if all([board[row][col] == player for row in range(3)]):
                return True
        # 检查对角线
        if all([board[i][i] == player for i in range(3)]):
            return True
        if all([board[i][2-i] == player for i in range(3)]):
            return True
        return False

    def _is_board_full(self, board):
        """检查棋盘是否已满"""
        return all(all(cell != ' ' for cell in row) for row in board)

    def _evaluate(self, board):
        """评估棋盘状态"""
        if self._is_winner(board, 'X'):
            return 1
        elif self._is_winner(board, 'O'):
            return -1
        else:
            return 0

    def _alpha_beta(self, board, depth, alpha, beta, maximizing_player):
        """Alpha-Beta剪枝算法"""
        score = self._evaluate(board)
        
        if score != 0 or self._is_board_full(board):
            return score
        
        if maximizing_player:
            max_eval = -float('inf')
            for row in range(3):
                for col in range(3):
                    if board[row][col] == ' ':
                        board[row][col] = 'X'
                        eval = self._alpha_beta(board, depth+1, alpha, beta, False)
                        board[row][col] = ' '
                        max_eval = max(max_eval, eval)
                        alpha = max(alpha, eval)
                        if beta <= alpha:
                            break
            return max_eval
        else:
            min_eval = float('inf')
            for row in range(3):
                for col in range(3):
                    if board[row][col] == ' ':
                        board[row][col] = 'O'
                        eval = self._alpha_beta(board, depth+1, alpha, beta, True)
                        board[row][col] = ' '
                        min_eval = min(min_eval, eval)
                        beta = min(beta, eval)
                        if beta <= alpha:
                            break
            return min_eval

    def _get_best_move(self, board, player):
        """获取最佳移动"""
        best_score = -float('inf') if player == 'X' else float('inf')
        best_move = (-1, -1)
        
        for row in range(3):
            for col in range(3):
                if board[row][col] == ' ':
                    board[row][col] = player
                    score = self._alpha_beta(board, 0, -float('inf'), float('inf'), player == 'O')
                    board[row][col] = ' '
                    
                    if player == 'X' and score > best_score:
                        best_score = score
                        best_move = (row, col)
                    elif player == 'O' and score < best_score:
                        best_score = score
                        best_move = (row, col)
        return best_move

    def generate_sample(self):
        """生成单个样本"""
        board = [[' ' for _ in range(3)] for _ in range(3)]
        current_player = random.choice(['X', 'O'])
        
        # 随机进行几步
        for _ in range(random.randint(0, 4)):
            if self._is_board_full(board) or self._evaluate(board) != 0:
                break
            empty_cells = [(r, c) for r in range(3) for c in range(3) if board[r][c] == ' ']
            if not empty_cells:
                break
            row, col = random.choice(empty_cells)
            board[row][col] = current_player
            current_player = 'O' if current_player == 'X' else 'X'
        
        # 如果游戏已结束，返回None
        if self._is_board_full(board) or self._evaluate(board) != 0:
            return None
        
        # 获取最佳移动
        best_row, best_col = self._get_best_move(board, current_player)
        
        # 创建输入张量
        x_channel = np.array([[1 if board[r][c] == 'X' else 0 for c in range(3)] for r in range(3)], dtype=np.float32)
        o_channel = np.array([[1 if board[r][c] == 'O' else 0 for c in range(3)] for r in range(3)], dtype=np.float32)

        if current_player == 'X':
            input_data = np.stack([x_channel, o_channel], axis=0)  # 通道0=X, 通道1=O
        else:
            input_data = np.stack([o_channel, x_channel], axis=0)  # 通道0=O, 通道1=X
        
        # 创建标签
        label = best_row * 3 + best_col
        
        return input_data, label

    def generate_and_save(self, num_samples, save_dir='data'):
        """生成并保存数据"""
        os.makedirs(save_dir, exist_ok=True)
        
        data_file = os.path.join(save_dir, 'tic_tac_toe_data.npy')
        label_file = os.path.join(save_dir, 'tic_tac_toe_labels.npy')
        
        data = []
        labels = []
        
        print(f"Generating {num_samples} training samples...")
        
        # 生成比需求多20%的样本以补偿无效样本
        total_to_generate = int(num_samples * 1.2)
        generated_count = 0
        
        with tqdm(total=num_samples, desc="Generating samples") as pbar:
            while generated_count < num_samples:
                sample = self.generate_sample()
                if sample is not None:
                    input_data, label = sample
                    data.append(input_data)
                    labels.append(label)
                    generated_count += 1
                    pbar.update(1)
                
                # 安全措施，防止无限循环
                if len(data) > total_to_generate and generated_count < num_samples:
                    print(f"Warning: Only generated {generated_count} valid samples (requested {num_samples})")
                    break
        
        # 保存数据
        np.save(data_file, np.array(data))
        np.save(label_file, np.array(labels))
        print(f"Data saved to {data_file} and {label_file}")
        
if __name__ == "__main__":
    generator = TicTacToeDataGenerator()
    generator.generate_and_save(num_samples=5000)