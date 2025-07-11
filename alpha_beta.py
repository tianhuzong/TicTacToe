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