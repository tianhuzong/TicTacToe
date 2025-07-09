import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class TicTacToeModel(nn.Layer):
    def __init__(self, dropout_prob=0.3):
        super(TicTacToeModel, self).__init__()
        # 输入形状: [batch_size, 3, 3, 2] (两个通道分别表示X和O的位置)
        self.conv1 = nn.Conv2D(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2D(64, 128, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(256, 9)  # 输出9个可能的移动位置
        
    def forward(self, x):
        # x shape: [batch_size, 2, 3, 3]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(x)
        x = paddle.flatten(x, 1)  # 展平
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x