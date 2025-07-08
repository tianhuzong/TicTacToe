import paddle
import paddle.nn as nn

class TicTacToeModel(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2D(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 9)
        self.dropout = nn.Dropout(0.5)  # 添加Dropout层
    
    def forward(self, x):
        x = paddle.transpose(x, [0, 3, 1, 2])
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = paddle.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)  # 在全连接层后应用Dropout
        return self.fc2(x)