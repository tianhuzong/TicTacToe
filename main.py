import paddle
from paddle.io import DataLoader

from tic_tac_toe import TicTacToeDataset, TicTacToeModel, TicTacToeTrainer, get_data_loaders

# 1. 初始化组件
model = TicTacToeModel(dropout_prob=0.3)
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.001)
loss_fn = paddle.nn.CrossEntropyLoss()

# 2. 准备数据
train_loader, test_loader = get_data_loaders(
    data_dir='data',      # 数据目录
    batch_size=64,        # 批量大小
    test_size=0.2,        # 测试集比例(20%)
    random_state=42       # 随机种子(确保可重复性)
)

lr_scheduler = paddle.optimizer.lr.ReduceOnPlateau(
    learning_rate=optimizer.get_lr(),
    mode='min',  # 监控验证损失
    factor=0.5,  # 学习率衰减因子
    patience=3,  # 容忍epoch数
    verbose=True  # 打印调整信息
)

# 3. 创建训练器
trainer = TicTacToeTrainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    train_loader=train_loader,
    test_loader=test_loader,
    checkpoint_dir='checkpoints',
    device='gpu' if paddle.device.is_compiled_with_cuda() else 'cpu',
    lr_scheduler=lr_scheduler
)

# 4. 训练模型 (从头开始)
history = trainer.train(epochs=1000)

#last_epoch = trainer.load_checkpoint('checkpoints/checkpoint_epoch_100.pdparams')
#history = trainer.train(epochs=1000, start_epoch=last_epoch)