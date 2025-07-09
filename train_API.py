import paddle
from tic_tac_toe import TicTacToeModel, TicTacToeDataset
Tmodel = TicTacToeModel()
ckpt = paddle.load("./checkpoints/best_model.pdparams")
Tmodel.load_dict(ckpt["model_state_dict"])
optimizer = paddle.optimizer.Adam(parameters=Tmodel.parameters(), learning_rate=0.001)
optimizer.set_state_dict(ckpt['optimizer_state_dict'])
model = paddle.Model(Tmodel)
model.prepare(
    optimizer=optimizer,
    loss=paddle.nn.CrossEntropyLoss(),
    metrics=paddle.metric.Accuracy(),
)

train_dataset = TicTacToeDataset("data", train=True, test_size=0.2, random_state=42)
test_dataset = TicTacToeDataset("data", train=False, test_size=0.2, random_state=42)

model.fit(train_dataset, epochs=100, batch_size=64, verbose=1)
eval_result = model.evaluate(test_dataset, verbose=1)
print(eval_result)
model.save("checkpoint/test")