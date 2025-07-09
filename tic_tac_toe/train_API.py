import paddle
from . import TicTacToeModel, TicTacToeDataset
Tmodel = TicTacToeModel()
Tmodel.load_dict(paddle.load("../checkpoints/best_model.pdparams"))
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.001)
optimizer.set_state_dict(Tmodel['optimizer_state_dict'])
model = paddle.Model(Tmodel['model_state_dict'])
model.prepare(
    optimizer=optimizer,
    loss=paddle.nn.CrossEntropyLoss(),
    metrics=paddle.metric.Accuracy(),
)

train_dataset = TicTacToeDataset("data", train=True, test_size=0.2, random_state=42)
test_dataset = TicTacToeDataset("data", train=False, test_size=0.2, random_state=42)

model.fit(train_dataset, epochs=50, batch_size=64, verbose=1)
eval_result = model.evaluate(test_dataset, verbose=1)
print(eval_result)
model.save("checkpoint/test")