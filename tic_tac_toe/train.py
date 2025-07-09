import os
import paddle
from tqdm import tqdm
from collections import defaultdict

class TicTacToeTrainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, test_loader=None,
                 checkpoint_dir='checkpoints', device='cpu', 
                 lr_scheduler=None, patience=3, min_lr=1e-5):
        """
        初始化训练器
        
        参数:
            model: 训练模型
            optimizer: 优化器
            loss_fn: 损失函数
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器(可选)
            checkpoint_dir: 检查点保存目录
            device: 训练设备(cpu/gpu)
            lr_scheduler: 学习率调度器 (如ReduceLROnPlateau)
            patience: 在降低学习率前等待的epoch数(没有提升)
            min_lr: 最小学习率下限
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.patience = patience
        self.min_lr = min_lr
        self.no_improve_epochs = 0  # 记录没有提升的epoch数
        self.best_acc = 0.0
        
        # 创建检查点目录
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
    
    def train_epoch(self, epoch, total_epochs):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # 使用tqdm显示进度条
        with tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{total_epochs}', unit='batch') as pbar:
            for data, labels in pbar:
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                outputs = self.model(data)
                loss = self.loss_fn(outputs, labels)
                
                # 计算准确率 - 修改这一行
                predicted = paddle.argmax(outputs, axis=1)  # 只获取预测结果
                correct += (predicted == labels).sum().item()
                total += labels.shape[0]
                
                # 反向传播和优化
                loss.backward()
                self.optimizer.step()
                self.optimizer.clear_grad()
                
                total_loss += loss.item()
                
                # 更新进度条信息
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.train_loader)
        train_acc = 100 * correct / total
        return avg_loss, train_acc
    
    def evaluate(self):
        """评估模型性能"""
        if self.test_loader is None:
            return 0.0, 0.0
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with paddle.no_grad():
            for data, labels in tqdm(self.test_loader, desc='Evaluating', unit='batch'):
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(data)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
                
                # 修改这一行
                predicted = paddle.argmax(outputs, axis=1)
                correct += (predicted == labels).sum().item()
                total += labels.shape[0]
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        
        filename = f'checkpoint_epoch_{epoch}.pdparams'
        if is_best:
            filename = 'best_model.pdparams'
        
        path = os.path.join(self.checkpoint_dir, filename)
        paddle.save(checkpoint, path)
        return path
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = paddle.load(checkpoint_path)
        self.model.set_state_dict(checkpoint['model_state_dict'])
        self.optimizer.set_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch']
    
    def train(self, epochs, start_epoch=0, save_freq=100):
        """
        训练模型
        
        参数:
            epochs: 总训练轮数
            start_epoch: 起始epoch(用于断点续训)
            save_freq: 保存检查点的频率
        """
        best_acc = 0.0
        
        for epoch in range(start_epoch, epochs):
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(epoch, epochs)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # 评估
            test_loss, test_acc = self.evaluate()
            if self.test_loader is not None:
                self.history['test_loss'].append(test_loss)
                self.history['test_acc'].append(test_acc)
                
                # 学习率调度 (新增部分)
                if self.lr_scheduler is not None:
                    if isinstance(self.lr_scheduler, paddle.optimizer.lr.ReduceOnPlateau):
                        self.lr_scheduler.step(test_loss)  # 根据验证损失调整
                    else:
                        self.lr_scheduler.step()  # 常规调整
                
                # 自定义早停和学习率衰减逻辑 (新增)
                if test_acc > best_acc:
                    best_acc = test_acc
                    self.no_improve_epochs = 0
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.no_improve_epochs += 1
                    # 如果连续patience个epoch没有提升，降低学习率
                    if self.no_improve_epochs >= self.patience:
                        current_lr = self.optimizer.get_lr()
                        new_lr = max(current_lr * 0.1, self.min_lr)
                        if new_lr < current_lr:
                            self.optimizer.set_lr(new_lr)
                            print(f"\n降低学习率到 {new_lr:.2e}")
                            self.no_improve_epochs = 0  # 重置计数器
                
                # 更新最佳模型
                if test_acc > best_acc:
                    best_acc = test_acc
                    self.save_checkpoint(epoch, is_best=True)
            
            # 打印epoch结果
            log_msg = (f'Epoch [{epoch+1}/{epochs}] - '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            if self.test_loader is not None:
                log_msg += (f', Test Loss: {test_loss:.4f}, '
                           f'Test Acc: {test_acc:.2f}% (Best: {best_acc:.2f}%)')
            print(log_msg)
            
            # 保存检查点
            if (epoch + 1) % save_freq == 0 or (epoch + 1) == epochs:
                saved_path = self.save_checkpoint(epoch + 1)
                print(f'Checkpoint saved to {saved_path}')
        
        return self.history