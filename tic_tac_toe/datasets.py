import paddle
from paddle.io import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split

class TicTacToeDataset(Dataset):
    def __init__(self, data_dir='data', train=True, test_size=0.2, random_state=42):
        """
        初始化数据集
        
        参数:
            data_dir: 数据目录
            train: 是否为训练集
            test_size: 测试集比例
            random_state: 随机种子
        """
        super(TicTacToeDataset, self).__init__()
        
        # 加载完整数据
        data = np.load(os.path.join(data_dir, 'tic_tac_toe_data.npy'))
        labels = np.load(os.path.join(data_dir, 'tic_tac_toe_labels.npy'))
        
        # 分割训练集和测试集
        train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, test_size=test_size, random_state=random_state
        )
        
        # 根据train参数选择数据集
        if train:
            self.data = train_data
            self.labels = train_labels
        else:
            self.data = test_data
            self.labels = test_labels
        
        # 转换为paddle tensor
        self.data = paddle.to_tensor(self.data, dtype='float32')
        self.labels = paddle.to_tensor(self.labels, dtype='int64')
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.data)

def get_data_loaders(data_dir='data', batch_size=64, test_size=0.2, random_state=42):
    """
    获取训练和测试DataLoader
    
    参数:
        data_dir: 数据目录
        batch_size: 批量大小
        test_size: 测试集比例
        random_state: 随机种子
        
    返回:
        train_loader, test_loader
    """
    # 创建训练集和测试集
    train_dataset = TicTacToeDataset(data_dir, train=True, test_size=test_size, random_state=random_state)
    test_dataset = TicTacToeDataset(data_dir, train=False, test_size=test_size, random_state=random_state)
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# 使用示例
if __name__ == "__main__":
    # 获取DataLoader
    train_loader, test_loader = get_data_loaders()
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # 检查一个批次的数据
    for batch_idx, (data, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx} - data shape: {data.shape}, labels shape: {labels.shape}")
        break