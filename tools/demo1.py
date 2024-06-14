import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


# 自定义数据集
class SimpleDataset(Dataset):
    def __init__(self, length=10):
        self.length = length

    def __getitem__(self, index):
        # NumPy 随机操作
        random_value = np.random.rand()

        # 获取 worker_info 仅在子进程中可用
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
        else:
            worker_id = -1  # 主进程

        # 返回随机值和 worker_id
        return random_value, worker_id

    def __len__(self):
        return self.length


# 不设置 worker_init_fn 的 DataLoader
dataloader = DataLoader(SimpleDataset(), batch_size=1, num_workers=4)

# 检查每个 worker 生成的数据
results = {}

for batch in dataloader:
    value, worker_id = batch
    value, worker_id = value.item(), worker_id.item()

    if worker_id not in results:
        results[worker_id] = []
    results[worker_id].append(value)

# 打印结果
for worker_id, values in results.items():
    print(f"Worker {worker_id}, Random Values: {values}")