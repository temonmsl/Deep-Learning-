import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(
    "./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True
)
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
        )

    def forward(self, x):
        x = self.model1(x)
        return x


loss = nn.CrossEntropyLoss()  # 交叉熵
tudui = Tudui()
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)  # 隨機梯度下降優化器
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        result_loss = loss(outputs, targets)  # 計算實際輸出與目標輸出的差距
        optim.zero_grad()  # 有調用 : 梯度清零 # 沒有調用 : 梯度會累積
        result_loss.backward()  # 反向傳播，計算損失函數的梯度 # 計算新梯度，並疊加到之前的梯度上
        optim.step()  # 根據梯度，對網絡的參數進行調優 # 使用累積的梯度進行參數更新（結果不正確）
        #print(result_loss)  # 對數據只看了一遍，只看了一輪，所以loss下降不大
        running_loss = running_loss + result_loss
    print(running_loss)
