import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential ,L1Loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)
inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))
loss = L1Loss()  # 默認為 maen
result = loss(inputs, targets)
print(result)

loss = L1Loss(reduction="sum")  # 修改為sum，三個值的差值，然後取和
result = loss(inputs, targets)
print(result)
# MSE損失函數
loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs, targets)
print(result_mse)
# 交叉熵損失函數
x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))  # 1的 batch_size，有三类
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)

# 搭建神經網絡
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
for data in dataloader:
    imgs, targets = data
    outputs = tudui(imgs)
    result_loss = loss(outputs, targets)  # 計算實際輸出與目標輸出的差距
    # backward:嘗試如何調整網路過程中的參數才會導致最終的loss變小(因為是從loss開始推導參數，和網路的順序相反所以叫backward)
    result_loss.backward()  # 計算出來的 loss 值有 backward 方法屬性，反向傳播來計算每個節點的更新的參數。這里查看網絡的屬性 grad 梯度屬性剛開始沒有，反向傳播計算出來後才有，後面優化器會利用梯度優化網絡參數。
    print("ok")
