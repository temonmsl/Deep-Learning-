import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(
    "./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True
)
dataloader = DataLoader(dataset, batch_size=64)


# 定義卷積神經網絡模型
# Tudui 類繼承自 nn.Module，表示我們定義了一個神經網絡模型。
# self.conv1: 定義了一個二維卷積層，卷積核大小為 3x3，輸入通道為 3（因為 CIFAR-10 是 RGB 彩色圖像，每張圖像有 3 個通道），輸出通道為 6（意味著經過卷積後輸出 6 個特征圖）。
# kernel_size=3: 3x3 的卷積核。
# stride=1: 步幅為 1，表示卷積核每次移動 1 個像素。
# padding=0: 無填充，輸出的大小會比輸入小。
# forward: 定義了網絡的前向傳播，輸入 x 會經過卷積操作，並返回卷積後的結果。
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(
            in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0
        )  # 彩色圖像輸入為3層，我們想讓它的輸出為6層，選3 * 3 的卷積

    def forward(self, x):
        x = self.conv1(x)
        return x


# 初始化模型和 TensorBoard 記錄器
# tudui: 創建了一個 Tudui 模型實例。
# writer: 使用 SummaryWriter 記錄數據到 "logs" 文件夾，便於通過 TensorBoard 可視化。
tudui = Tudui()
writer = SummaryWriter("logs")
# 循環處理每個批次的數據
step = 0
# for data in dataloader: 從數據集中加載每一批數據（64 張圖片）。
# imgs, targets = data: imgs 是圖片數據，targets 是對應的標簽。
# tudui(imgs): 將圖片數據傳入模型，通過卷積層計算輸出。
# 打印 imgs.shape 和 output.shape：
# imgs.shape 會是 [64, 3, 32, 32]，表示 64 張 RGB 圖像，每張圖像大小為 32x32。
# output.shape 會是 [64, 6, 30, 30]，表示經過卷積後，輸出 64 張 6 通道的特征圖，大小為 30x30（因為卷積核為 3x3，沒有填充，輸出尺寸減少了 2 個像素）。
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    print(imgs.shape)
    print(output.shape)
    writer.add_images("input", imgs, step)
    output = torch.reshape(
        output, (-1, 3, 30, 30)
    )  # 把原來6個通道拉為3個通道，為了保證所有維度總數不變，其余的分量分到第一個維度中
    writer.add_images("output", output, step)
    step = step + 1
