import torch
from torch import nn


class Jiang(nn.Module):
    def __init__(self):
        super(Jiang, self).__init__()  # 繼承父類的初始化

    def forward(self, input):  # 將forward函數進行重寫
        output = input + 1
        return output


jiang = Jiang()  # 創建 Jiang 類的實例 jiang。

x = torch.tensor(1.0)  # 創建一個值為 1.0 的tensor
output = jiang(
    x
)  # 將張量 x 輸入到模型 jiang  中，模型的 forward 方法會被調用，計算結果存儲在 output 變量中。
print(output)
