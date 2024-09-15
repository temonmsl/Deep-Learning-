import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# 定義數據集轉換
transform = transforms.Compose(
    [
        transforms.Resize(256),  # 將圖像縮放到 256x256 像素
        transforms.RandomCrop(224),  # 隨機裁剪為 224x224 像素
        transforms.ToTensor(),  # 將圖像轉換為 Tensor 格式
    ]
)

# 創建 CIFAR10 數據集加載器
train_set = torchvision.datasets.CIFAR10(
    root="./dataset",  # 數據集根目錄
    train=True,  # train=True是訓練集，train=False是測試集
    transform=transform,  # 應用數據集轉換
    download=True,  # 如果數據集不存在則下載
)
test_set = torchvision.datasets.CIFAR10(
    root="./dataset",  # 數據集根目錄
    train=True,  # train=True是訓練集，train=False是測試集
    transform=transform,  # 應用數據集轉換
    download=True,  # 如果數據集不存在則下載
)

writer = SummaryWriter("logs")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)
    print(img.size())

writer.close()
