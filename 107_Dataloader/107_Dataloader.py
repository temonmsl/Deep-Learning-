import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 創建 CIFAR10 數據集加載器
test_data = torchvision.datasets.CIFAR10(
    root="./dataset",  # 數據集根目錄
    train=False,  # train=True是訓練集，train=False是測試集
    transform=torchvision.transforms.ToTensor(),  # 應用數據集轉換
    download=True,  # 如果數據集不存在則下載
)
# batch_size=4 使得 img0, target0 = dataset[0]、img1, target1 = dataset[1]、img2, target2 = dataset[2]、img3, target3 = dataset[3]，然后这四个数据作为Dataloader的一个返回
test_loader = DataLoader(
    dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False
)
img, target = test_data[0]
print(img.shape)
print(img)

# 用for循環取出DataLoader打包好的四個數據
writer = SummaryWriter("logs")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = (
            data  # 每個data都是由4張圖片組成，imgs.size 為 [4,3,32,32]，四張32×32圖片三通道，targets由四個標簽組成
        )
        writer.add_images("Epoch：{}".format(epoch), imgs, step)
        step = step + 1

writer.close()
