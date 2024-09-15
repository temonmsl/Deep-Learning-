import torchvision

train_set = torchvision.datasets.CIFAR10(
    root="./dataset", train=True, download=True
)  # root為存放數據集的相對路線
test_set = torchvision.datasets.CIFAR10(
    root="./dataset", train=False, download=True
)  # train=True是訓練集，train=False是測試集

print(test_set[0])  # 輸出的3是target
print(test_set.classes)  # 測試數據集中有多少種

img, target = test_set[0]  # 分別獲得圖片、target
print(img)
print(target)

print(test_set.classes[target])  # 3號target對應的種類
img.show()
