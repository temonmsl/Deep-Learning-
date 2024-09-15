import torchvision
from torch import nn
from torchvision.models import VGG16_Weights

dataset = torchvision.datasets.CIFAR10(
    "./dataset",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
# 從 torchvision 0.13 版本開始，模型中的 pretrained 參數已經被棄用（即將來可能會移除），應該使用 weights 參數來代替。
# vgg16_true = torchvision.models.vgg16(
#     pretrained=True
# )  # 下載卷積層對應的參數是多少、池化層對應的參數時多少，這些參數時ImageNet訓練好了的

# 使用 'weights' 參數代替 'pretrained'
# vgg16_true = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

# 如果想加載模型的最新預訓練權重，可以使用 VGG16_Weights.DEFAULT。
# 使用最新的預訓練權重
vgg16_true = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
# 網絡模型添加
vgg16_true.add_module(
    "add_linear", nn.Linear(1000, 10)
)  # 在VGG16後面添加一個線性層，使得輸出為適應CIFAR10的輸出，CIFAR10需要輸出10個種類
print("vgg16_true", vgg16_true)
vgg16_true.classifier.add_module(
    "add_linear", nn.Linear(1000, 10)
)  # 在VGG16中classifie裡面添加一個線性層，可改為vgg16_true.classifier.add_module
print("gg16_true.classifier :", vgg16_true)

# 網絡模型修改
vgg16_false = torchvision.models.vgg16(weights=None)  # 沒有預訓練的參數
print("vgg16_false", vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print("vgg16_false.classifier", vgg16_false)
