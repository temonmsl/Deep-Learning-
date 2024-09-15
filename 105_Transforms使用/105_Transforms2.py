from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


# 傳遞一個字符串 "logs" 作為目錄名，用來存儲日志數據。
writer = SummaryWriter("logs")
# SummaryWriter(logs)
# 期望 logs 是一個之前定義過的變量，並且這個變量指向某個文件路徑。如果 logs 變量未定義，這樣寫會導致錯誤。
img = Image.open("105_Transforms使用\S__4710434.jpg")
print(img)

train_totensor = transforms.ToTensor()
img_tensor = train_totensor(img)
writer.add_image("totensor", img_tensor)

# Normalize
print(img_tensor[0][0][0])
tensor_norm = transforms.Normalize(
    [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
)  # input[channel]=(input[chnnel]-mean[channel])/std[channel]
img_norm = tensor_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("norm", img_norm)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
# PIL數據類型的 img -> resize -> PIL數據類型的 img_resize
img_resize = trans_resize(img)
# PIL 數據類型的 PIL -> totensor -> img_resize tensor
img_resize = train_totensor(img_resize)
print(img_resize.size())  # PIL類型的圖片原始比例為 3×512×512，3通道
writer.add_image("img_resize", img_resize)

# Compose -Resize -2
trans_resize_2 = transforms.Resize(512)  # 512/464 = 1.103 551/500 = 1.102
# PIL類型的 Image -> resize -> PIL類型的 Image -> totensor -> tensor類型的 Image
trans_compose = transforms.Compose(
    [trans_resize_2, train_totensor]
)  # Compose函數中後面一個參數的輸入為前面一個參數的輸出
img_resize_2 = trans_compose(img)
print(img_resize_2.size())

# RandomCrop
trans_Random = transforms.RandomCrop(1080, 1080)  # 随即裁剪成 1080×1080 的
trans_compose2 = transforms.Compose([trans_Random, train_totensor])
for i in range(5):
    img_crop = trans_compose2(img)
    writer.add_image("RandomCrop", img_crop, i)  # 這裡的i是步長
    print(img_crop.size())


writer.add_image("img_resize_2", img_resize_2)
# tensorboard --logdir=logs --port=6009
writer.close()
