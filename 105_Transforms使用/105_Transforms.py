from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os

current_path = os.path.dirname(os.path.abspath(__file__))

log_dir = os.path.join(current_path, "logs")
# ① Transforms當成工具箱的話，里面的class就是不同的工具。例如像totensor、resize這些工具。
# ② Transforms拿一些特定格式的圖片，經過Transforms里面的工具，獲得我們想要的結果。
img_path = "105_Transforms使用/data/train/bees_image/16838648_415acd9e3f.jpg"
img = Image.open(img_path)
print(type(img))

writer = SummaryWriter(log_dir)

tensor_trans = transforms.ToTensor()  # 創建 transforms.ToTensor類 的實例化對象
tensor_img = tensor_trans(img)  # 調用 transforms.ToTensor類 的__call__的魔術方法

writer.add_image("tensor_img", tensor_img)
print(type(tensor_img))
print(tensor_img)
writer.close()
