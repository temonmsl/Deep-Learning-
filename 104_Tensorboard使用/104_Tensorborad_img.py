from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import os

# 獲取當前腳本文件的目錄
current_dir = os.path.dirname(os.path.abspath(__file__))

# 在當前目錄下創建 "logs" 文件夾
log_dir = os.path.join(current_dir, "logs")

img_path1 = "104_Tensorboard使用/data/train/ants_image/0013035.jpg" 
img_PIL1 = Image.open(img_path1)
img_array1 = np.array(img_PIL1)

img_path2 = "104_Tensorboard使用/data/train/bees_image/17209602_fe5a5a746f.jpg" 
img_PIL2 = Image.open(img_path2)
img_array2 = np.array(img_PIL2)

if not os.path.exists(img_path1):
    print(f"Image path {img_path1} does not exist.")
if not os.path.exists(img_path2):
    print(f"Image path {img_path2} does not exist.")

writer = SummaryWriter(log_dir) 
writer.add_image("test",img_array1,1,dataformats="HWC") # 1 表示該圖片在第1步
writer.add_image("test",img_array2,2,dataformats="HWC") # 2 表示該圖片在第2步                   
writer.close()
print(f"日誌文件已保存到: {log_dir}")