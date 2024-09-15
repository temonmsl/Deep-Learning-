from torch.utils.data import Dataset, ConcatDataset
#Dataset：PyTorch 提供的用於創建自定義數據集的基類。
#ConcatDataset：用於合併多個數據集的 PyTorch 工具。
from PIL import Image
import os

class MyData(Dataset):     
    def __init__(self, root_dir, label_dir) :
        # 初始化方法，在創建類的實例時自動調用
        self.root_dir = root_dir  # 保存根目錄路徑
        self.label_dir = label_dir  # 保存標籤目錄名稱
        self.path = os.path.join(self.root_dir, self.label_dir)  # 組合完整的數據路徑
        self.img_path = os.listdir(self.path)  # 獲取指定路徑下所有文件的列表
        
    def __getitem__(self, idx):
        # 根據索引獲取單個數據項的方法
        img_name = self.img_path[idx]  # 獲取指定索引的圖片文件名
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)  # 組合完整的圖片文件路徑
        img = Image.open(img_item_path)  # 打開圖片文件
        label = self.label_dir  # 使用目錄名作為標籤
        return img, label  # 返回圖片對象和對應的標籤
    
    def __len__(self):
        # 返回數據集中的樣本數量
        return len(self.img_path)  # 返回圖片文件列表的長度
    
# 設置數據的根目錄路徑
root_dir = "103/dataset/hymenoptera_data/hymenoptera_data/train"

# 設置螞蟻和蜜蜂圖片的標籤目錄名
ants_label_dir = "ants"
bees_label_dir = "bees"

# 創建螞蟻數據集實例
ants_dataset = MyData(root_dir, ants_label_dir)

# 創建蜜蜂數據集實例
bees_dataset = MyData(root_dir, bees_label_dir)

# 打印螞蟻數據集的樣本數量
print(len(ants_dataset))

# 打印蜜蜂數據集的樣本數量
print(len(bees_dataset))

# 使用 ConcatDataset 合併螞蟻和蜜蜂數據集
train_dataset = ConcatDataset([ants_dataset, bees_dataset])

# 打印合併後的數據集總樣本數量
print(len(train_dataset))

# 從合併後的數據集中獲取索引為 200 的樣本
img, label = train_dataset[200]

# 打印獲取的樣本的標籤
print("label:", label)

# 顯示獲取的圖片
img.show()