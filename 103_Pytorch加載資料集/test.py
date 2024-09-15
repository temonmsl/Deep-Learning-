from torch.utils.data import Dataset
from PIL import Image
import os

#定義 MyData 類繼承Dataset類：
class mydata(Dataset):
    def __init__(self,rootdir,rootlab):
        self.rootdir = rootdir
        self.rootlab = rootlab
        self.path = os.path.join(self.rootdir,self.rootlab)
        self.imglist = os.listdir(self.path)    

#__getitem__ 方法：
    def __getitem__(self, index):
        self.imgname = self.imglist[index]
        self.imgpath = os.path.join(self.rootdir,self.rootlab,self.imgname)
        openimg = Image.open(self.imgpath)
        lable = self.rootlab
        return openimg, lable

#__len__ 方法：
    def __len__(self):
        return len(self.imglist)
rootdir = '103/dataset/hymenoptera_data/hymenoptera_data/train'
antlab = 'ants'
beelab = 'bees'

#創實例
antdata = mydata(rootdir,antlab)
beedata = mydata(rootdir,beelab)
# 打印螞蟻數據集的樣本數量
print(len(antdata))

# 打印蜜蜂數據集的樣本數量
print(len(beedata))

traindata = antdata + beedata
print(len(traindata))