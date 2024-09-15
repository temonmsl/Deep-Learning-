import os
from torch.utils.tensorboard import SummaryWriter

# 獲取當前腳本文件的目錄
current_dir = os.path.dirname(os.path.abspath(__file__))

# 在當前目錄下創建 "logs" 文件夾
log_dir = os.path.join(current_dir, "logs")

# 創建 SummaryWriter 實例，指定日誌目錄
writer = SummaryWriter(log_dir)

for i in range(100):
    writer.add_scalar("y=x", 2*i, i)

writer.close()
# Tensorboard 讀日志 : tensorboard --logdir=logs --port=6008 **logs可以替換成絕對路徑，要注意當終端機的path在D:\Deep-Learning時，必須將logs移至此目錄下。
# tensorboard --logdir=D:\Deep-Learning\104_Tensorboard使用\\logs --port=6008
print(f"日誌文件已保存到: {log_dir}")