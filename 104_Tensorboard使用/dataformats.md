# 為何在使用 NumPy 時需要設置 `dataformats`

在使用 `torch.utils.tensorboard.SummaryWriter` 的 `add_image` 方法時，`dataformats` 參數是用來指定輸入圖像數據的形狀。這可以幫助 TensorBoard 正確解析並顯示圖像。這一設置非常重要，因為不同的數據格式代表著不同的張量維度順序。

## NumPy 圖像格式

當我們使用 NumPy 來表示圖像時，通常使用的數據格式是 `(H, W, C)`，表示：

- **H**：圖片的高度（Height）
- **W**：圖片的寬度（Width）
- **C**：圖片的通道數（Channels，例如 RGB 圖片有 3 個通道）

但是，PyTorch 的張量格式通常是 `(C, H, W)`，即通道數放在最前面。因此，我們需要明確告訴 `add_image` 方法我們所使用的數據格式，以便 TensorBoard 正確解釋數據的維度順序。

## 常見的 `dataformats` 選項

- **HWC**：表示數據的維度順序為 `Height x Width x Channels`，這是 NumPy 圖像數據的常見格式。
- **CHW**：表示數據的維度順序為 `Channels x Height x Width`，這是 PyTorch 張量數據的常見格式。

## 為何 NumPy 需要指定 `dataformats`

當你將 NumPy 數組傳入 `add_image` 方法時，由於 TensorBoard 並不知道你數據的維度順序，因此你需要通過 `dataformats` 明確告知數據格式。這樣才能確保 TensorBoard 正確地解析和顯示圖片。

### 例子說明

假設你有一張圖片被讀取為 NumPy 數組，其形狀為 `(H, W, C)`（高度、寬度、通道）。這種格式是 NumPy 圖像數據的常見格式，因此我們需要使用 `dataformats="HWC"` 來告訴 TensorBoard 數據的順序。

以下是示例代碼：

```python
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

# 讀取圖像並轉換為 numpy 數組
img_path1 = "path/to/your/image.jpg"
img_PIL1 = Image.open(img_path1)
img_array1 = np.array(img_PIL1)

# 創建 SummaryWriter 對象
writer = SummaryWriter("logs")

# 將圖像添加到 TensorBoard，並指定數據格式為 "HWC"
writer.add_image("test_image", img_array1, 1, dataformats="HWC")

# 關閉 writer
writer.close()
```
