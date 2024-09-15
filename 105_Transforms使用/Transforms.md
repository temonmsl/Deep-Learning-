### 關鍵步驟說明：

#### 1. 加載圖片：

使用 `Image.open(img_path)` 將圖片加載為 PIL 圖像對象。PIL 是一個非常流行的 Python 圖像處理庫，可以處理多種格式的圖片。加載後的 `img` 類型為 `PIL.Image.Image`。

#### 2. 轉換為張量：

使用 `transforms.ToTensor()` 類將 PIL 圖像轉換為 PyTorch 張量。`ToTensor()` 是 `torchvision.transforms` 模塊中的一個工具，它可以將 PIL 圖像（或 NumPy 數組）轉換為形狀為 `(C, H, W)` 的 PyTorch 張量，且張量的數值範圍從 `[0, 255]` 變為 `[0.0, 1.0]`。

#### 3. 魔術方法 `__call__`：

當你調用 `tensor_trans(img)` 時，本質上是調用了 `transforms.ToTensor` 的 `__call__` 方法。它會自動將 PIL 圖像轉換為 PyTorch 張量。
ToTensor 中的`___call__`\_方法可以使實例化對象變成可調用對象，實例化對象可接受一個圖片

- **call**:`tensor_img = tensor_trans(img)`等價於`tensor_img = tensor_trans.__call__(img)`

#### 4. 輸出張量：

`tensor_img` 是一個 3D 的 PyTorch 張量，其形狀為 `(C, H, W)`，其中：

- **C**：通道數（如 RGB 圖像有 3 個通道）。
- **H**：圖像高度。
- **W**：圖像寬度。

### 需要 Tensor 數據類型原因

① Tensor 有一些屬性，比如反向傳播、梯度等屬性，它包裝了神經網絡需要的一些屬性。
