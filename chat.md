# ChatGPT 對話紀錄：Raspberry Pi 4 影像辨識模型選擇

**問：在 Raspberry Pi 4 上，要運行模型做影像辨識，有哪些模型可以選擇。目前是用 SVM 直接將 image 轉成一維向量進行分類，再幫我列出兩個另外的模型。**

**答：**
在 Raspberry Pi 4 上做影像辨識時，因為硬體資源有限（CPU 與記憶體不像電腦 GPU 那麼強），通常會選擇「輕量化模型 (Lightweight Model)」。

你目前的方法是：
1. 將 image flatten 成一維向量
2. 再用簡單分類器（可能是 SVM / KNN / MLP）

這種方法簡單、速度快，但缺點是：
- 無法保留空間特徵
- 對旋轉、光線、角度較敏感
- 準確率通常不如 CNN 類模型

除了你現在的方法，推薦另外兩個適合 Raspberry Pi 4 的模型：

### 1. MobileNet (最推薦)
* **特點**：專門為手機與嵌入式設備設計，非常適合 Raspberry Pi。速度快、模型小。
* **優點**：推論速度快、精度高於傳統 flatten 方法、支援即時影像辨識、社群資源多。
* **缺點**：比簡單分類器稍微耗 CPU，需要 CNN 架構。
* **建議版本**：MobileNetV2, MobileNetV3 Small (更快)。
* **RPi 效能**：使用 TensorFlow Lite INT8 量化後，可達接近即時辨識。

### 2. EfficientNet Lite
* **特點**：比 MobileNet 更高準確率，Lite 版本專門給 edge device。
* **優點**：精度通常比 MobileNet 高、模型大小仍然不大、適合較複雜分類。
* **缺點**：有時比 MobileNet 慢，Raspberry Pi 長時間運行可能會發熱。
* **建議版本**：EfficientNet-Lite0。
* **RPi 效能**：在 RPi 4 上 EfficientNet-Lite0 有機會比舊版 MobileNet 更快且更準。

---

### 三種方法比較

| 方法 | 速度 | 準確率 | 難度 | RPi 適合度 |
| :--- | :--- | :--- | :--- | :--- |
| **Flatten + 傳統分類器** | 很快 | 低~中 | 最簡單 | 很高 |
| **MobileNet** | 快 | 高 | 中等 | 非常高 |
| **EfficientNet Lite** | 中等 | 很高 | 中等 | 高 |

---

### 建議
* 想要「容易做 + 即時辨識」：選 **MobileNetV2 + TensorFlow Lite**。
* 想要「準確率高」：選 **EfficientNet-Lite0**。
* 想要「超簡單實作」：維持 **Flatten + SVM/KNN**。

如果你之後想做物件偵測（即時框框辨識），可以考慮 **YOLO Tiny** 或 **SSD MobileNet**。

---

# 今日開發進度總結 (Task Summary)

1.  **倉庫遷移與同步**：
    - 從原始倉庫複製代碼，並成功 PUSH 到新的個人倉庫 `https://github.com/penny0411/HW4`。
2.  **模型優化與重新訓練**：
    - 針對原本模型「只會辨識 Paper」的問題，使用 2520 張平衡樣本重新訓練了 SVM 模型，將測試集準確率提升至 **68%**。
    - 修復了 Windows 系統上的 `UnicodeEncodeError` 編碼錯誤。
3.  **新增 MediaPipe 穩定版本**：
    - 實作了 `gesture_recognition_mediapipe.py`，改用 21 個手部關節點進行邏輯辨識。
    - 解決了背景光線干擾問題，並新增了 **"No hand" (無手)** 的即時狀態顯示。
4.  **MobileNetV2 整合**：
    - 為了符合報告要求，新增了 MobileNetV2 的訓練腳本 (`train_mobilenet.py`) 與辨識 Demo (`gesture_recognition_MobileNet.py`)。
5.  **專案結構整理**：
    - 將 `demo/` 資料夾整理為三種不同的辨識方案 (SVM, MediaPipe, MobileNet)。
    - 更新 `README.md`，在最上方加入「快速開始」指令，方便使用者直接執行。
6.  **環境教學**：
    - 說明了 Python 版本對 TensorFlow 的影響，以及如何建立虛擬環境 (venv) 來管理專案。

---

# 2026-05-19 開發進度總結 (Task Summary)

1.  **錯誤辨識與狀態顯示 (Error Detection)**：
    - **MobileNet**：加入了「信心門檻 (Confidence Threshold = 0.8)」。若辨識機率過低或畫面上沒有手，會即時顯示紅色的 **"Error (Low Confidence)"**。
    - **MediaPipe**：統一了錯誤顯示邏輯，將原本的 "No hand" 或 "Unknown" 改為紅色的 **"Error"** 標籤，讓使用者能清楚知道目前處於無效辨識狀態。
2.  **效能優化 (Lag & Latency)**：
    - **跳幀處理 (Frame Skipping)**：實作了跳幀偵測邏輯（MobileNet 每 3 幀預測一次、MediaPipe 每 2 幀處理一次），大幅減少 CPU 計算量，解決了即時畫面卡頓的問題。
    - **解析度降低**：將攝像頭解析度強制設定在 **640x480**，平衡了影像品質與推論速度。
3.  **模型重新訓練 (MobileNetV2 Retraining)**：
    - 成功克服環境限制，在 **Python 3.13** 環境下完成 MobileNetV2 的重新訓練，並更新了 demo 使用的 `.h5` 模型檔案。
4.  **GitHub 自動化同步**：
    - 所有的程式優化與重新訓練後的模型檔案，皆已成功推送到 `https://github.com/penny0411/HW4` 儲存庫中。
