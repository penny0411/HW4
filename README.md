# 智慧物聯網 HW4 - 剪刀石頭布 手勢辨識系統

## 📝 專案總結 (Project Summary)
本專案為智慧物聯網的第四次作業，主要實作一個可部署於 **Raspberry Pi 4** 上的「剪刀、石頭、布」手勢辨識系統。
為達成作業要求（尋找並修改兩種模型架構，並比較其差異與呈現效能指標），本專案除了原有的攝像頭測試外，共實作了以下三種辨識版本：

1. **MediaPipe 版本 (`gesture_recognition_mediapipe.py`)**
   - 使用 Google 的 MediaPipe 框架提取手部關鍵節點進行手勢判斷，為目前辨識最穩定、準確度最高的版本。
2. **SVM 版本 (`gesture_recognition_svm.py`)**
   - 傳統機器學習方法，基於支援向量機 (Support Vector Machine) 進行影像分類。
   - 包含自定義的訓練腳本 (`train_svm.py`) 與測試評估腳本 (`test_svm.py`)。
3. **MobileNet 版本 (`gesture_recognition_MobileNet.py`)**
   - 採用輕量級卷積神經網路 (CNN) 模型 MobileNet，專為移動與邊緣設備設計。
   - 包含模型訓練 (`train_mobilenet.py`) 與完整的效能評估腳本 (`test_mobilenet.py`)。

本專案同時實作了測試腳本，能夠自動計算並輸出 **Accuracy (準確率), Precision (精確率), Recall (召回率), F1-score** 等評估指標，以利於期末報告中的模型效能比較與分析。

---

## 📂 目錄結構與主要檔案

- **`demo/`**: 存放即時辨識、效能測試與攝像頭腳本
  - `carema.py`: 攝像頭基本畫面測試腳本
  - `gesture_recognition_*.py`: 三種不同架構的即時手勢辨識腳本
  - `test_svm.py` & `test_mobilenet.py`: 針對測試集進行預測，並輸出各項模型效能指標 (Metrics)
  - 模型權重檔 (`.pkl`, `.h5`) 
- **`train/`**: 存放模型訓練腳本
  - `train_svm.py`: 讀取資料集並訓練 SVM 模型
  - `train_mobilenet.py`: 讀取資料集並訓練 MobileNet 模型
- **`dataset/`**: 剪刀、石頭、布的影像資料集

---

## 🚀 快速開始 (Quick Start)

### 1. 安裝環境依賴
建議在虛擬環境中執行以下指令以安裝所需套件：
```bash
pip install -r demo/requirements.txt
```

### 2. 執行攝像頭測試
```bash
python demo/carema.py
```

### 3. 模型效能測試 (計算 Accuracy, Precision, Recall, F1-score)
```bash
# 測試 SVM 模型
python demo/test_svm.py

# 測試 MobileNet 模型
python demo/test_mobilenet.py
```

### 4. 執行手勢辨識 Demo
本專案提供三種版本的即時手勢辨識：

- **MediaPipe 版本 (推薦：最穩定)**
  ```bash
  python demo/gesture_recognition_mediapipe.py
  ```
- **SVM 版本 (原始像素分類)**
  ```bash
  python demo/gesture_recognition_svm.py
  ```
- **MobileNet 版本 (CNN 模型)**
  ```bash
  python demo/gesture_recognition_MobileNet.py
  ```
  *(註：SVM 與 MobileNet 版本需確保資料夾內有對應的訓練權重檔)*

---

## 🍓 Raspberry Pi 4 刷機與執行步驟

參考來源: [Raspberry Pi Software](https://www.raspberrypi.com/software/)

1. 下載 Raspberry Pi Imager
   ![alt text](image.png)
2. 選擇 Raspberry Pi 4 64-bit
   ![alt text](image-1.png)
3. 插上讀卡機
   ![alt text](image-2.png)
4. 輸入主機名，供 SSH 連線使用
   ![alt text](image-3.png)
5. 設定地區 (首都 Taipei) 與時區 (Asia/Taipei)
   ![alt text](image-4.png)
6. 輸入用戶名及密碼，供 SSH 連線使用
   ![alt text](image-5.png)
7. 開啟 SSH 功能
   ![alt text](image-6.png)
8. 完成寫入
   ![alt text](image-7.png)

刷機完成後，可透過 SSH 連線進入 Raspberry Pi，克隆本專案並依照前述「快速開始」步驟安裝套件並執行辨識程式。
