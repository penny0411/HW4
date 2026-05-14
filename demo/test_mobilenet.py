import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report

def main():
    # 1. 設定模型與資料集路徑
    model_name = 'rps_mobilenet_v2.h5'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, model_name)
    
    # 假設 demo 資料夾與 dataset 資料夾在同一個主目錄下
    base_dir = os.path.dirname(script_dir)
    test_dir = os.path.join(base_dir, 'dataset', 'test')

    if not os.path.exists(model_path):
        print(f"❌ 錯誤：找不到模型檔案 '{model_path}'，請確認是否已放入 demo 資料夾。")
        return
    
    if not os.path.exists(test_dir):
        print(f"❌ 錯誤：找不到測試資料集 '{test_dir}'。")
        return

    # 2. 載入模型
    print("⏳ 載入 MobileNetV2 模型中...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ 模型載入成功！\n")
    except Exception as e:
        print(f"❌ 載入模型失敗: {e}")
        return

    # ImageDataGenerator 通常按字母排序: paper (0), rock (1), scissors (2)
    label_map = {'paper': 0, 'rock': 1, 'scissors': 2}
    # 為了最後顯示報告，我們需要名稱列表
    target_names = ['Paper', 'Rock', 'Scissors']
    
    X_test, y_test = [], []

    # 3. 讀取並處理測試圖片
    print("📂 正在讀取測試集圖片並進行預測...")
    for category, label_idx in label_map.items():
        category_path = os.path.join(test_dir, category)
        
        # 處理資料夾內可能多包一層的情況 (與 test.py 保持一致)
        if not os.path.exists(category_path):
            subdirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
            if subdirs:
                category_path = os.path.join(test_dir, subdirs[0], category)

        if not os.path.exists(category_path):
            print(f"⚠️ 找不到 {category} 的圖片資料夾，略過...")
            continue
            
        print(f"正在處理 {category}...")
        for filename in os.listdir(category_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(category_path, filename)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # 資料前處理 (必須與訓練時一致)
                    # MobileNetV2 通常使用 RGB，cv2 讀入是 BGR
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(img_rgb, (224, 224))
                    X_test.append(resized)
                    y_test.append(label_idx)

    if not X_test:
        print("❌ 錯誤：沒有讀取到任何圖片，請檢查資料夾結構。")
        return

    # 轉為 numpy array 並正規化
    X_test = np.array(X_test) / 255.0
    y_test = np.array(y_test)

    # 4. 進行預測與評估
    print(f"\n🚀 正在對 {len(y_test)} 張圖片進行預測...")
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 MobileNetV2 測試結果統整:")
    print(f"總共測試了 {len(y_test)} 張圖片")
    print(f"🎯 模型準確率: {accuracy * 100:.2f}%\n")
    print("📝 分類詳細報告:")
    print(classification_report(y_test, y_pred, target_names=target_names))

if __name__ == "__main__":
    main()
