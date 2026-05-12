import cv2
import numpy as np
import joblib
import os

def main():
    # 1. 設定模型路徑
    # 這裡假設腳本在 demo 資料夾內執行，模型也在 demo 資料夾內
    model_name = 'rps_svm_model.pkl'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, model_name)

    if not os.path.exists(model_path):
        print(f"Error: Cannot find model file '{model_path}'")
        return

    # 2. 載入模型
    print("⏳ 載入模型中...")
    try:
        clf = joblib.load(model_path)
        print("✅ 模型載入成功！")
    except Exception as e:
        print(f"❌ 載入模型失敗: {e}")
        return

    # 標籤應與訓練時一致: 0=Rock, 1=Paper, 2=Scissors
    label_map = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}

    # 3. 開啟攝像頭
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    print("Camera started! Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 鏡像翻轉 (讓操作更直覺)
        frame = cv2.flip(frame, 1)

        # 4. 影像處理 (需與訓練時一致)
        # a. 轉灰階
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # b. 裁切中間區域 (通常手勢會放在中間) 或直接縮放全螢幕
        # 為了保證一致性，我們先縮放全圖
        resized = cv2.resize(gray, (64, 64))
        
        # c. 攤平並正規化
        input_data = resized.flatten().reshape(1, -1) / 255.0

        # 5. 進行預測
        prediction = clf.predict(input_data)[0]
        gesture = label_map.get(prediction, "Unknown")

        # 6. 顯示結果
        cv2.putText(frame, f"Gesture: {gesture}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
