import cv2
import numpy as np
import tensorflow as tf
import os

def main():
    # 1. 設定模型路徑
    model_name = 'rps_mobilenet_v2.h5'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, model_name)

    if not os.path.exists(model_path):
        print(f"Error: Cannot find model file '{model_path}'. Please run training first.")
        return

    # 2. 載入模型
    print("Loading MobileNetV2 model...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 標籤應與訓練時一致
    label_map = {0: 'Paper', 1: 'Rock', 2: 'Scissors'} # 注意：ImageDataGenerator 通常按字母排序

    # 3. 開啟攝像頭
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    print("MobileNetV2 Recognition started! Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 鏡像翻轉
        frame = cv2.flip(frame, 1)

        # 4. 影像處理
        img = cv2.resize(frame, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # 5. 進行預測
        prediction = model.predict(img, verbose=0)
        class_idx = np.argmax(prediction)
        gesture = label_map.get(class_idx, "Unknown")
        confidence = prediction[0][class_idx]

        # 6. 顯示結果
        cv2.putText(frame, f"Gesture: {gesture} ({confidence:.2f})", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("MobileNetV2 Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
