import cv2
import numpy as np
import tensorflow as tf
import os
import threading
import time

# 全域變數用於執行緒間溝通
latest_frame = None
latest_result = ("Initializing...", 0.0, (0, 255, 0))
running = True

def inference_worker(model, label_map, threshold):
    global latest_frame, latest_result, running
    while running:
        if latest_frame is not None:
            # 複製目前的影像進行處理
            img_to_proc = latest_frame.copy()
            
            # 4. 影像處理
            img = cv2.resize(img_to_proc, (224, 224))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            # 5. 進行預測
            prediction = model.predict(img, verbose=0)
            class_idx = np.argmax(prediction)
            confidence = prediction[0][class_idx]
            
            # 門檻判斷
            if confidence < threshold:
                gesture = "Error (Low Confidence)"
                color = (0, 0, 255) # 紅色
            else:
                gesture = label_map.get(class_idx, "Unknown")
                color = (0, 255, 0) # 綠色
            
            latest_result = (gesture, confidence, color)
        
        # 稍微暫停避免過度佔用 CPU
        time.sleep(0.01)

def main():
    global latest_frame, latest_result, running
    
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

    label_map = {0: 'Paper', 1: 'Rock', 2: 'Scissors'}
    CONFIDENCE_THRESHOLD = 0.8

    # 3. 開啟攝像頭
    cap = cv2.VideoCapture(0)
    # 設定較低的解析度以提升效能
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    # 啟動辨識執行緒
    thread = threading.Thread(target=inference_worker, args=(model, label_map, CONFIDENCE_THRESHOLD))
    thread.daemon = True
    thread.start()

    print(f"MobileNetV2 Threaded Recognition started! Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 鏡像翻轉
        frame = cv2.flip(frame, 1)
        
        # 更新全域變數給執行緒使用
        latest_frame = frame

        # 取得最新的辨識結果
        gesture, confidence, text_color = latest_result

        # 6. 顯示結果
        cv2.putText(frame, f"Gesture: {gesture} ({confidence:.2f})", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        cv2.imshow("MobileNetV2 Threaded (320x240)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
