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

# 定義偵測框的大小與中心
BOX_SIZE = 224 # 與模型輸入一致

def fixed_box_inference_worker(model, label_map, threshold, box_coords):
    global latest_frame, latest_result, running
    xmin, ymin, xmax, ymax = box_coords
    
    while running:
        if latest_frame is not None:
            # 複製目前的影像進行處理
            img_to_proc = latest_frame.copy()
            
            # 1. 根據固定框進行裁切
            hand_crop = img_to_proc[ymin:ymax, xmin:xmax]
            
            if hand_crop.size != 0:
                # 影像預處理
                img = cv2.resize(hand_crop, (224, 224))
                img = img / 255.0
                img = np.expand_dims(img, axis=0)

                # 2. 進行預測
                prediction = model.predict(img, verbose=0)
                class_idx = np.argmax(prediction)
                confidence = prediction[0][class_idx]
                
                if confidence < threshold:
                    gesture = "Error (Unknown)"
                    color = (0, 0, 255)
                else:
                    gesture = label_map.get(class_idx, "Unknown")
                    color = (0, 255, 0)
                
                latest_result = (gesture, confidence, color)
            else:
                latest_result = ("Error (Crop failed)", 0.0, (0, 0, 255))
        
        time.sleep(0.01)

def main():
    global latest_frame, latest_result, running
    
    # 1. 設定模型路徑
    model_name = 'rps_mobilenet_v2.h5'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, model_name)

    if not os.path.exists(model_path):
        print(f"Error: Cannot find model file '{model_path}'.")
        return

    # 2. 載入模型
    print("Loading Optimized MobileNetV2 model...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("System Ready!")
    except Exception as e:
        print(f"Error initializing: {e}")
        return

    label_map = {0: 'Paper', 1: 'Rock', 2: 'Scissors'}
    CONFIDENCE_THRESHOLD = 0.75 

    # 3. 開開啟攝像頭
    cap = cv2.VideoCapture(0)
    # 設定解析度
    W, H = 320, 240
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    # 計算中央偵測框座標
    box_x = (W - BOX_SIZE) // 2
    box_y = (H - BOX_SIZE) // 2
    box_coords = [box_x, box_y, box_x + BOX_SIZE, box_y + BOX_SIZE]

    # 啟動辨識執行緒
    thread = threading.Thread(target=fixed_box_inference_worker, 
                              args=(model, label_map, CONFIDENCE_THRESHOLD, box_coords))
    thread.daemon = True
    thread.start()

    print("Stable Fixed-Box Gesture Recognition started! Press 'q' to quit.")
    print("Please place your hand inside the GREEN box.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        latest_frame = frame

        # 取得最新結果
        gesture, confidence, text_color = latest_result

        # 繪製固定偵測框
        cv2.rectangle(frame, (box_x, box_y), (box_x + BOX_SIZE, box_y + BOX_SIZE), (0, 255, 0), 2)
        cv2.putText(frame, "Scan Area", (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 顯示結果
        cv2.putText(frame, f"Gesture: {gesture}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
