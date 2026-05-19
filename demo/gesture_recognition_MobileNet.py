import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import threading
import time

# 全域變數用於執行緒間溝通
latest_frame = None
latest_result = ("Initializing...", 0.0, (0, 255, 0))
latest_bbox = None # [xmin, ymin, xmax, ymax]
running = True

def hybrid_inference_worker(model, hands, label_map, threshold):
    global latest_frame, latest_result, latest_bbox, running
    
    while running:
        if latest_frame is not None:
            # 複製目前的影像進行處理
            img_to_proc = latest_frame.copy()
            h, w, _ = img_to_proc.shape
            
            # 1. 使用 MediaPipe 進行手部偵測
            rgb_frame = cv2.cvtColor(img_to_proc, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                # 取得第一隻手的邊界框
                hand_landmarks = results.multi_hand_landmarks[0]
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                
                xmin, xmax = int(min(x_coords) * w), int(max(x_coords) * w)
                ymin, ymax = int(min(y_coords) * h), int(max(y_coords) * h)
                
                # 增加邊距 (Padding) 並確保不超出邊界
                pad = 40
                xmin, ymin = max(0, xmin - pad), max(0, ymin - pad)
                xmax, ymax = min(w, xmax + pad), min(h, ymax + pad)
                
                latest_bbox = [xmin, ymin, xmax, ymax]

                # 2. 裁切手部區域
                hand_crop = img_to_proc[ymin:ymax, xmin:xmax]
                
                if hand_crop.size != 0:
                    # 影像預處理
                    img = cv2.resize(hand_crop, (224, 224))
                    img = img / 255.0
                    img = np.expand_dims(img, axis=0)

                    # 3. 進行預測
                    prediction = model.predict(img, verbose=0)
                    class_idx = np.argmax(prediction)
                    confidence = prediction[0][class_idx]
                    
                    if confidence < threshold:
                        gesture = "Error (Unsure)"
                        color = (0, 0, 255)
                    else:
                        gesture = label_map.get(class_idx, "Unknown")
                        color = (0, 255, 0)
                    
                    latest_result = (gesture, confidence, color)
                else:
                    latest_result = ("Error (Crop failed)", 0.0, (0, 0, 255))
            else:
                latest_bbox = None
                latest_result = ("Error (No hand)", 0.0, (0, 0, 255))
        
        time.sleep(0.01)

def main():
    global latest_frame, latest_result, latest_bbox, running
    
    # 1. 設定模型路徑
    model_name = 'rps_mobilenet_v2.h5'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, model_name)

    if not os.path.exists(model_path):
        print(f"Error: Cannot find model file '{model_path}'.")
        return

    # 2. 載入模型與 MediaPipe
    print("Loading Models...")
    try:
        model = tf.keras.models.load_model(model_path)
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                               min_detection_confidence=0.7, min_tracking_confidence=0.5)
        print("Hybrid system ready!")
    except Exception as e:
        print(f"Error initializing: {e}")
        return

    label_map = {0: 'Paper', 1: 'Rock', 2: 'Scissors'}
    CONFIDENCE_THRESHOLD = 0.8

    # 3. 開開啟攝像頭
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    # 啟動辨識執行緒
    thread = threading.Thread(target=hybrid_inference_worker, 
                              args=(model, hands, label_map, CONFIDENCE_THRESHOLD))
    thread.daemon = True
    thread.start()

    print("Hybrid Gesture Recognition started! Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        latest_frame = frame

        # 取得最新結果與邊界框
        gesture, confidence, text_color = latest_result
        bbox = latest_bbox

        # 繪製邊界框 (如有)
        if bbox:
            xmin, ymin, xmax, ymax = bbox
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), text_color, 2)

        # 顯示結果
        cv2.putText(frame, f"Gesture: {gesture} ({confidence:.2f})", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        cv2.imshow("Hybrid Recognition (MediaPipe + MobileNet)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
