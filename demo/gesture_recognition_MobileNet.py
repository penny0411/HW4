import cv2
import numpy as np
import tensorflow as tf
import os
import threading
import time
import math
import mediapipe as mp

def get_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# 全域變數用於執行緒間溝通
latest_frame = None
latest_result = ("Initializing...", 0.0, (0, 255, 0))
latest_landmarks = None # 新增：用於繪製骨架
running = True

# 定義偵測框的大小與中心
BOX_SIZE = 224 # 與模型輸入一致

def hybrid_inference_worker(model, label_map, threshold, box_coords, hands):
    global latest_frame, latest_result, latest_landmarks, running
    xmin, ymin, xmax, ymax = box_coords
    
    while running:
        if latest_frame is not None:
            # 複製目前的影像進行處理
            img_to_proc = latest_frame.copy()
            rgb_frame = cv2.cvtColor(img_to_proc, cv2.COLOR_BGR2RGB)
            
            # 1. 使用 MediaPipe 進行手部偵測與過濾
            results = hands.process(rgb_frame)
            
            gesture = "Error (No hand)"
            confidence = 0.0
            color = (0, 0, 255)
            landmarks = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = hand_landmarks
                    wrist = hand_landmarks.landmark[0]
                    mcp_middle = hand_landmarks.landmark[9]
                    hand_size = get_distance(wrist, mcp_middle)
                    if hand_size == 0: hand_size = 0.1

                    # 判斷手指伸展狀態
                    finger_tips = [8, 12, 16, 20]
                    finger_pips = [6, 10, 14, 18]
                    up_states = []
                    for tip, pip in zip(finger_tips, finger_pips):
                        up_states.append(get_distance(hand_landmarks.landmark[tip], wrist) > 
                                       get_distance(hand_landmarks.landmark[pip], wrist) * 1.1)
                    
                    up_count = sum(up_states)

                    # --- 指縫與組合檢查 ---
                    gap_error = False
                    for i in range(len(up_states) - 1):
                        if up_states[i] and up_states[i+1]:
                            dist = get_distance(hand_landmarks.landmark[finger_tips[i]], hand_landmarks.landmark[finger_tips[i+1]])
                            if dist / hand_size < 0.4:
                                gap_error = True
                                break
                    
                    if gap_error:
                        gesture = "Error (Gaps too small)"
                    elif up_count in [0, 2, 4, 5]:
                        # 檢查 2 指是否為食指+中指 (剪刀)
                        if up_count == 2 and not (up_states[0] and up_states[1]):
                            gesture = "Error (Invalid combo)"
                        else:
                            # 通過過濾，進行 MobileNet 分類
                            hand_crop = img_to_proc[ymin:ymax, xmin:xmax]
                            if hand_crop.size != 0:
                                img = cv2.resize(hand_crop, (224, 224))
                                img = (img / 255.0).astype(np.float32)
                                img = np.expand_dims(img, axis=0)

                                prediction = model.predict(img, verbose=0)
                                class_idx = np.argmax(prediction)
                                confidence = prediction[0][class_idx]
                                
                                if confidence < threshold:
                                    gesture = "Error (Low Conf)"
                                else:
                                    gesture = label_map.get(class_idx, "Unknown")
                                    color = (0, 255, 0)
                            else:
                                gesture = "Error (Crop failed)"
                    else:
                        gesture = f"Error (Invalid combo: {up_count})"

            latest_result = (gesture, confidence, color)
            latest_landmarks = landmarks
        
        time.sleep(0.01)

def main():
    global latest_frame, latest_result, latest_landmarks, running
    
    # 1. 初始化 MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    # 2. 設定模型路徑
    model_name = 'rps_mobilenet_v2.h5'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, model_name)

    if not os.path.exists(model_path):
        print(f"Error: Cannot find model file '{model_path}'.")
        return

    # 3. 載入模型
    print("Loading Optimized MobileNetV2 model...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("System Ready!")
    except Exception as e:
        print(f"Error initializing: {e}")
        return

    label_map = {0: 'Paper', 1: 'Rock', 2: 'Scissors'}
    CONFIDENCE_THRESHOLD = 0.75

    # 4. 開啟攝像頭
    cap = cv2.VideoCapture(0)
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

    # 5. 啟動辨識執行緒 (改用 Hybrid Worker)
    thread = threading.Thread(target=hybrid_inference_worker, 
                               args=(model, label_map, CONFIDENCE_THRESHOLD, box_coords, hands))
    thread.daemon = True
    thread.start()

    print("Hybrid Gesture Recognition started! Press 'q' to quit.")
    print("Detection Rules: Valid combinations only (e.g. 1.3.4 = Error) + Finger Gaps check.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        latest_frame = frame

        # 取得最新結果與骨架
        gesture, confidence, text_color = latest_result
        if latest_landmarks:
            mp_draw.draw_landmarks(frame, latest_landmarks, mp_hands.HAND_CONNECTIONS)

        # 繪製偵測框
        cv2.rectangle(frame, (box_x, box_y), (box_x + BOX_SIZE, box_y + BOX_SIZE), (0, 255, 0), 2)
        cv2.putText(frame, "Scan Area", (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 顯示結果
        cv2.putText(frame, f"Gesture: {gesture}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        if confidence > 0:
            cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        cv2.imshow("Hybrid Recognition (MediaPipe + MobileNet)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
