import cv2
import mediapipe as mp
import math
import threading
import time

def get_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# 全域變數用於執行緒間溝通
latest_frame = None
latest_result = ("Initializing...", (0, 255, 0))
latest_landmarks = None
running = True

def mediapipe_worker(hands, mp_draw, mp_hands):
    global latest_frame, latest_result, latest_landmarks, running
    while running:
        if latest_frame is not None:
            # 複製目前的影像進行處理
            img_to_proc = latest_frame.copy()
            rgb_frame = cv2.cvtColor(img_to_proc, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            gesture = "Error (No hand)"
            landmarks = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = hand_landmarks
                    wrist = hand_landmarks.landmark[0]
                    mcp_middle = hand_landmarks.landmark[9]
                    
                    # 1. 計算手掌大小作為基準 (腕部到中指根部)
                    hand_size = get_distance(wrist, mcp_middle)
                    if hand_size == 0: hand_size = 0.1
                    
                    # 2. 判斷手指是否伸展 (距離基準)
                    # 提示：食指(8), 中指(12), 無名指(16), 小指(20)
                    finger_tips = [8, 12, 16, 20]
                    finger_pips = [6, 10, 14, 18]
                    
                    up_states = []
                    for tip, pip in zip(finger_tips, finger_pips):
                        dist_tip = get_distance(hand_landmarks.landmark[tip], wrist)
                        dist_pip = get_distance(hand_landmarks.landmark[pip], wrist)
                        # 判定為伸出的條件：指尖距離腕部明顯大於第二關節
                        up_states.append(dist_tip > dist_pip * 1.1)
                    
                    up_count = sum(up_states)
                    
                    # 3. 根據伸出數量初步分類
                    if up_count == 0:
                        gesture = "Rock"
                    elif up_count == 2:
                        # 4. 指縫偵測 (V-Shape Check for Scissors)
                        # 檢查食指(8)與中指(12)是否伸出，且兩者間距足夠
                        if up_states[0] and up_states[1]: # 食指與中指伸出
                            index_tip = hand_landmarks.landmark[8]
                            middle_tip = hand_landmarks.landmark[12]
                            tip_gap = get_distance(index_tip, middle_tip)
                            
                            # 指縫與手長的比例 (通常 V 型大於 0.4)
                            gap_ratio = tip_gap / hand_size
                            if gap_ratio > 0.45:
                                gesture = "Scissors"
                            else:
                                gesture = "Error (Squeezed Scissors)"
                        else:
                            gesture = "Error (Invalid Fingers)"
                    elif up_count >= 4:
                        # 5. 布的指縫偵測 (可選：檢查是否五指張開)
                        gesture = "Paper"
                    else:
                        gesture = f"Error ({up_count} fingers)"

            color = (0, 0, 255) if "Error" in gesture else (0, 255, 0)
            latest_result = (gesture, color)
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

    # 2. 開啟攝像頭
    cap = cv2.VideoCapture(0)
    # 設定較低的解析度以提升效能
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    # 啟動辨識執行緒
    thread = threading.Thread(target=mediapipe_worker, args=(hands, mp_draw, mp_hands))
    thread.daemon = True
    thread.start()

    print("MediaPipe Threaded Camera started! Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 鏡像翻轉
        frame = cv2.flip(frame, 1)
        
        # 更新全域變數
        latest_frame = frame

        # 取得最新結果與關節點
        gesture, color = latest_result
        if latest_landmarks:
            mp_draw.draw_landmarks(frame, latest_landmarks, mp_hands.HAND_CONNECTIONS)

        # 顯示結果
        cv2.putText(frame, f"Status: {gesture}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow("MediaPipe Threaded (320x240)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
