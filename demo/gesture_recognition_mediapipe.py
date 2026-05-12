import cv2
import mediapipe as mp
import math

def get_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def main():
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
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    print("MediaPipe Camera started! Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 鏡像翻轉
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # 轉換顏色空間為 RGB (MediaPipe 需要)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        gesture = "No hand"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 繪制手部關節
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 判斷手指是否伸展
                # 我們檢測 4 根手指 (食指、中指、無名指、小指)
                # 邏輯：指尖 (Tip) 到手腕的距離 > 第二關節 (PIP) 到手腕的距離 => 伸展
                finger_tips = [8, 12, 16, 20]
                finger_pips = [6, 10, 14, 18]
                wrist = hand_landmarks.landmark[0]
                
                up_count = 0
                for tip, pip in zip(finger_tips, finger_pips):
                    dist_tip = get_distance(hand_landmarks.landmark[tip], wrist)
                    dist_pip = get_distance(hand_landmarks.landmark[pip], wrist)
                    if dist_tip > dist_pip:
                        up_count += 1
                
                # 特殊處理拇指 (比較 Tip 與 IP 的距離)
                # 簡單起見，我們主要依靠 4 根手指來判斷剪刀石頭布
                
                if up_count == 0:
                    gesture = "Rock"
                elif up_count == 2:
                    # 檢查是否為食指和中指 (剪刀)
                    dist_index = get_distance(hand_landmarks.landmark[8], wrist)
                    dist_middle = get_distance(hand_landmarks.landmark[12], wrist)
                    dist_ring = get_distance(hand_landmarks.landmark[16], wrist)
                    if dist_index > get_distance(hand_landmarks.landmark[6], wrist) and \
                       dist_middle > get_distance(hand_landmarks.landmark[10], wrist) and \
                       dist_ring < get_distance(hand_landmarks.landmark[14], wrist):
                        gesture = "Scissors"
                    else:
                        gesture = f"Unknown ({up_count} fingers)"
                elif up_count >= 4:
                    gesture = "Paper"
                else:
                    gesture = f"Unknown ({up_count} fingers)"

        # 顯示結果
        color = (0, 0, 255) if gesture == "No hand" else (0, 255, 0)
        cv2.putText(frame, f"Status: {gesture}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
        cv2.imshow("MediaPipe Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
