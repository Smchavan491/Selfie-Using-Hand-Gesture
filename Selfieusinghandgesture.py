import cv2
import mediapipe as mp

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands  #mediapipe hands module
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils    #used to draw handmarks on images

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define a function to check if the palm is open
def is_palm_open(hand_landmarks):
    if hand_landmarks:
        # Get coordinates of landmarks for wrist, index finger MCP, and pinky finger MCP
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

        # Check if the hand is open by comparing distances
        if index_mcp.y < wrist.y and pinky_mcp.y < wrist.y:
            return True
    return False

while cap.isOpened():   #as long as the caemra is open it continously captures video frames
    success, image = cap.read()     #reading the frame
    if not success:
        break

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #convert BGR image to RGB 
    results = hands.process(image_rgb)  #Process the image to detect hand landmarks

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS) # Draw hand landmarks on the image
            if is_palm_open(hand_landmarks): # Check if the palm is open
                cv2.imwrite('selfie.png', image) # Save the image as a selfie
                print("Selfie taken!")
                break  # Exit the loop after taking a selfie

    # Display the image
    cv2.imshow('Hand Gesture Selfie', image)

    if cv2.waitKey(5) & 0xFF == 27: #Exit the loop or frame when Esc key is press
        break

cap.release()
cv2.destroyAllWindows() #closed all opencv window
