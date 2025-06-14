import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

def calculate_ear(landmarks, eye_indices):
    eye = [landmarks[i] for i in eye_indices]
    eye = np.array([[p.x, p.y] for p in eye])
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

cap = cv2.VideoCapture(0)
EAR_THRESHOLD = 0.25
SLEEP_FRAMES = 15
counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear = calculate_ear(face_landmarks.landmark, LEFT_EYE)
            right_ear = calculate_ear(face_landmarks.landmark, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                counter += 1
                if counter > SLEEP_FRAMES:
                    cv2.putText(frame, "SLEEPING", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                counter = 0
                cv2.putText(frame, "AWAKE", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sleep Detection", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
