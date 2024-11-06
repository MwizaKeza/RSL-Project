from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

model = tf.keras.models.load_model('rsl_model_1.h5')

mp_holistic = mp.solutions.holistic

app = Flask(__name__)

label_map = {
    'Amakuru yawe': 0, 'Muraho': 1, 'Murakoze': 2, 'Ni meza': 3, 'Nitwa': 4, 'Ntuye': 5, 'Ukora iki': 6,
    'Guhagarika': 7, 'Gukunda': 8, 'Kugira': 9, 'Kunywa': 10, 'Kurya': 11, 'Mfasha': 12, 'Ndashaka': 13,
    'A': 14, 'B': 15, 'C': 16, 'D': 17, 'E': 18, 'F': 19, 'G': 20, 'H': 21, 'I': 22, 'J': 23, 'K': 24,
    'L': 25, 'M': 26, 'N': 27, 'O': 28, 'P': 29, 'Q': 30, 'R': 31, 'S': 32, 'T': 33, 'U': 34, 'V': 35,
    'W': 36, 'X': 37, 'Y': 38, 'Z': 39, 'Hehe': 40, 'Inde': 41, 'Ryari': 42, 'Amafaranga': 43, 'Amata': 44,
    'Ishuli': 45, 'Rwanda': 46, 'Umuntu': 47, 'Umuryango': 48, 'Iminsi': 49
}

actions = np.array(list(label_map.keys()))

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

def generate_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam resolution: {width}x{height}")

    sequence = []
    prediction_threshold = 0.5

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            image, results = mediapipe_detection(frame, holistic)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                if np.max(res) > prediction_threshold:
                    action = actions[np.argmax(res)]
                    print(f"Predicted action: {action} with confidence {np.max(res):.2f}")

                    cv2.rectangle(image, (0, height - 50), (width, height), (0, 0, 0), -1)
                    cv2.putText(image, action, (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()
        print("Video capture released.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
