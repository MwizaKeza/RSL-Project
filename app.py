from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import base64
import re

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

sequence = []
prediction_threshold = 0.2

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global sequence
    try:
        data = request.json
        image_data = re.sub('^data:image/.+;base64,', '', data['image'])
        decoded_image = base64.b64decode(image_data)
        np_arr = np.frombuffer(decoded_image, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
            image, results = mediapipe_detection(frame, holistic)

            keypoints = extract_keypoints(results)
            print("Extracted Keypoints:", keypoints)

            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(f"Model prediction result: {res}")

                if np.max(res) > prediction_threshold:
                    action = actions[np.argmax(res)]
                    confidence = np.max(res)
                    return jsonify({'action': str(action), 'confidence': float(confidence)})
                else:
                    print("No confident prediction found.")
                    return jsonify({'message': 'Frame processed but no confident prediction'})

        return jsonify({'message': 'Frame processed but no confident prediction'})
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'message': 'Error processing frame'}), 500

if __name__ == "__main__":
    app.run(debug=True)