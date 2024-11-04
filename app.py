from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import time

# Initialize the Flask app
app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('rsl_model_1.h5')
DATA_PATH = 'RSL_Dataset'

# Load actions list
actions = []
for category in os.listdir(DATA_PATH):
    category_path = os.path.join(DATA_PATH, category)
    if os.path.isdir(category_path):
        sign_folders = [folder for folder in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, folder))]
        actions.extend(sign_folders)
actions = np.array(actions)
print("Loaded actions list:", actions)  # Modify with your model's specific actions

# Initialize Mediapipe holistic
mp_holistic = mp.solutions.holistic

sentence = ""
predictions = []
last_label_time = time.time()
is_spelling_word = False
threshold = 0.5

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def is_hand_moving(sequence):
    if len(sequence) < 2:
        return False
    movement = np.sum(np.abs(sequence[-1] - sequence[-2]))
    movement_threshold = 0.01  # Adjust this value based on your model's sensitivity
    return movement > movement_threshold

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def generate_frames():
    global sentence, predictions, last_label_time, is_spelling_word
    cap = cv2.VideoCapture(0)
    sequence = []

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run Mediapipe model without drawing landmarks
            image, results = mediapipe_detection(frame, holistic)
            
            # Extract keypoints and make predictions only if there is movement
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if is_hand_moving(sequence) and len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                
                current_time = time.time()
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        detected_label = actions[np.argmax(res)]
                        
                        if detected_label.isalpha() and len(detected_label) == 1:  # Letter detection
                            is_spelling_word = True
                            sentence += detected_label
                            last_label_time = current_time
                        else:  # Word detection
                            if not is_spelling_word or (is_spelling_word and (current_time - last_label_time > 3)):
                                is_spelling_word = False
                                sentence = detected_label
                                last_label_time = current_time

                # Clear the label after 3 seconds of inactivity
                if current_time - last_label_time > 3:
                    sentence = ""
                    is_spelling_word = False
            
            # Draw subtitles at the bottom of the video feed
            h, w, _ = image.shape
            cv2.rectangle(image, (0, h - 50), (w, h), (0, 0, 0), -1)
            cv2.putText(image, sentence, (w // 2 - len(sentence) * 10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()
    cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
