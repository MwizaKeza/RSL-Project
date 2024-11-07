from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load your existing trained model
model = tf.keras.models.load_model('rsl_model_1.h5')

label_map = {
    'Amakuru yawe': 0, 'Muraho': 1, 'Murakoze': 2, 'Ni meza': 3, 'Nitwa': 4, 'Ntuye': 5, 'Ukora iki': 6,
    'Guhagarika': 7, 'Gukunda': 8, 'Kugira': 9, 'Kunywa': 10, 'Kurya': 11, 'Mfasha': 12, 'Ndashaka': 13,
    'A': 14, 'B': 15, 'C': 16, 'D': 17, 'E': 18, 'F': 19, 'G': 20, 'H': 21, 'I': 22, 'J': 23, 'K': 24,
    'L': 25, 'M': 26, 'N': 27, 'O': 28, 'P': 29, 'Q': 30, 'R': 31, 'S': 32, 'T': 33, 'U': 34, 'V': 35,
    'W': 36, 'X': 37, 'Y': 38, 'Z': 39, 'Hehe': 40, 'Inde': 41, 'Ryari': 42, 'Amafaranga': 43, 'Amata': 44,
    'Ishuli': 45, 'Rwanda': 46, 'Umuntu': 47, 'Umuryango': 48, 'Iminsi': 49
}

actions = np.array(list(label_map.keys()))
prediction_threshold = 0.5

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['frames']
        sequence = np.array(data).reshape(1, 30, -1)  # Reshape to expected input shape for the model
        prediction = model.predict(sequence)[0]

        if np.max(prediction) > prediction_threshold:
            action = actions[np.argmax(prediction)]
            confidence = float(np.max(prediction))
            return jsonify({'action': action, 'confidence': confidence})
        else:
            return jsonify({'action': 'unknown', 'confidence': 0.0})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
