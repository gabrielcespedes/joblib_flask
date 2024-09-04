from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load('modelo_entrenado.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form #obtiene los datos del formulario HTML
    features = [float(data['feature1']), float(data['feature2']), float(data['feature3']), float(data['feature4'])]
    prediction = model.predict([features])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(port = 5000)