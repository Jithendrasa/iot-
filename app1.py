from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the machine learning model using pickle
with open("jio.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get sensor inputs from the form
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        gas_level = float(request.form['gas_level'])
        water_level = float(request.form['water_level'])

        # Make a prediction using the loaded machine learning model
        sensor_data = np.array([[temperature, humidity, gas_level, water_level]])
        prediction = model.predict(sensor_data)

        # Map the prediction to a human-readable status
        status = "Unsafe" if prediction == 1 else "Safe"

        return render_template('out.html', output=status)

if __name__ == '__main__':
    app.run(debug=True)
