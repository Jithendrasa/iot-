import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__, template_folder="templates")
model = pickle.load(open('sensor.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Extracting input features from the form
        temperature = float(request.form.get('temperature'))
        humidity = float(request.form.get('humidity'))
        gas_level = float(request.form.get('gas_level'))
        water_level = float(request.form.get('water_level'))

        # Making a prediction
        arr= np.array([[temperature, humidity, gas_level, water_level]])
        brr=np.asarray(arr,dtype=float)
        output = model.predict(arr)
        
        # Determining safety status based on the prediction
        if output == 1:
            safety_status = 'Unsafe'
        else:
            safety_status = 'Safe'

        return render_template('out1.html', safety_status=safety_status)

    except Exception as e:
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
