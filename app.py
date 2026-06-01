import pickle
from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app = Flask(__name__)

## Route for homepage

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    try:
        # Read form values (field names must match templates/home.html)
        pregnancies = int(float(request.form.get('Pregnancies')))
        glucose = int(float(request.form.get('Glucose')))
        blood_pressure = int(float(request.form.get('BloodPressure')))
        skin_thickness = float(request.form.get('SkinThickness'))
        insulin = int(float(request.form.get('Insulin')))
        bmi = float(request.form.get('BMI'))
        dpf = float(request.form.get('DiabetesPedigreeFunction'))
        age = int(float(request.form.get('Age')))

        # Build feature vector using CustomData class
        data = CustomData(
            Pregnancies=pregnancies,
            Glucose=glucose,
            BloodPressure=blood_pressure,
            SkinThickness=skin_thickness,
            Insulin=insulin,
            BMI=bmi,
            DiabetesPedigreeFunction=dpf,
            Age=age,
        )

        df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(df)

        # model.predict returns array-like with shape (1,)
        pred_value = int(prediction[0]) if hasattr(prediction, '__len__') else int(prediction)

        result_text = 'Diabetes Present' if pred_value == 1 else 'Diabetes Not Present'
        return render_template('home.html', prediction=result_text)

    except Exception as e:
        return render_template('home.html', prediction=f'Error: {e}')

if __name__=='__main__':
    app.run(host="0.0.0.0",debug=True)