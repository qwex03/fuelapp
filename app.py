from flask import render_template, Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open('fuel.pickle', 'rb'))
cols = ['MODELYEAR', 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict',methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    final = np.array(features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction=model.predict(data_unseen)
    result=prediction[0]
    return render_template('index.html',resultat=f"Les émission C02 du véhicule sont {result:.2f}")

if __name__ == '__main__':
    app.run()

