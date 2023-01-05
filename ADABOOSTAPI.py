from flask import Flask, request, jsonify
import numpy as np
import pickle

model = pickle.load(open('model_ada.pkl', 'rb'))
ADABOOSTAPI = Flask(__name__)


@ADABOOSTAPI.route('/')
def home():
    return "Hello world"

@ADABOOSTAPI.route('/predictionwithadaboost', methods=['POST'])
def predict():
    ph = request.form.get('ph')
    Solids = request.form.get('Solids')
    Turbidity = request.form.get('Turbidity')
    Temp = request.form.get('Temp')

    input_query = np.array([[ph, Solids, Turbidity, Temp]], dtype=object)
    result = model.predict(input_query)[0]
    return jsonify({'Water is clean': str(result)})


if __name__ == '__main__':
    ADABOOSTAPI.debug = True
    ADABOOSTAPI.run()
