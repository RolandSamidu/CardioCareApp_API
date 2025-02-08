import uuid
from flask import Blueprint, request, jsonify
from firebase_admin import firestore
import joblib
import pandas as pd
import numpy as np

db = firestore.client()
user_Ref = db.collection('ecg')
doctor_patient_Ref = db.collection('doctor-patient')

# Load Model
model_heart = joblib.load('api/model_heart.joblib')

heartpatientmodel = Blueprint('heartpatientmodel', __name__)

@heartpatientmodel.route('/predict_heart', methods=['POST'])
def predict_heart_endpoint():
    try:
        # Get the JSON data from the request
        features = request.json

        # Convert the features to a numpy array
        input_data_as_numpy_array = np.asarray(list(features.values()))

        # Reshape the numpy array as we are predicting for only one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        # Predict with the heart disease model
        prediction = model_heart.predict(input_data_reshaped).item()

        if prediction == 0:
            result = {'prediction': 'The Person does not have a Heart Disease', 'probability': None}
        else:
            probability = model_heart.predict_proba(input_data_reshaped)[:, 1].item()
            result = {'prediction': 'The Person has Heart Disease', 'probability': probability}

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})
