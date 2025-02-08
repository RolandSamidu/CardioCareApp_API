import uuid
from flask import Blueprint,request,jsonify
from firebase_admin import firestore
import joblib
import pandas as pd

db=firestore.client()
user_Ref=db.collection('user')
doctor_patient_Ref=db.collection('doctor-patient')
predictions_Ref=db.collection('predictions')

# Load Model
model2 = joblib.load('api/heart_disease_model.joblib')

userapi=Blueprint('userapi',__name__)

def predict_cardio_presence(features):
    # Convert the features to a DataFrame
    features_df = pd.DataFrame([features])

    # Predict the probability and class
    probability = model2.predict_proba(features_df)[:, 1].item()
    prediction = model2.predict(features_df).item()

    return prediction, probability

@userapi.route('/add',methods=['POST'])
def createuser():
    try:
        id=uuid.uuid4()
        user_Ref.document(id.hex).set(request.json)
        return jsonify({"success" : True}),200
    except Exception as e:
        return f"An Error Occurs : {e}"

@userapi.route('/get', methods=['GET'])
def get_user_by_email():
    try:
        # Get the email parameter from the URL
        email = request.args.get('email')

        if email:
            # Query Firestore to find the user with the specified email
            user_query = user_Ref.where('email', '==', email).limit(1).get()

            if user_query:
                user_data = user_query[0].to_dict()
                return jsonify({"success": True, "user": user_data}), 200
            else:
                return jsonify({"success": False, "message": "User not found"}), 404
        else:
            return jsonify({"success": False, "message": "Email parameter is missing"}), 400

    except Exception as e:
        return jsonify({"success": False, "error": f"An Error Occurs: {e}"}), 500

@userapi.route('/get-doctors', methods=['GET'])
def get_doctors():
    try:
        role = request.args.get('role', default='Doctor')
        doctors_query = user_Ref.where('role', '==', role).get()

        if doctors_query:
            doctors_data = [doctor.to_dict() for doctor in doctors_query]
            return jsonify({"success": True, "doctors": doctors_data}), 200
        else:
            return jsonify({"success": False, "message": "No doctors found"}), 404

    except Exception as e:
        return jsonify({"success": False, "error": f"An error occurred: {e}"}), 500

@userapi.route('/create-update-record', methods=['POST'])
def create_update_record():
    try:
        # Get patient's and doctor's email from the request JSON
        patient_email = request.json.get('patient_email')
        # doctor_email = request.json.get('doctor_email')
        doctor_data = request.json.get('doctor')

        if not patient_email or not doctor_data:
            return jsonify({"success": False, "message": "Patient email and doctor email are required"}), 400

        # Check if the record already exists
        record_ref = doctor_patient_Ref.document(patient_email)

        if record_ref.get().exists:
            # Update the existing record
            record_ref.update({
                'doctor': doctor_data,
            })
            return jsonify({"success": True, "message": "Record updated successfully"}), 200
        else:
            # Create a new record
            record_ref.set({
                'doctor': doctor_data,
            })
            return jsonify({"success": True, "message": "Record created successfully"}), 200

    except Exception as e:
        return jsonify({"success": False, "error": f"An error occurred: {e}"}), 500


@userapi.route('/get-record', methods=['GET'])
def get_record():
    try:
        # Get patient's email from the request parameters
        patient_email = request.args.get('patient_email')

        if not patient_email:
            return jsonify({"success": False, "message": "Patient email is required"}), 400

        # Retrieve the record
        record_data = doctor_patient_Ref.document(patient_email).get()

        if record_data.exists:
            return jsonify({"success": True, "record": record_data.to_dict()}), 200
        else:
            return jsonify({"success": False, "message": "Record not found"}), 404

    except Exception as e:
        return jsonify({"success": False, "error": f"An error occurred: {e}"}), 500


@userapi.route('/get-patient', methods=['GET'])
def get_patient():
    try:
        # Get the patient's email parameter from the URL
        patient_email = request.args.get('patient_email')

        if patient_email:
            # Query Firestore to find the patient with the specified email
            patient_query = doctor_patient_Ref.document(patient_email).get()

            if patient_query.exists:
                patient_data = patient_query.to_dict()
                return jsonify({"success": True, "patient": patient_data}), 200
            else:
                return jsonify({"success": False, "message": "Patient not found"}), 404
        else:
            return jsonify({"success": False, "message": "Patient email parameter is missing"}), 400

    except Exception as e:
        return jsonify({"success": False, "error": f"An error occurred: {e}"}), 500

@userapi.route('/get-patients-by-doctor', methods=['GET'])
def get_patients_by_doctor():
    try:
        doctor_email = request.args.get('doctor_email')

        if doctor_email:
            patients_query = doctor_patient_Ref.where('doctor.email', '==', doctor_email).get()

            if patients_query:
                patients_data = {doc.id: doc.to_dict() for doc in patients_query}
                return jsonify({"success": True, "patients": patients_data}), 200
            else:
                return jsonify({"success": False, "message": "No patients found for the specified doctor"}), 404
        else:
            return jsonify({"success": False, "message": "Doctor email parameter is missing"}), 400

    except Exception as e:
        return jsonify({"success": False, "error": f"An error occurred: {e}"}), 500

@userapi.route('/get-patients-by-doctor-array', methods=['GET'])
def get_patients_by_doctor_array():
    try:
        doctor_email = request.args.get('doctor_email')

        if doctor_email:
            patients_query = doctor_patient_Ref.where('doctor.email', '==', doctor_email).get()

            if len(patients_query) > 0:
                patients_ids = [doc.id for doc in patients_query]
                return jsonify({"success": True, "patients": patients_ids}), 200
            else:
                return jsonify({"success": False, "message": "No patients found for the specified doctor"}), 404
        else:
            return jsonify({"success": False, "message": "Doctor email parameter is missing"}), 400

    except Exception as e:
        return jsonify({"success": False, "error": f"An error occurred: {e}"}), 500


@userapi.route('/get-medical-records', methods=['GET'])
def get_medical_records():
    try:
        # Get the patient's email parameter from the URL
        patient_email = request.args.get('patient_email')

        if patient_email:
            records_query = predictions_Ref.where('user', '==', patient_email).get()

            if records_query:
                records_data = {doc.id: doc.to_dict() for doc in records_query}
                return jsonify({"success": True, "medical_records": records_data}), 200
            else:
                return jsonify({"success": False, "message": "No medical records found for the specified patient"}), 404
        else:
            return jsonify({"success": False, "message": "Patient email parameter is missing"}), 400

    except Exception as e:
        return jsonify({"success": False, "error": f"An error occurred: {e}"}), 500
    

@userapi.route('/predict-cardio', methods=['POST'])
def predict_cardio():
    try:
        features = request.json['data']
        _age = features['age']
        _gender = features['gender']
        _cp = features['cp']
        _bp = features['bp']
        _cole = features['cole']
        _fbs = features['fbs']
        _recg = features['recg']
        _hrate = features['hrate']
        _exe = features['exe']
        _old = features['old']
        _slope = features['slope']
        _vessel = features['vessel']
        _thal = features['thal']

        # Perform prediction using the individual features
        prediction, probability = predict_cardio_presence((_age, _gender, _cp, _bp, _cole, _fbs, _recg, _hrate, _exe, _old, _slope, _vessel, _thal))

        # Prepare response
        response = {
            'prediction': prediction,
            'probability': probability
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
