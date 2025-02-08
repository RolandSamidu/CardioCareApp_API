from flask import Blueprint, request, jsonify
from firebase_admin import firestore
import uuid
from datetime import datetime

db = firestore.client()
medical_records_Ref = db.collection('medical_records')

recordapi = Blueprint('medical_records_api', __name__)

@recordapi.route('/create_medical_record', methods=['POST'])
def create_medical_record():
    try:
        data = request.get_json()

        # Extract data from the request
        user_email = data.get('user_email')
        condition = data.get('condition')
        severity = data.get('severity')
        comment = data.get('comment')

        # Map severity levels to integers
        severity_levels = {'high': 2, 'medium': 1, 'normal': 0}
        severity_index = severity_levels.get(severity.lower(), 0)

        # Get the current date and time
        recorded_date = datetime.now().strftime('%Y-%m-%d')

        # Generate a unique ID for the medical record
        record_id = str(uuid.uuid4())

        # Create the medical record document
        medical_record_data = {
            'id': record_id,
            'user_email': user_email,
            'recorded_date': recorded_date,
            'condition': condition,
            'severity': severity_index,
            'comment': comment
        }

        # Add the medical record to Firestore
        medical_records_Ref.document(record_id).set(medical_record_data)

        return jsonify({'message': 'Medical record created successfully'})

    except Exception as e:
        return jsonify({'error': str(e)})

@recordapi.route('/get_all_medical_records', methods=['GET'])
def get_all_medical_records():
    try:
        # Query all medical records from Firestore
        records = medical_records_Ref.order_by('recorded_date', direction='DESCENDING').stream()

        records_list = []
        for record in records:
            record_data = record.to_dict()
            records_list.append(record_data)

        return jsonify({'medical_records': records_list})

    except Exception as e:
        return jsonify({'error': str(e)})

@recordapi.route('/get_medical_records_by_user_email', methods=['POST'])
def get_medical_records_by_user_email():
    try:
        data = request.get_json()
        user_email = data.get('user_email')
        records = medical_records_Ref.where('user_email', '==', user_email).order_by('recorded_date', direction='DESCENDING').stream()

        records_list = []
        for record in records:
            record_data = record.to_dict()
            records_list.append(record_data)

        return jsonify({'medical_records': records_list})

    except Exception as e:
        return jsonify({'error': str(e)})
