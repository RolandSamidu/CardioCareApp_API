import uuid
from flask import Blueprint, request, jsonify
from firebase_admin import firestore
import joblib
import pandas as pd
import numpy as np
from textblob import TextBlob
import requests

db = firestore.client()
ph_Ref = db.collection('pharmacies')
comments_Ref = db.collection('comments')
doctor_patient_Ref = db.collection('doctor-patient')

# Load Model
ph_model = joblib.load('api/model_heart.joblib')

pharmaciesapi = Blueprint('pharmaciesapi', __name__)

def get_comments_for_pharmacy(pharmacy_id):
    # Retrieve comments for a specific pharmacy
    comments = comments_Ref.where('pharmacy', '==', pharmacy_id).stream()
    
    comments_list = []
    for comment in comments:
        # print(comment)
        comment_data = comment.to_dict()
        comments_list.append(comment_data)
    # print(comments_list)
    return comments_list

@pharmaciesapi.route('/post_comment', methods=['POST'])
def post_comment():
    try:
        # Get data from the request
        data = request.get_json()
        print(data)

        # Extract necessary information
        pharmacy_id = data.get('pharmacy_id')
        comment_text = data.get('comment_text')

        # Perform sentiment analysis using TextBlob
        comment_blob = TextBlob(comment_text)
        sentiment_score = comment_blob.sentiment.polarity

        # Create a new comment document
        comment_id = str(uuid.uuid4())
        comment_data = {
            'id': comment_id,
            'pharmacy': pharmacy_id,
            'comment': comment_text,
            'sentiment_score': sentiment_score
        }

        comments_Ref.document(comment_id).set(comment_data)

        return jsonify({'message': 'Comment posted successfully'})

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})
    
@pharmaciesapi.route('/sort_pharmacies_by_polarity', methods=['POST'])
def sort_pharmacies_by_polarity():
    try:
        data = request.get_json()
        pharmacies_list = data.get('pharmacies_list')

        # Calculate total polarity for each pharmacy
        sorted_pharmacies = []
        for pharmacy_data in pharmacies_list:
            comments_list = pharmacy_data.get('comments', [])
            total_polarity = sum(comment.get('sentiment_score', 0) for comment in comments_list)
            pharmacy_data['total_polarity'] = total_polarity
            sorted_pharmacies.append(pharmacy_data)

        # Sort pharmacies based on total polarity (highest to lowest)
        sorted_pharmacies = sorted(sorted_pharmacies, key=lambda x: x['total_polarity'], reverse=True)

        return jsonify(sorted_pharmacies)

    except Exception as e:
        return jsonify({'error': str(e)})

@pharmaciesapi.route('/get_pharmacies_comments', methods=['GET', 'POST'])
def get_pharmaciesComments():
    if request.method == 'POST':
        # Handle POST request
        data = request.json
        location = data.get('location')
        radius = data.get('radius')
    else:
        # Handle GET request
        location = request.args.get('location', '6.838402833673086, 80.00356651648937')
        radius = request.args.get('radius', '10000')

    # Make a request to the Google Places API
    api_key = 'AIzaSyDAsJYZSQ92_NQAz9kiSpW1XpyuCxRl_uI'
    places_url = f'https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={location}&radius={radius}&types=pharmacy&key={api_key}'
    response = requests.get(places_url)
    data = response.json()

    # Extract relevant information from the API response
    pharmacies = [{'name': place['name'], 'location': place['geometry']['location']} for place in data['results']]

    # Combine with comments from Firestore
    for pharmacy in pharmacies:
        # Get comments for the current pharmacy
        comments_list = get_comments_for_pharmacy(pharmacy['name'])
        
        # Analyze sentiment for each comment and calculate the overall score
        total_polarity = 0
        total_subjectivity = 0
        for comment_data in comments_list:
            comment_text = comment_data.get('comment', '')
            comment_blob = TextBlob(comment_text)
            
            # Summing up the polarity and subjectivity
            total_polarity += comment_blob.sentiment.polarity
            total_subjectivity += comment_blob.sentiment.subjectivity
        
        # Calculate average polarity and subjectivity
        avg_polarity = total_polarity / len(comments_list) if len(comments_list) > 0 else 0
        avg_subjectivity = total_subjectivity / len(comments_list) if len(comments_list) > 0 else 0
        
        # Add sentiment scores to the pharmacy data
        pharmacy['sentiment_score'] = {
            'average_polarity': avg_polarity,
            'average_subjectivity': avg_subjectivity
        }
        
        # Add comments to the pharmacy data
        pharmacy['comments'] = comments_list

    return jsonify({'pharmacies': pharmacies})