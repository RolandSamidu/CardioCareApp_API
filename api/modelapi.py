import uuid
from flask import Blueprint,request,jsonify,Flask,Response
from firebase_admin import firestore
from keras.models import load_model
from keras.preprocessing import image
import torch
import torch.nn as nn
from torchvision import transforms
from api.model1 import ConvNet_1
from PIL import Image
from torchvision import datasets
from torch.utils.data import DataLoader
import cv2
import numpy as np
from scipy.signal import find_peaks
import os
from PIL import Image

db=firestore.client()
user_Ref=db.collection('predictions')

modelapi=Blueprint('modelapi',__name__)

# Load trained model weights.
model = ConvNet_1()  
model.load_state_dict(torch.load('api/model_1_state_dict.pth'))

model.eval()

transform         = transforms.Compose(
                                       [transforms.Resize([120,120]),
                                        transforms.Grayscale(), 
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5), (0.5))
                                       ])

def predict_image(image_path, model, transform):
    # Load and preprocess the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add a batch dimension

    # Make predictions
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        
    return predicted.item()  # Return the predicted class label as an integer

@modelapi.route('/save',methods=['POST'])
def createrecord():
    try:
        id=uuid.uuid4()
        user_Ref.document(id.hex).set(request.json)
        return jsonify({"success" : True}),200
    except Exception as e:
        return f"An Error Occurs : {e}"
    
    
@modelapi.route('/previous',methods=['GET'])
def retriverecords():
    column_name = request.args.get('name')
    print(column_name)
    # Replace 'your_collection' with the name of your Firestore collection
    # This query filters documents where 'column_name' equals a specific value
    data = db.collection('predictions').where('user', '==', column_name).get()
    
    data_dict = {doc.id: doc.to_dict() for doc in data}
    return jsonify(data_dict)

@modelapi.route('/all',methods=['GET'])
def retriveallrecords():

    # Replace 'your_collection' with the name of your Firestore collection
    # This query filters documents where 'column_name' equals a specific value
    data = db.collection('predictions').get()
    
    data_dict = {doc.id: doc.to_dict() for doc in data}
    return jsonify(data_dict)
    
@modelapi.route('/toapprove',methods=['GET'])
def retriveforverifyrecords():
   
    # Replace 'your_collection' with the name of your Firestore collection
    # This query filters documents where 'column_name' equals a specific value
    data = db.collection('predictions').where('DoctorVeri', '==', 'To Be Confirm').get()
    
    data_dict = {doc.id: doc.to_dict() for doc in data}
    return jsonify(data_dict)
    
    
@modelapi.route('/approvedlist',methods=['GET'])
def verifiedrecords():
   
    # Replace 'your_collection' with the name of your Firestore collection
    # This query filters documents where 'column_name' equals a specific value
    data = db.collection('predictions').where('DoctorVeri', '!=', 'To Be Confirm').get()
    
    data_dict = {doc.id: doc.to_dict() for doc in data}
    return jsonify(data_dict)

@modelapi.route('/update_doctorveri', methods=['POST'])
def update_doctor_veri():
    data = request.json  # Get user input as JSON
    print("Received Data:", data)

    if 'user' in data and 'Date' in data and 'DoctorVeri' in data:
        user = data['user']
        date = data['Date']
        doctor_veri = data['DoctorVeri']

        # Query Firestore for the specific user and date
        query = db.collection('predictions').where('user', '==', user).where('Date', '==', date)
        docs = query.stream()

        for doc in docs:
            doc_ref = db.collection('predictions').document(doc.id)
            doc_ref.update({'DoctorVeri': doctor_veri})
            #print("Received Data:", doc_ref)

        return jsonify({"message": "DoctorVeri updated successfully"})
    else:
        return jsonify({"error": "Invalid input data"}), 400
  

# @modelapi.route("/upload", methods=["POST"])
# def get_submitOutput():
#     if request.method=="POST":
#         img=request.files['my_image']
        
#         img_path = "C:/Users/TempO/OneDrive/Desktop/flask_api/api_fl/static" + img.filename
#         img.save(img_path)
        
#         p=predict_image(img_path, model, transform)
        
#     return jsonify({
#         'prediction' : p      
# })
    
# @modelapi.route("/uploadoriginal", methods=["POST"])
# def get_submitOutput():
#     if request.method=="POST":
#         img=request.files['my_image']
        
#         img_path = "C:/Users/TempO/OneDrive/Desktop/flask_api/api_fl/static" + img.filename
#         img.save(img_path)
        
#         original_image = cv2.imread(img_path)
        
#         gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        
#         _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
#         edges = cv2.Canny(binary_image, 30, 70)
        
#         roi_x = 40  # X-coordinate of the top-left corner of ROI
#         roi_y = 70  # Y-coordinate of the top-left corner of ROI
#         roi_width = 250  # Width of ROI
#         roi_height = 90 # Height of ROI
#         roi = binary_image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
        
#         pathof="C:/Users/TempO/OneDrive/Desktop/flask_api/api_fl/static/roi1.png"
#         cv2.imwrite(pathof, roi)
        
        
#         p=predict_image(pathof, model, transform)
        
#     return jsonify({
#         'prediction' : p      
# })
    
@modelapi.route("/uploadoriginalcompatible", methods=["POST"])
def get_submitOutput():
    if request.method=="POST":
        img=request.files['my_image']
        
        img_path = "./static" + img.filename
        img.save(img_path)
        
        original_image = cv2.imread(img_path)
        
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        
        _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        edges = cv2.Canny(binary_image, 30, 70)
        
        roi_x = 40  # X-coordinate of the top-left corner of ROI
        roi_y = 70  # Y-coordinate of the top-left corner of ROI
        roi_width = 250  # Width of ROI
        roi_height = 90 # Height of ROI
        roi = binary_image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
        
        pathof="./static/roi1.png"
        cv2.imwrite(pathof, roi)
        
        # Load the ROI image
        roi_image = cv2.imread(pathof, cv2.IMREAD_GRAYSCALE)

        # Find the high peaks in the image
        peaks, _ = find_peaks(roi_image.flatten(), height=100)  # You can adjust the height parameter as needed

        # Define a margin for cropping around the peaks
        margin = 10

        # Initialize variables to keep track of the cropping coordinates
        prev_peak = 0

        #peaks croped images
        i=0
        
        segfinal =""
        filenamefinal=""

        # Iterate through the detected peaks
        for peak in peaks:
            # Calculate the cropping coordinates based on the peak location and margin
            x1 = max(0, prev_peak - margin)
            x2 = min(roi_image.shape[1], peak + margin)
    
            # Crop the segment
            segment = roi_image[:, x1:x2]

            # Save the segment with a unique filename
            segment_filename = f'segment_{prev_peak}-{peak}.png'
            if segment_filename != '' and segment.size != 0:
                i=i+1
                if(i==2):
                    cv2.imwrite(segment_filename, segment)
                    save_path = "./static/"
                    cv2.imwrite(os.path.join(save_path, segment_filename), segment)  # Save to the specific path
                    segfinal =segment
                    filenamefinal=segment_filename
    
            # Update the previous peak
                prev_peak = peak

            # Crop the last segment if needed
                if prev_peak < roi_image.shape[1]:
                    x1 = max(0, prev_peak - margin)
                    x2 = roi_image.shape[1]
                    last_segment = roi_image[:, x1:x2]
                    last_segment_filename = f'last_segment_{prev_peak}-{roi_image.shape[1]}.png'
                    if(i==3):
                        cv2.imwrite(last_segment_filename, last_segment)

        newpath = "./static/" + filenamefinal
        p=predict_image(newpath, model, transform)
        
    return jsonify({
        'prediction' : p      
})
 
@modelapi.route("imagesave", methods=["GET"])   
def upload_image():
    # Read the image file using OpenCV
    image = cv2.imread("C:/Users/TempO/OneDrive/Desktop/python_firebase/segment_51-139.png")

    if image is not None:
        # Convert the image to bytes
        _, image_bytes = cv2.imencode('.jpg', image)

        # Set the appropriate content type in the response headers
        response = Response(image_bytes.tobytes(), content_type='image/jpeg')

        # Optionally, you can set other response headers, such as 'Content-Disposition' to suggest a filename
        # response.headers['Content-Disposition'] = 'attachment; filename=image.jpg'

        return response
    else:
        return 'Failed to read the image file'

    
    


