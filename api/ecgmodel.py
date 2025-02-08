import uuid
from flask import Blueprint, request, jsonify
from firebase_admin import firestore
from PIL import Image
import numpy as np
import joblib
import pandas as pd
import pickle
from skimage.io import imread
from skimage import color
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, gaussian
from skimage.transform import resize
from skimage.metrics import structural_similarity
from skimage import measure
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
from natsort import natsorted

db = firestore.client()
user_Ref = db.collection('ecg')
doctor_patient_Ref = db.collection('doctor-patient')


# Load Model
loaded_model = joblib.load('api/model_ecg.pkl')

ecgmodel = Blueprint('ecgmodel', __name__)

def preprocess_image(image_path):
    image = imread(image_path)
    # image = imread('api/test/Abnormal/HB(29).jpg')
    image_gray = color.rgb2gray(image)
    image_gray = resize(image_gray, (1572, 2213))

    image1 = imread('api/test/PMI/PMI(53).jpg')
    image1 = color.rgb2gray(image1)
    image1 = resize(image1, (1572, 2213))

    image2 = imread('api/test/Abnormal/HB(5).jpg')
    image2 = color.rgb2gray(image2)
    image2 = resize(image2, (1572, 2213))

    image3 = imread('api/test/Normal/Normal(20).jpg')
    image3 = color.rgb2gray(image3)
    image3 = resize(image3, (1572, 2213))

    image4 = imread('api/test/Myocardial/MI(2).jpg')
    image4 = color.rgb2gray(image4)
    image4 = resize(image4, (1572, 2213))

    similarity_score = max(
      structural_similarity(image_gray, image1, data_range=image1.max() - image1.min()),
      structural_similarity(image_gray, image2, data_range=image2.max() - image2.min()),
      structural_similarity(image_gray, image3, data_range=image3.max() - image3.min()),
      structural_similarity(image_gray, image4, data_range=image4.max() - image4.min())
    )

    print(similarity_score)
    if similarity_score > 0.70:
        # plt.imshow(image)
        # plt.show()

        # Dividing the ECG leads
        Lead_1 = image[300:600, 150:643]
        Lead_2 = image[300:600, 646:1135]
        Lead_3 = image[300:600, 1140:1625]
        Lead_4 = image[300:600, 1630:2125]
        Lead_5 = image[600:900, 150:643]
        Lead_6 = image[600:900, 646:1135]
        Lead_7 = image[600:900, 1140:1625]
        Lead_8 = image[600:900, 1630:2125]
        Lead_9 = image[900:1200, 150:643]
        Lead_10 = image[900:1200, 646:1135]
        Lead_11 = image[900:1200, 1140:1625]
        Lead_12 = image[900:1200, 1630:2125]
        Lead_13 = image[1250:1480, 150:2125]
        Leads=[Lead_1,Lead_2,Lead_3,Lead_4,Lead_5,Lead_6,Lead_7,Lead_8,Lead_9,Lead_10,Lead_11,Lead_12,Lead_13]
        #plotting lead 1-12
        fig , ax = plt.subplots(4,3)
        fig.set_size_inches(10, 10)
        x_counter=0
        y_counter=0

        for x,y in enumerate(Leads[:len(Leads)-1]):
          if (x+1)%3==0:
            ax[x_counter][y_counter].imshow(y)
            ax[x_counter][y_counter].axis('off')
            ax[x_counter][y_counter].set_title("Leads {}".format(x+1))
            x_counter+=1
            y_counter=0
          else:
            ax[x_counter][y_counter].imshow(y)
            ax[x_counter][y_counter].axis('off')
            ax[x_counter][y_counter].set_title("Leads {}".format(x+1))
            y_counter+=1

        fig.savefig('Leads_1-12_figure.png')
        fig1 , ax1 = plt.subplots()
        fig1.set_size_inches(10, 10)
        ax1.imshow(Lead_13)
        ax1.set_title("Leads 13")
        ax1.axis('off')
        fig1.savefig('Long_Lead_13_figure.png')

        # Preprocessing leads
        fig2 , ax2 = plt.subplots(4,3)
        fig2.set_size_inches(10, 10)
        #setting counter for plotting based on value
        x_counter=0
        y_counter=0

        for x,y in enumerate(Leads[:len(Leads)-1]):
          #converting to gray scale
          grayscale = color.rgb2gray(y)
          #smoothing image
          blurred_image = gaussian(grayscale, sigma=0.9)
          #thresholding to distinguish foreground and background
          #using otsu thresholding for getting threshold value
          global_thresh = threshold_otsu(blurred_image)

          #creating binary image based on threshold
          binary_global = blurred_image < global_thresh
          #resize image
          binary_global = resize(binary_global, (300, 450))
          if (x+1)%3==0:
            ax2[x_counter][y_counter].imshow(binary_global,cmap="gray")
            ax2[x_counter][y_counter].axis('off')
            ax2[x_counter][y_counter].set_title("pre-processed Leads {} image".format(x+1))
            x_counter+=1
            y_counter=0
          else:
            ax2[x_counter][y_counter].imshow(binary_global,cmap="gray")
            ax2[x_counter][y_counter].axis('off')
            ax2[x_counter][y_counter].set_title("pre-processed Leads {} image".format(x+1))
            y_counter+=1
        fig2.savefig('Preprossed_Leads_1-12_figure.png')

        #plotting lead 13
        fig3 , ax3 = plt.subplots()
        fig3.set_size_inches(10, 10)
        #converting to gray scale
        grayscale = color.rgb2gray(Lead_13)
        #smoothing image
        blurred_image = gaussian(grayscale, sigma=0.7)
        #thresholding to distinguish foreground and background
        #using otsu thresholding for getting threshold value
        global_thresh = threshold_otsu(blurred_image)
        print(global_thresh)
        #creating binary image based on threshold
        binary_global = blurred_image < global_thresh
        ax3.imshow(binary_global,cmap='gray')
        ax3.set_title("Leads 13")
        ax3.axis('off')
        fig3.savefig('Preprossed_Leads_13_figure.png')

        # Extracting signals
        fig4 , ax4 = plt.subplots(4,3)
        fig4.set_size_inches(10, 10)
        x_counter=0
        y_counter=0
        for x,y in enumerate(Leads[:len(Leads)-1]):
          #converting to gray scale
          grayscale = color.rgb2gray(y)
          #smoothing image
          blurred_image = gaussian(grayscale, sigma=0.9)
          #thresholding to distinguish foreground and background
          #using otsu thresholding for getting threshold value
          global_thresh = threshold_otsu(blurred_image)

          #creating binary image based on threshold
          binary_global = blurred_image < global_thresh
          #resize image
          binary_global = resize(binary_global, (300, 450))
          #finding contours
          contours = measure.find_contours(binary_global,0.8)
          contours_shape = sorted([x.shape for x in contours])[::-1][0:1]
          for contour in contours:
            if contour.shape in contours_shape:
              test = resize(contour, (3060, 2))
          if (x+1)%3==0:
            ax4[x_counter][y_counter].invert_yaxis()
            ax4[x_counter][y_counter].plot(test[:, 1], test[:, 0],linewidth=1,color='black')
            ax4[x_counter][y_counter].axis('image')
            ax4[x_counter][y_counter].set_title("Contour {} image".format(x+1))
            x_counter+=1
            y_counter=0
          else:
            ax4[x_counter][y_counter].invert_yaxis()
            ax4[x_counter][y_counter].plot(test[:, 1], test[:, 0],linewidth=1,color='black')
            ax4[x_counter][y_counter].axis('image')
            ax4[x_counter][y_counter].set_title("Contour {} image".format(x+1))
            y_counter+=1
          #scaling the data and testing
          lead_no=x
          scaler = MinMaxScaler()
          fit_transform_data = scaler.fit_transform(test)
          Normalized_Scaled=pd.DataFrame(fit_transform_data[:,0], columns = ['X'])
          Normalized_Scaled=Normalized_Scaled.T
          #scaled_data to CSV
          Normalized_Scaled.to_csv('Scaled_1DLead_{lead_no}.csv'.format(lead_no=lead_no+1),index=False)

        fig4.savefig('Contour_Leads_1-12_figure.png')

        #lets try combining all 12 leads
        test_final=pd.read_csv('Scaled_1DLead_1.csv')
        location= '/'
        for files in natsorted(os.listdir(location)):
          if files.endswith(".csv"):
            if files!='Scaled_1DLead_1.csv':
                df=pd.read_csv('{}'.format(files))
                test_final=pd.concat([test_final,df],axis=1,ignore_index=True)

        # print(test_final)
        loaded_model = joblib.load('api/model_ecg.pkl')
        result = loaded_model.predict(test_final)
        print("Predictions:")
        if result[0] == 0:
            return "You ECG corresponds to Abnormal Heartbeat"

        if result[0] == 1:
            return "You ECG corresponds to Myocardial Infarction"

        if result[0] == 2:
            return "You ECG is Normal"

        if result[0] == 3:
            return "You ECG corresponds to History of Myocardial Infarction"
    else:
        return "Sorry, Our App won't be able to parse this image format right now!!! Please check the image input sample section for supported images"

@ecgmodel.route('/predict_ecg', methods=['POST'])
def predict_ecg_endpoint():
    try:
        # Get the file from the request
        file = request.files['file']
        
        # Save the file to a temporary location
        image_path = f'tmp/{uuid.uuid4()}.png'
        file.save(image_path)
        
        # Call the predict_ecg function
        prediction, probability = predict_ecg(image_path)
        
        # Return the prediction and probability as JSON response
        return jsonify({'prediction': prediction, 'probability': probability})
    except Exception as e:
        return jsonify({'error': str(e)})
    

@ecgmodel.route('/predict_ecg_test', methods=['POST'])
def predict_ecg_chk_endpoint():
    try:
        # Get the file from the request
        file = request.files['file']
        
        # Save the file to a temporary location
        image_path = f'tmp/{uuid.uuid4()}.png'
        file.save(image_path)
        
        # Call the predict_ecg function
        msg = preprocess_image(image_path)
        
        # Return the prediction and probability as JSON response
        return jsonify({'prediction': msg})
    except Exception as e:
        return jsonify({'error': str(e)})
