import os
from flask import Blueprint, request, jsonify
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
import yolov5
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# import cloudinary.api
# import cloudinary.uploader

dietmodel = Blueprint('dietmodel', __name__)
model = yolov5.load('api/weights/last.pt')
model.conf = 0.25
model.iou = 0.45
model.agnostic = False
model.multi_label = False
model.max_det = 1000

items = [
    "bread", "bun", "cup_cake", "cake", "short_eats", "banana", "avocado", 
    "mango", "apple", "wood_apple", "watermelon", "lemon", "carrot", "pumpkin", 
    "potato", "tomato", "onion", "garlic", "leeks", "chili", "meat", "fish", 
    "egg", "curry", "mix_salad", "sausages", "banana_blossom", "brinjal", 
    "papadum", "kakiri", "durian", "donut", "rice", "cutlet", "beans", 
    "bitter_gourd", "broccoli", "sweet_potato", "beetroot", "okra"
]

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

def predict(image):
    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        results = model(input_tensor)

    # Process the output
    predictions = []

    # Iterate over the detections
    for det in results[0]:
        # Extract bounding box coordinates, confidence, and class scores
        xmin, ymin, xmax, ymax, _, confidence, *class_scores = det.tolist()

        # Check and correct bounding box coordinates if needed
        xmin, xmax = min(xmin, xmax), max(xmin, xmax)
        ymin, ymax = min(ymin, ymax), max(ymin, ymax)

        # Find the class with the highest score
        max_score_index = class_scores.index(max(class_scores))
        max_score = class_scores[max_score_index]

        # Append the prediction to the list
        predictions.append({
            'box': [xmin, ymin, xmax, ymax],
            'confidence': confidence * max_score,  # Multiply objectness score with the class score
            'class_id': max_score_index
        })

    # Get labels and count
    labels = [f"Object {prediction['class_id']}" for prediction in predictions]
    num_objects = len(predictions)

    # Create a copy of the original image to draw bounding boxes
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    for prediction in predictions:
        box = prediction['box']
        draw.rectangle(box, outline="red", width=1)

    # Save the annotated image with bounding boxes in the 'tmp' folder
    annotated_image.save('tmp/output_image.jpg')

    return {'labels': labels, 'num_objects': num_objects}



@dietmodel.route('/predict-foods', methods=['POST'])
def predict_endpoint():
    try:
        # Assuming you're receiving an image file in the request
        image = request.files['image']

        # Load the image using PIL
        pil_image = Image.open(image)

        # Convert the image to RGB (remove alpha channel)
        pil_image = pil_image.convert('RGB')

        # Perform prediction
        result = predict(pil_image)

        # result.save(save_dir='tmp/')

        # Return the prediction as JSON
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

# RESERVED
@dietmodel.route('/predict-foods-upload', methods=['POST'])
def predict_endpoint2():
    try:
        # Images
        imgs = request.files['image']
        pil_image = Image.open(imgs)

        pil_image = pil_image.resize((640, 640))

        # Inference
        results = model(pil_image)

        # Access predictions
        predictions = results.pred[0]

        # Initialize recognized objects dictionary
        recognized_objects = {}

        # Loop through each prediction
        for pred in predictions:
            # Extract the index of the predicted class
            class_index = int(pred[5])

            # Retrieve the name of the predicted class from the items array
            class_name = items[class_index]

            # If the class name already exists in the recognized objects dictionary, increment the count
            if class_name in recognized_objects:
                recognized_objects[class_name] += 1
            # Otherwise, initialize the count to 1
            else:
                recognized_objects[class_name] = 1

        # Print or return the recognized objects
        print("Recognized Objects:")
        print(recognized_objects)

        return jsonify(recognized_objects)

    except Exception as e:
        return jsonify({'error': str(e)})


# @dietmodel.route('/predict-foods-upload', methods=['POST'])
# def predict_endpoint2():
#     try:
#         # Images
#         imgs = request.files['image']
#         pil_image = Image.open(imgs)

#         pil_image = pil_image.resize((360, 360))

#         # Inference
#         results = model(pil_image)

#         # Save the results to a file using torch.save()
#         torch.save(results, 'results.pt')
        
#         # Extract labels from the results
#         labels = results.names

#         # Access predictions
#         predictions = results.pred[0]

#         # Process the output
#         recognized_objects = []

#         # Print or return the recognized objects
#         print("recognized_objects")
#         print(predictions)
#         # print(predictions)
#         return jsonify({'success': True})

#     except Exception as e:
#         return jsonify({'error': str(e)})
