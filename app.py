# import os
# import numpy as np
# from flask import Flask, render_template, request
# from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import load_img, img_to_array
# from werkzeug.utils import secure_filename
# import tensorflow as tf
# import joblib

# # Setup Flask
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # ===============================
# # Load the CNN Model
# # ===============================
# model = load_model('image_recognition_model_latest.h5')

# # ===============================
# # Load the KNN Model
# # ===============================
# model_data = joblib.load('knn_model.joblib')

# knn_model = model_data['model']
# label_encoder = model_data['label_encoder']
# class_names = model_data['class_names']

# # Class labels (in training order)
# flower_names = [
#     'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper',
#     'corn', 'cucumber', 'daisy', 'dandelion', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon',
#     'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish',
#     'rose', 'soy beans', 'spinach', 'sunflower', 'sweetcorn', 'sweetpotato', 'tomato', 'tulip', 'turnip', 'watermelon'
# ]

# # ===============================
# # CNN Model Function
# # ===============================
# def predict_cnn(image_path):
#     input_image = load_img(image_path, target_size=(224, 224))  # ✅ Check your model input size
#     input_image_array = img_to_array(input_image)
#     input_image_expanded = np.expand_dims(input_image_array, axis=0)  # ✅ Batch dimension

#     predictions = model.predict(input_image_expanded)
#     result = predictions[0]

#     predicted_index = np.argmax(result)
#     predicted_class = flower_names[predicted_index]
#     confidence = round(float(result[predicted_index]) * 100, 2)

#     return predicted_class, confidence

# # ===============================
# # KNN Model Function
# # ===============================
# def predict_knn(image_path):
#     # Preprocess the image
#     img = load_img(image_path, target_size=(64, 64))  # ✅ Match to training size
#     img = img.convert('RGB')
#     img_array = img_to_array(img) / 255.0
#     img_flat = img_array.flatten().reshape(1, -1)  # Flatten and reshape for prediction

#     # Make prediction
#     prediction = knn_model.predict(img_flat)[0]
#     probabilities = knn_model.predict_proba(img_flat)[0]

#     predicted_class = label_encoder.inverse_transform([prediction])[0]
#     confidence = round(np.max(probabilities) * 100, 2)

#     return predicted_class, confidence


# # Get Started Page Route
# @app.route("/")
# def index():
#     return render_template("index.html")


# @app.route("/prediction", methods=["GET", "POST"])
# def prediction():
#     algorithm = ""
#     predicted_class = ""
#     confidence = ""
#     image_path = ""

#     if request.method == "POST":
#         selected_algorithm = request.form.get("selected_algorithm")
#         file = request.files.get("image")
#         app.logger.info(f"Selected Algorithm: {selected_algorithm}")
        
#         # Check if the file is provided
#         if file:
#             filename = secure_filename(file.filename)
#             image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(image_path)

#             # Check if the searched algorithm is 'CNN, KNN OR SVM'
#             match selected_algorithm.lower():
#                 # case 'SVM':
#                 #     predicted_class, confidence = classify_image(image_path)
#                 case 'knn':
#                     algorithm = "KNN"
#                     predicted_class, confidence = predict_knn(image_path)
#                     app.logger.info(f"Predicted Class: {predicted_class}")
#                     app.logger.info(f"Confidence: {confidence}%")

#                 case 'cnn':
#                     algorithm = "CNN"
#                     predicted_class, confidence = predict_cnn(image_path)
#                     app.logger.info(f"Predicted Class: {predicted_class}")
#                     app.logger.info(f"Confidence: {confidence}%")

#                 case _:
#                     algorithm = "CNN"
#                     predicted_class, confidence = predict_cnn(image_path)
#                     app.logger.info(f"Predicted Class: {predicted_class}")
#                     app.logger.info(f"Confidence: {confidence}%")

#     return render_template(
#         "get-Started.html",
#         algorithm=algorithm,
#         predicted_class=predicted_class,
#         confidence=confidence,
#         image=image_path
#     )


# # Get Started Page Route
# @app.route("/get_started")
# def get_started():
#     return render_template("get-Started.html")


# if __name__ == "__main__":
#     app.run(debug=True)


import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename
import joblib

# Setup Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ===============================
# Load CNN Model
# ===============================
model = load_model('image_recognition_model_latest.h5')

# ===============================
# Load KNN Model
# ===============================
model_data = joblib.load('knn_model.joblib')
knn_model = model_data['model']
label_encoder = model_data['label_encoder']
class_names = model_data['class_names']

# ===============================
# Load SVM Model
# ===============================
svm_model = joblib.load('svm_model_output/svm_model_compressed.pkl')
svm_scaler = joblib.load('svm_model_output/scaler.pkl')
svm_pca = joblib.load('svm_model_output/pca.pkl')
svm_label_encoder = joblib.load('svm_model_output/label_encoder.pkl')

# ===============================
# Class Labels (used by CNN)
# ===============================
flower_names = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper',
    'corn', 'cucumber', 'daisy', 'dandelion', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon',
    'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish',
    'rose', 'soy beans', 'spinach', 'sunflower', 'sweetcorn', 'sweetpotato', 'tomato', 'tulip', 'turnip', 'watermelon'
]

# ===============================
# CNN Model Function
# ===============================
def predict_cnn(image_path):
    input_image = load_img(image_path, target_size=(224, 224))
    input_image_array = img_to_array(input_image)
    input_image_expanded = np.expand_dims(input_image_array, axis=0)
    predictions = model.predict(input_image_expanded)
    result = predictions[0]

    predicted_index = np.argmax(result)
    predicted_class = flower_names[predicted_index]
    confidence = round(float(result[predicted_index]) * 100, 2)

    return predicted_class, confidence

# ===============================
# KNN Model Function
# ===============================
def predict_knn(image_path):
    img = load_img(image_path, target_size=(64, 64))
    img = img.convert('RGB')
    img_array = img_to_array(img) / 255.0
    img_flat = img_array.flatten().reshape(1, -1)

    prediction = knn_model.predict(img_flat)[0]
    probabilities = knn_model.predict_proba(img_flat)[0]

    predicted_class = label_encoder.inverse_transform([prediction])[0]
    confidence = round(np.max(probabilities) * 100, 2)

    return predicted_class, confidence

# ===============================
# ✅ SVM Model Function
# ===============================
def predict_svm(image_path):
    img = load_img(image_path, target_size=(32, 32))  # Match SVM training size
    img = img.convert('RGB')
    img_array = img_to_array(img) / 255.0
    img_flat = img_array.flatten().reshape(1, -1)

    img_scaled = svm_scaler.transform(img_flat)
    img_pca = svm_pca.transform(img_scaled)

    prediction = svm_model.predict(img_pca)[0]
    probabilities = svm_model.predict_proba(img_pca)[0]

    predicted_class = svm_label_encoder.inverse_transform([prediction])[0]
    confidence = round(np.max(probabilities) * 100, 2)

    return predicted_class, confidence

# ===============================
# Home Route
# ===============================
@app.route("/")
def index():
    return render_template("index.html")

# ===============================
# Prediction Route
# ===============================
@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    algorithm = ""
    predicted_class = ""
    confidence = ""
    image_path = ""

    if request.method == "POST":
        selected_algorithm = request.form.get("selected_algorithm")
        file = request.files.get("image")
        app.logger.info(f"Selected Algorithm: {selected_algorithm}")
        
        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            match selected_algorithm.lower():
                case 'cnn':
                    algorithm = "CNN"
                    predicted_class, confidence = predict_cnn(image_path)

                case 'knn':
                    algorithm = "KNN"
                    predicted_class, confidence = predict_knn(image_path)

                case 'svm':
                    algorithm = "SVM"
                    predicted_class, confidence = predict_svm(image_path)

                case _:
                    algorithm = "CNN"
                    predicted_class, confidence = predict_cnn(image_path)

            app.logger.info(f"Predicted Class: {predicted_class}")
            app.logger.info(f"Confidence: {confidence}%")

    return render_template(
        "get-Started.html",
        algorithm=algorithm,
        predicted_class=predicted_class,
        confidence=confidence,
        image=image_path
    )

# ===============================
# Get Started Page Route
# ===============================
@app.route("/get_started")
def get_started():
    return render_template("get-Started.html")

# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
