import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename
import joblib

# ===============================
# Flask App Setup
# ===============================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ===============================
# Load the KNN Model
# ===============================
model_data = joblib.load('knn_model.joblib')

knn_model = model_data['model']
label_encoder = model_data['label_encoder']
class_names = model_data['class_names']


# ===============================
# Image Classification Function
# ===============================
def classify_image(image_path):
    # Preprocess the image
    img = load_img(image_path, target_size=(64, 64))  # ✅ Match to training size
    img = img.convert('RGB')
    img_array = img_to_array(img) / 255.0
    img_flat = img_array.flatten().reshape(1, -1)  # Flatten and reshape for prediction

    # Make prediction
    prediction = knn_model.predict(img_flat)[0]
    probabilities = knn_model.predict_proba(img_flat)[0]

    predicted_class = label_encoder.inverse_transform([prediction])[0]
    confidence = round(np.max(probabilities) * 100, 2)

    return predicted_class, confidence


# ===============================
# Flask Routes
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    predicted_class = ""
    confidence = ""
    image_path = ""

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            predicted_class, confidence = classify_image(image_path)

    return render_template(
        "index.html",
        predicted_class=predicted_class,
        confidence=confidence,
        image=image_path
    )


# ===============================
# Run the Flask App
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
