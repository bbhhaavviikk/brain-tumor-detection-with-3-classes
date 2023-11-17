# app.py
from flask import Flask, render_template, request
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)

# Load the model and labels
model = load_model("keras_second_model.h5", compile=False)
class_names = open("second_labels.txt", "r").readlines()

def process_image(image_path):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    return data

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index2.html', message='No file part')

    file = request.files['file']
    
    if file.filename == '':
        return render_template('index2.html', message='No selected file')

    try:
        file_path = "uploads/" + file.filename
        file.save(file_path)
        data = process_image(file_path)

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        result_text = f"Class: {class_name[2:]}, Confidence Score: {confidence_score:.2f}"
        return result_text
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
