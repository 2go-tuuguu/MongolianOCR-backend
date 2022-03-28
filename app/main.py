from flask import Flask, request, jsonify
import io
from PIL import Image

from app.torch_utils import transform_image, get_prediction

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        bytesOfImage = request.get_data()
        with open('image.jpeg', 'wb') as out:
            out.write(bytesOfImage)

        img = Image.open('image.jpeg')
        tensor = transform_image(img)
        prediction = get_prediction(tensor)
        data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}
        return jsonify(data)