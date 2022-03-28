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
        # try:
        imageData = request.get_data()
        return jsonify(imageData)
        img = Image.open(imageData)

        tensor = transform_image(img)
        prediction = get_prediction(tensor)
        data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}
        return jsonify(data)
        # except:
        #     return jsonify({'error': 'error during files'})