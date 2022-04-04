from flask import Flask, request

from app.ocr import recognize

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

        # tensor = transform_image(bytesOfImage)
        # prediction = get_prediction(tensor)
        # data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}

        result, success = recognize('image.jpeg')
        if success:
            return {'recognized_text': result,
                    'message': 'Success' }
        else:
            return {'message': "No text was recognized."}