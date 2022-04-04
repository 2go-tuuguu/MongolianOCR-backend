from flask import Flask, request

from app.ocr import recognize

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        bytesOfImage = request.get_data()
        with open('image.jpeg', 'wb') as out:
            out.write(bytesOfImage)

        result, success = recognize('image.jpeg')
        if success:
            return {'recognized_text': result,
                    'message': 'Success' }
        else:
            return {'message': "No text was recognized."}