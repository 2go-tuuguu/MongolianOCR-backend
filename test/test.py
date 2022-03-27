import requests 

# https://your-heroku-app-name.herokuapp.com/predict
# http://localhost:5000/predict
resp = requests.post("https://pytorch-flask-2go.herokuapp.com/predict", files={'file': open('eight.png', 'rb')})

print(resp.text)