import requests 

# https://your-heroku-app-name.herokuapp.com/predict
# http://localhost:5000/predict
resp = requests.post("http://127.0.0.1:5000/predict", files={'file': open('seven.png', 'rb')})

print(resp.text)