from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime, timedelta
import bcrypt
import secrets

from ocr import recognize

from config import SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS, JWT_SECRET_KEY, JWT_ACCESS_TOKEN_EXPIRES

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = SQLALCHEMY_TRACK_MODIFICATIONS
app.config['JWT_SECRET_KEY'] = JWT_SECRET_KEY
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = JWT_ACCESS_TOKEN_EXPIRES
CORS(app)

db = SQLAlchemy(app) 
jwt = JWTManager(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    api_keys = db.relationship('ApiKey', backref='user', lazy=True)

    def set_password(self, password):
        password_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(password_bytes, salt).decode('utf-8')
        self.password_hash = hashed_password

    def check_password(self, password):
        password_bytes = password.encode('utf-8')
        hashed_password_bytes = self.password_hash.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_password_bytes)

class ApiKey(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    key = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)

    def is_valid(self):
        return self.expires_at > datetime.utcnow()

    def to_dict(self):
        return {
            'id': self.id,
            'prefix': self.key[:15],
            'name': self.name,
            'created_at': self.created_at,
            'expires_at': self.expires_at,
        }

@app.route('/ocr', methods=['POST'])
@jwt_required()
def ocr():
    if request.method == 'POST':
        file = request.files['file']
        file.save('image.jpeg')

        ocrType = request.headers.get('ocrType')

        if ocrType == 'Printed' or ocrType == 'Handwritten':
            result, success = recognize('image.jpeg', printedOrHandwritten=ocrType)
        else:
            return {'message': 'Invalid text-type'}, 400

        if success:
            return {'recognized_text': result, 'message': 'Success'}
        else:
            return {'message': 'No text was recognized.'}, 400

@app.route('/api/ocr', methods=['POST'])
def api_ocr():
    if request.method == 'POST':
        api_key = request.headers.get('API-Key')

        # Check if the API key exists and is valid
        api_key_obj = ApiKey.query.filter_by(key=api_key).first()
        if api_key_obj is None or not api_key_obj.is_valid():
            return {'message': 'Invalid API key'}, 401

        file = request.files['file']
        file.save('image.jpeg')

        ocrType = request.headers.get('ocrType')

        if ocrType == 'Printed' or ocrType == 'Handwritten':
            result, success = recognize('image.jpeg', printedOrHandwritten=ocrType)
        else:
            return {'message': 'Invalid text-type'}, 400

        if success:
            return {'recognized_text': result, 'message': 'Success'}
        else:
            return {'message': 'No text was recognized.'}, 400

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    if not email or not password:
        return {'message': 'Email and password are required.'}, 400

    if User.query.filter_by(email=email).first():
        return {'message': 'Email already exists.'}, 400

    user = User(email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    return {'message': 'User created successfully.'}, 201


@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    if not email or not password:
        return {'message': 'Email and password are required.'}, 400

    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return {'message': 'Invalid email or password.'}, 401

    access_token = create_access_token(identity=user.id, expires_delta=timedelta(hours=24))
    return {'access_token': access_token}, 200

@app.route('/user', methods=['GET', 'DELETE'])
@jwt_required()  # Requires authentication with a valid access token
def user():
    current_user_id = get_jwt_identity()

    if request.method == 'GET':
        # Find the user in the database
        user = User.query.filter_by(id=current_user_id).first()
        if not user:
            return {'message': 'User not found.'}, 404

        # Return user email
        return {'email': user.email}, 200

    elif request.method == 'DELETE':
        # Find the user in the database
        user = User.query.filter_by(id=current_user_id).first()
        if not user:
            return {'message': 'User not found.'}, 404

        # Delete all API keys associated with the user
        ApiKey.query.filter_by(user_id=current_user_id).delete()
        # Delete the user account
        db.session.delete(user)

        db.session.commit()

        return {'message': 'Account deleted successfully.'}, 200

@app.route('/api-keys', methods=['POST'])
@jwt_required()
def create_api_key():
    user_id = get_jwt_identity()
    user = User.query.filter_by(id=user_id).first()

    name = request.json['api_key_name']

    if len(user.api_keys) >= 3:
        return {'message': 'You have reached the maximum number of API keys.'}, 400

    while True:
        key = secrets.token_urlsafe(32)
        if not ApiKey.query.filter_by(key=key).first():
            break

    expires_at = datetime.utcnow() + timedelta(days=30)
    api_key = ApiKey(user_id=user.id, key=key, name=name, expires_at=expires_at)
    db.session.add(api_key)
    db.session.commit()

    return {'message': 'API key created successfully.', 'api_key': key}, 201

@app.route('/api-keys', methods=['GET'])
@jwt_required()
def get_api_keys():
    user_id = get_jwt_identity()
    user = User.query.filter_by(id=user_id).first()

    api_keys = [api_key.to_dict() for api_key in user.api_keys if api_key.is_valid()]
    return {'api_keys': api_keys}, 200

@app.route('/api-keys/<api_key_id>', methods=['DELETE'])
@jwt_required()
def delete_api_key(api_key_id):
    user_id = get_jwt_identity()
    user = User.query.filter_by(id=user_id).first()

    api_key = ApiKey.query.filter_by(id=api_key_id, user=user).first()
    if not api_key:
        return {'message': 'API key not found.'}, 404

    db.session.delete(api_key)
    db.session.commit()

    return {'message': 'API key deleted successfully.'}, 200


@app.errorhandler(400)
def bad_request(error):
    return {'message': 'Bad request.'}, 400

@app.errorhandler(401)
def unauthorized(error):
    return {'message': 'Unauthorized.'}, 401

@app.errorhandler(404)
def not_found(error):
    return {'message': 'Not found.'}, 404

@app.errorhandler(500)
def internal_server_error(error):
    return {'message': 'Internal server error.'}, 500

if __name__ == '__main__':
    db.create_all()
    app.run()

