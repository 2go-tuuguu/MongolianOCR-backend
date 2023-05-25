from datetime import timedelta

# SQLALCHEMY_DATABASE_URI = 'postgresql://gen_user:tuguldur921@92.53.104.23/default_db'
SQLALCHEMY_DATABASE_URI = 'postgresql://postgres:tuguldur921@localhost/melmii'
SQLALCHEMY_TRACK_MODIFICATIONS = False
JWT_SECRET_KEY = '92c8aa53638cd2d7682018feadf253a5e54cc2d54e297302d9c7ccbdd1333b99'
JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)