import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    UFC_DB = os.getenv('UFC_DB')
    # SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key') dont need until auth stuff