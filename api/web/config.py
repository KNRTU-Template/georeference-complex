import logging
import os

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO, format=u'%(filename)s:%(lineno)d #%(levelname)-8s [%(asctime)s] - %(name)s - %(message)s',
)

load_dotenv()

SECRET_KEY = os.getenv('SECRET_KEY')

HOST = os.getenv('HOST', 'localhost')
PORT = int(os.getenv('PORT', 8000))

ALGORITHM_API_HOST = os.getenv('ALGORITHM_API_HOST')

DATABASE_USER = os.getenv('POSTGRES_USER')
DATABASE_PASSWORD = os.getenv('POSTGRES_PASSWORD')
DATABASE_HOST = os.getenv('DBHOST')
DATABASE_NAME = os.getenv('POSTGRES_DB')
DATABASE_PORT = os.getenv('POSTGRES_PORT')

DATABASE_URL = (
    f'postgresql+asyncpg://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}'
)

CSV_FILENAME = 'csv/coords.csv'
