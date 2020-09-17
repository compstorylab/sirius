print('Establishing environment credentials...')
import os
os.environ['KAGGLE_USERNAME'] = "compstorylab" # For demo use only
os.environ['KAGGLE_KEY'] = "f3f9220e8d85a7427864bd4f96f23ff2" # Please use your own API token if making frequent queries

print('Querying Kaggle API...')
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.competition_download_file('widsdatathon2020','training_v2.csv')

print('Extracing zip file...')
import zipfile
with zipfile.ZipFile('training_v2.csv.zip', 'r') as zip_ref:
    zip_ref.extractall()

print('Renaming csv and removing zip file...')
try:
    os.rename('training_v2.csv','data.csv')
    os.remove('training_v2.csv.zip')
except:
    print('Error renaming files')