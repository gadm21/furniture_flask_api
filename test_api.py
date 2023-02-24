import requests
from os.path import join
from apiwrapper.utils import dataset_path
import base64

# Set API endpoint URL
url = "http://127.0.0.1:3000/predict_image"

# Load image file
image_path = join(dataset_path, 'test', 'Bed', 'Baxton Studio Adela Modern and Contemporary Grey Finished Wood Queen Size Platform Bed.jpg')

with open(image_path, 'rb') as f:
    image_bytes = f.read()

# Encode the image data as a base64 string
image_string = base64.b64encode(image_bytes).decode('utf-8')

# Set up the request data
data = {'img': image_string}

response = requests.post(url, data = data) 

# check the status code
if response.status_code == 200:
    print(response.json())
else:
    print("Error , status code is ", response.status_code)
