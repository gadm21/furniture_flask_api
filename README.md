# Furniture Image Classification API
Flask API for classifying furniture images. Takes an image as input and returns the predicted label and class as output.

## Dataset
The dataset provided for this task consists of 300 images of furniture items, divided into three classes: "Bed", "Sofa", and "Chair".

## Model
The model used for this task is a deep learning classification model built using TensorFlow Keras. The model was trained on the dataset using the flow_from_directory() function from TensorFlow's data generator to directly read images from the directory. The dataset was split into two directories, "train" and "test", each with all classes with a 80/20 split. The model was trained for 10 epochs with a batch size of 32. 

## API
The API was built using Python and Flask. The API server is run by executing python run_api.py, which imports the trained model and initializes the Flask app. The app's endpoints are located in `apiwrapper.py`.

To use the API, a client sends an image in the form of a base64-encoded string to the /predict_image endpoint. The API then returns a JSON response with a status field, a label field containing the predicted label, and a class field containing the class of the predicted label.

To test the API, run python `test_api.py`. This script loads an example image from the dataset, encodes it, sends it to the API, and prints the response.

## Docker
The API is containerized using Docker. The Dockerfile contains the necessary commands to build the Docker image and the `run_api.py` file is set as the entry point.

To build the Docker image:
```
docker build -t furniture-flask-api .
```

To run the Docker container:
```
docker run -p 3000:3000 furniture-flask-api
```




## Endpoints

 * **/** [GET] : info about the model
 * **/predict_image** [POST] : return the prediction of the model for the given input image.
