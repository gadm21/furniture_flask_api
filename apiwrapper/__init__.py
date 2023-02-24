#!flask/bin/python
from flask import Flask, request, jsonify
import json

import numpy as np
from PIL import Image
import io
import base64




from apiwrapper.utils import prepare_image, infer
from apiwrapper.exceptions import WrapperError


nomodelmessage = "No model loaded yet."\
    "Run api.load_model(model) before launching the api"


class Api(object):
	'''
    Class to wrap a ML model into a callable API
    model must have predict and predict_proba function
    Bonus : load an helper function with the model for
    the API to give info about the goal of the model
    '''
	def __init__(self, model=None, helper=None):
		self.app = Flask(__name__)
		self.model = model
		self.reload_routes()
		self.helper = helper
		self.register_errors()

	def register_errors(self):
		@self.app.errorhandler(WrapperError)
		def handle_invalid_usage(error):
			response = jsonify(error.to_dict())
			response.status_code = error.status_code
			return response

	def load_model(self, model, helper=None):
		self.model = model
		if helper:
			self.helper = helper
		self.reload_routes()

	def reload_routes(self):
		@self.app.route('/', methods=['GET'])
		def index():
			if self.helper:
				return self.helper
			else:
				message = "APIwrapper for ML models.\n"
				if self.model:
					return message + "Model loaded and ready."
				else:
					raise WrapperError(nomodelmessage)


		@self.app.route('/predict_image', methods=['POST'])
		def predict_image():
			# Decode the base64-encoded image string
			image_string = request.form['img']
			image_bytes = base64.b64decode(image_string)

			# Read the image data using Pillow
			image = Image.open(io.BytesIO(image_bytes))
			
			res = infer(image) 
			return res
			# return jsonify({'status': 'success', 'label': 4, 'class': 'dog'})



	def run(self, **kwargs):
		self.app.run(**kwargs)
