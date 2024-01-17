#!/usr/bin/env python
# coding: utf-8

import logging
import os
import json

from flask import Flask, jsonify, request, send_file
from apig_wsgi import make_lambda_handler

import tflite_runtime.interpreter as tflite
import numpy as np
from io import BytesIO
from urllib import request as url_request

from PIL import Image

def get_globals():
    config = {}
    #load the model and dict vectorizer
    # with open(DV_PATH, "rb") as dv_raw:
    #     dv = pickle.load(dv_raw)
    # config["DV"] = dv

    # with open(MODEL_PATH, "rb") as model_raw:
    #     model = pickle.load(model_raw)
    # config["MODEL"] = model

    # get all the unique categorical parameters
    # parameter_options = get_categorical_parameters(training_data)
    # config["PARAMETER_OPTIONS"] = parameter_options
    return config

model_globals = get_globals()

logging.basicConfig(level=os.getenv('LOG_LEVEL',logging.DEBUG))
logger = logging.getLogger(__name__)

def download_image(url):
    print(url)
    logger.info(f"Downloading image from {url}")
    with url_request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_image(img, target_size):
    img = prepare_image(img, target_size)
    img = np.array(img)
    img = img.astype('float32')
    # img /= 255.0
    img = np.expand_dims(img, axis=0)
    return img

interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def predict(img):
    X = preprocess_image(img, target_size=(150, 150))

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    
    float_predictions = preds[0].tolist()

    # return float_predictions
    [middle, old, young] = float_predictions
    output = {"young": young, "middle": middle, "old": old}
    return output

app = Flask(__name__)

@app.route("/")
def index():
    return send_file("./static/index.html")

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = json.loads(request.data.decode("utf-8"))
    img = download_image(data.get("url"))
    output = predict(img)
    return jsonify(results=output)

lambda_handler = make_lambda_handler(app)

# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=9696)
