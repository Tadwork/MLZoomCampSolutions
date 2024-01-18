#!/usr/bin/env python
# coding: utf-8

import logging
import os

from flask import Flask, jsonify, request, send_file
from apig_wsgi import make_lambda_handler

import tflite_runtime.interpreter as tflite
import numpy as np
from io import BytesIO
from urllib import request as url_request

from PIL import Image

def get_globals():
    config = {}
    MODEL_PATH = os.getenv("MODEL_PATH", "model.tflite")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    config["MODEL"] = (
        interpreter,
        input_index,
        output_index
    )

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
    img = np.expand_dims(img, axis=0)
    return img

def predict(img):
    X = preprocess_image(img, target_size=(150, 150))
    (
        interpreter,
        input_index,
        output_index
    )  = model_globals["MODEL"]
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    
    float_predictions = preds[0].tolist()

    # return float_predictions
    [middle, old, young] = float_predictions
    output = {"young": young, "middle": middle, "old": old}
    return {
            "raw": output,
            "prediction": max(output, key=output.get),
        }

app = Flask(__name__)

@app.route("/")
def index():
    return send_file("./static/index.html")

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    # print(request.environ)
    img = download_image(request.json.get("url"))
    output = predict(img)
    return jsonify(results=output)

lambda_handler = make_lambda_handler(app)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
