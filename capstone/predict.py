#!/usr/bin/env python
# coding: utf-8

import logging
import os

from flask import Flask, jsonify, request, send_file
from apig_wsgi import make_lambda_handler

import numpy as np
from io import BytesIO
from urllib import request as url_request

from PIL import Image


def get_globals():
    config = {}
    
    LAMBDA_TASK_ROOT = os.getenv("LAMBDA_TASK_ROOT")
    if LAMBDA_TASK_ROOT is not None:
        # We're running in Lambda
        import tflite_runtime.interpreter as tflite
        MODEL_PATH = os.getenv("MODEL_PATH", "model.tflite")
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()

        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]
        config["MODEL"] = (interpreter, input_index, output_index)
    else:
        from tensorflow import keras
        MODEL_PATH = os.getenv("MODEL_PATH", "capstone/model.hdf5")
        model = keras.models.load_model(MODEL_PATH)
        config["MODEL"] = model

    return config


model_globals = get_globals()

logging.basicConfig(level=os.getenv("LOG_LEVEL", logging.DEBUG))
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
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess_image(img, target_size):
    img = prepare_image(img, target_size)
    img = np.array(img)
    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)
    return img


def predict(img):
    img = preprocess_image(img, target_size=(150, 150))
    LAMBDA_TASK_ROOT = os.getenv("LAMBDA_TASK_ROOT")
    if LAMBDA_TASK_ROOT is not None:
        # We're running in Lambda
        (interpreter, input_index, output_index) = model_globals["MODEL"]
        interpreter.set_tensor(input_index, img)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_index)
    else:
        model = model_globals["MODEL"]
        preds = model.predict(img)
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
    print(request.files)
    if 'file' in request.files:
        img = Image.open(request.files["file"])
    else:
        url = request.json.get("url")
        img = download_image(url)
    output = predict(img)
    return jsonify(results=output)


lambda_handler = make_lambda_handler(app)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
