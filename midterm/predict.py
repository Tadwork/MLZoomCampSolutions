import pickle
import json
import logging
import os

import pandas as pd
from flask import Flask, jsonify, request, send_file, g

from training_data_helpers import find_index_of_closest_price, get_categorical_parameters

MODEL_PATH = os.getenv("MODEL_PATH", "./model.bin")
DV_PATH = os.getenv("DV_PATH", "./dv.bin")
TRAINING_DATA_PATH = os.getenv(
    "TRAINING_DATA_PATH", "./data/amazon_laptop_prices_v01_cleaned.csv"
)

logging.basicConfig(level=os.getenv('LOG_LEVEL',logging.DEBUG))
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route("/")
def index():
    return send_file("./static/index.html")


@app.route("/parameters")
def parameters():
    return jsonify(results=g.get("PARAMETER_OPTIONS"))


@app.route("/predict", methods=["POST"])
def predict():
    model = g.get("MODEL")
    dv = g.get("DV")
    training_data_records = g.get("TRAINING_DATA_RECORDS")
    # get the data
    data = request.get_json(force=True)
    val = [
        {
            "brand": data.get("brand") or "unknown",
            "screen_size": data.get("screen_size") or "unknown",
            "cpu": data.get("cpu") or "unknown",
            "OS": data.get("OS") or "unknown",
            "cpu_mfr": data.get("cpu_mfr") or "unknown",
            "graphics_type": data.get("graphics_type") or "unknown",
            "graphics_mfr": data.get("graphics_mfr") or "unknown",
            "harddisk_gb": data.get("harddisk_gb") or 0,
            "ram_gb": data.get("ram_gb") or 0,
        }
    ]
    X_val = dv.transform(val)
    prediction = model.predict(X_val)
    price = round(float(prediction[0]), 2)

    closest_idx = find_index_of_closest_price(training_data_records, price)
    logger.debug('closest record = %s at idx %s',json.dumps(training_data_records[closest_idx]),closest_idx)
    five_lower = training_data_records[max(closest_idx - 4,0) : closest_idx + 1]
    five_higher = training_data_records[closest_idx + 1: min(closest_idx + 6, len(training_data_records))]
    
    top10 = five_lower + [{**data, 'diff':0, 'price': price}] + five_higher

    output = {"price": price, "top10": top10}
    return jsonify(results=output)

def get_globals():
    config = {}
    training_data = pd.read_csv(TRAINING_DATA_PATH)
    config["TRAINING_DATA"] = training_data
    training_data_records = training_data.to_dict(orient="records")
    # replace np.nan with None
    for item in training_data_records:
        for key in item:
            if pd.isnull(item[key]):
                item[key] = None
    config["TRAINING_DATA_RECORDS"] = training_data_records
    
    #load the model and dict vectorizer
    with open(DV_PATH, "rb") as dv_raw:
        dv = pickle.load(dv_raw)
    config["DV"] = dv

    with open(MODEL_PATH, "rb") as model_raw:
        model = pickle.load(model_raw)
    config["MODEL"] = model

    # get all the unique categorical parameters
    parameter_options = get_categorical_parameters(training_data)
    config["PARAMETER_OPTIONS"] = parameter_options
    return config

class GlobalValuesMiddleware:
    def __init__(self, app):
        self.app = app
        self.global_values = None
        # Attach a before_request hook to the Flask application that will run in the application context
        app.before_request(self.before_request)

    def __call__(self, environ, start_response):
        return self.app(environ, start_response)

    def before_request(self):
        # This hook is executed within an application context, so it's safe to modify `g`
        if not self.global_values:
            self.global_values = get_globals()
        for key, value in self.global_values.items():
            setattr(g, key, value)


GlobalValuesMiddleware(app)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
