import pickle
import os

import pandas as pd
from flask import Flask, jsonify, request, send_file
# import xgboost as xgb

import clean 

MODEL_PATH = os.getenv('MODEL_PATH','./model.bin')
DV_PATH = os.getenv('DV_PATH','./dv.bin')
TRAINING_DATA_PATH = os.getenv('TRAINING_DATA_PATH','./data/amazon_laptop_prices_v01.csv')
training_data = pd.read_csv(TRAINING_DATA_PATH)    
training_data = clean.clean_data(training_data)
    
app = Flask(__name__)

with open(DV_PATH, 'rb') as dv_raw:
    dv = pickle.load(dv_raw)
    
with open(MODEL_PATH, 'rb') as model_raw:
    model = pickle.load(model_raw)

categorical_parameter_names = [
    "brand",
    "screen_size",
    "cpu",
    "OS",
    "cpu_mfr",
    "graphics_type",
    "graphics_mfr",
]
parameter_options = {}
for param in categorical_parameter_names:
    parameter_options[param] = training_data[param].unique().tolist()

for key in parameter_options:
    #filter out NaN from the list
    parameter_options[key] = list(filter(lambda x: x==x, parameter_options[key]))

@app.route("/")
def index():
    return send_file('./static/index.html')

@app.route("/parameters")
def parameters():
    return jsonify(results=parameter_options)

@app.route('/predict', methods=['POST'])
def predict():
    # get the data
    data = request.get_json(force=True)
    val = [
        {
            "brand": data.get('brand') or 'unknown',
            "screen_size": data.get('screen_size') or 'unknown',
            "cpu": data.get('cpu') or 'unknown',
            "OS": data.get('OS') or 'unknown',
            "cpu_mfr": data.get('cpu_mfr') or 'unknown',
            "graphics_type": data.get('graphics_type') or 'unknown',
            "graphics_mfr": data.get('graphics_mfr') or 'unknown',
            "harddisk_gb": data.get('harddisk_gb') or 0,
            "ram_gb": data.get('ram_gb') or 0
        }
    ]
    X_val = dv.transform(val)
    prediction = model.predict(X_val)
    price = round(float(prediction[0]),2)
    
    # find 10 roles with prices closest to the predicted price in the training data
    t_data = pd.concat([training_data, pd.DataFrame([{
        'price': price,
        **data
    }])], ignore_index=True)

    t_data['diff'] = abs(t_data['price'] - price)
    t_data = t_data.sort_values(by=['diff'])
    top10 = t_data.head(10).sort_values(by='price').to_dict(orient='records')
    #replace np.nan with None
    for item in top10:
        for key in item:
            if pd.isnull(item[key]):
                item[key] = None
    
    output = {
        'price': price,
        'top10': top10
    }
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)