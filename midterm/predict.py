import pickle
import os

import pandas as pd
from flask import Flask, jsonify, request, send_file
import xgboost as xgb

import clean 

MODEL_PATH = os.getenv('MODEL_PATH','./model.bin')
DV_PATH = os.getenv('DV_PATH','./dv.bin')
TRAINING_DATA_PATH = os.getenv('TRAINING_DATA_PATH','./data/amazon_laptop_prices_v01.csv')
training_data = pd.read_csv(TRAINING_DATA_PATH)    
training_data = clean.clean_brand(training_data)
training_data = clean.clean_cpu(training_data)
training_data = clean.clean_graphics(training_data)
training_data = clean.clean_harddisk(training_data)
training_data = clean.clean_os(training_data)
training_data = clean.clean_price(training_data)
training_data = clean.clean_ram(training_data)
training_data = clean.clean_rating(training_data)
training_data = clean.clean_screen_size(training_data)
training_data = clean.clean_special_features(training_data)
    
app = Flask(__name__)

with open(DV_PATH, 'rb') as dv_raw:
    dv = pickle.load(dv_raw)
    
with open(MODEL_PATH, 'rb') as model_raw:
    model = pickle.load(model_raw)

@app.route("/")
def index():
  return send_file('./static/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # get the data
    data = request.get_json(force=True)
    X_val = dv.transform([data])
    features = dv.get_feature_names_out().tolist()
    dtest = xgb.DMatrix(X_val, feature_names=features)
    
    prediction = model.predict(dtest)
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