import pickle
import os

from flask import Flask, jsonify, request

MODEL_PATH = os.getenv('MODEL_PATH','./model1.bin')
DV_PATH = os.getenv('DV_PATH','./dv.bin')
app = Flask(__name__)

with open(DV_PATH, 'rb') as dv_raw:
    dv = pickle.load(dv_raw)
    
with open(MODEL_PATH, 'rb') as model_raw:
    model = pickle.load(model_raw)
    
@app.route('/predict', methods=['POST'])
def predict():
    # get the data
    data = request.get_json(force=True)
    print(data)
    X_val = dv.transform([data])
    
    prediction = model.predict_proba(X_val)
    print(prediction)
    y_pred = prediction[:, 1]
    # send back to browser
    output = {'results': float(y_pred[0])}
    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)