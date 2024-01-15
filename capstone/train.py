"""
Training the final model
Saving it to a file (e.g. pickle)
"""
import json

import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error,r2_score
SEED = 65
n_splits = 5

def train(df_train, C=1.0):
    # dv = DictVectorizer()
    # df_train, y_train = prepare(df_train)
    # dicts = df_train.to_dict(orient='records')
    # X_train = dv.fit_transform(dicts)
    
    # model = LinearRegression()
    # model.fit(X_train, y_train)
    
    return model, dv

def predict(df_test, dv, model):
    # df_test, y_test = prepare(df_test)
    # dicts = df_test.to_dict(orient='records')
    # X_val = dv.transform(dicts)
    
    # y_pred = model.predict(X_val)
    
    return y_pred, y_test

def validate(df_full_train):
    # validation
    print('-------------------')
    print(f'doing validation')
    kfold = KFold(n_splits=n_splits, shuffle=True)
    scores = []
    fold = 0
    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        model,dv = train(df_train)
        y_pred,y_val = predict(df_val, dv, model)

        r2 = r2_score(y_val, y_pred)
        scores.append(r2)

        print(f'r2 on fold {fold} is {r2}')
        fold = fold + 1
    print('validation results:')
    print('%.3f +- %.3f' % ( np.mean(scores), np.std(scores)))
    print('-------------------')
    
if __name__ == '__main__':
    data = pd.read_csv('./data/amazon_laptop_prices_v01.csv')
    data = clean.clean_data(data)
    df_train_full, df_test = train_test_split(data, test_size=0.2,random_state=SEED)
    model, dv = train(df_train_full)
    y_pred, y_test = predict(df_test, dv, model)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f'RMSE: {rmse:.3f}')
    print(f'R2: {r2:.2f}')
    validate(df_train_full)
    with open('model.bin', 'wb') as model_out:
        pickle.dump(model, model_out)
    with open('dv.bin', 'wb') as dv_out:
        pickle.dump(dv, dv_out)
    print('successfully saved model and dv')
    sample_input = data.to_dict(orient='records')[0]
    del sample_input['price']
    print(f'sample inference input {json.dumps(sample_input, indent=2)}')
