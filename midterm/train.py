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
import clean
SEED = 65
n_splits = 5

def prepare(df):
    df = df.reset_index(drop=True)
    y = df['price'].values
    df = df.drop('price', axis=1)

    categorical_columns = list(df.dtypes[ df.dtypes == (object)].index)
    numeric_columns = list(df.dtypes[ df.dtypes == (float)].index)

    for col in categorical_columns:
        df[col].fillna('unknown',inplace=True)
    for col in numeric_columns:
        df[col].fillna(0,inplace=True)
    
        
    df.reset_index(drop=True, inplace=True)
    return df, y

def remove_columns(df):
        
    #remove the color column since I don't feel like it contributes heavily to price
    df = df.drop('color', axis=1)
    #remove the model column since it is too distinct to generalize over 
    df = df.drop('model', axis=1)
    
    df = df.drop('cpu_speed', axis=1)
    df = df.drop('rating', axis=1)

    # get all the columns that start with sf_ and remove them 
    sf_columns = [col for col in df.columns if col.startswith('sf_')]
    df = df.drop(sf_columns, axis=1)
    
    # data = data.drop('harddisk_gb', axis=1)
    # data = data.drop('graphics_mfr', axis=1)
    # data = data.drop('graphics_type', axis=1)
    # data = data.drop('brand', axis=1)
    # data = data.drop('OS', axis=1)
    return df

def train(df_train, C=1.0):
    dv = DictVectorizer()
    df_train, y_train = prepare(df_train)
    dicts = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(dicts)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, dv

def predict(df_test, dv, model):
    df_test, y_test = prepare(df_test)
    dicts = df_test.to_dict(orient='records')
    X_val = dv.transform(dicts)
    
    y_pred = model.predict(X_val)
    
    return y_pred, y_test

# def train(df_train):
#     dv = DictVectorizer(sparse=True)
#     df_train, y_train = prepare(df_train)
#     df_train = remove_columns(df_train)
#     X_full_train = dv.fit_transform(df_train.to_dict(orient='records'))
#     features = dv.get_feature_names_out().tolist()
#     best_params = {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_depth': 7, 'min_child_weight': 0.5, 'subsample': 0.8}
#     xgb_params = {
#         'eta': 0.1, 
#         'objective': 'reg:squarederror',
#         'nthread': 8,
#         'verbosity': 1,
#         **best_params
#     }
#     dfulltrain = xgb.DMatrix(X_full_train, label=y_train, feature_names=features)
#     model = xgb.train(xgb_params, dfulltrain, num_boost_round=70)
#     return model, dv

# def predict(df_test, dv, model):
#     features = dv.get_feature_names_out().tolist()
#     df_test, y_test = prepare(df_test)
#     X_test = dv.transform(df_test.to_dict(orient='records'))
#     dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)
#     y_pred = model.predict(dtest)
#     return y_pred, y_test


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
