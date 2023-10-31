import numpy as np
import pandas as pd

def clean_screen_size(data):
    #convert any float values in the screen_size column to strings
    data['screen_size'] = data['screen_size'].astype(str)
    # remove the word inches from the screen_size column
    data['screen_size'] = data['screen_size'].str.replace('inches', '', case=False)
    #remove spaces from the screen_size column
    data['screen_size'] = data['screen_size'].str.replace(' ', '')
    data['screen_size'] = data['screen_size'].replace('nan', np.nan)
    # data['screen_size'] = data['screen_size'].astype(float).round().astype(str)
    # data = data.drop('screen_size', axis=1)
    return data

def clean_special_features(data):
    # replacements = {
    #     'anti-glare screen': 'anti glare',
    #     'anti glare coating': 'anti glare',
    #     'anti-glare': 'anti glare',
    #     'wifi & bluetooth': 'bluetooth',
    #     'backlit kb': 'backlit keyboard',
    #     'fingerprint': 'fingerprint reader',
    #     'high definition audio': 'hd audio',
    #     'nanoedge bezel': 'thin bezel',
    #     'support stylus': 'pen',
    #     'stylus': 'pen',
    # }

    # #for each value in special_features column, split it by comma and add it to the counter
    # special_features = Counter()
    # for row in data['special_features']:
    #     if isinstance(row, str):
    #         values = row.lower().split(',')
    #         #strip any leading or trailing spaces from each value
    #         values = [value.strip() for value in values]
    #         # for each replacement, replace the value with the key
    #         values = [replacements.get(value, value) for value in values]
    #         special_features.update(values)

    # # remove 'information not available' from the special_features counter
    # special_features.pop('information not available', None)
    # # select all the special_features that have a count greater than 15
    # special_features = {key: value for key, value in special_features.items() if value > 15}

    # top_feature_names = special_features.keys()
    # special_feature_column_names = [name.replace(' ', '_') for name in top_feature_names]
    # # for each special_feature, create a new column in the dataframe if the special_feature is present in the special_features column
    # for feature in special_feature_column_names:
    #     data[feature] = data['special_features'].str.contains(feature, case=False)
    #     data[feature] = data[feature].fillna(False)
    # for feature in special_feature_column_names:
    #     data[feature] = data[feature].astype(str)

    # drop the special_features column
    data = data.drop('special_features', axis=1)
    return data

def clean_price(data):
    # fix the price column by removing the dollar sign and commas
    data['price'] = data['price'].str.replace('$', '').str.replace(',', '').astype(float)
    #remove all rows where the price is nan
    data = data.dropna(subset=['price'])
    return data

def clean_ram(data):
    #replace nan values with ''
    data['ram'] = data['ram'].fillna('')
    # if any of the values in the ram column have MB in them, convert them to GB and place in the gb_ram column
    data.loc[data['ram'].str.contains('MB'), 'ram_gb'] = data.loc[data['ram'].str.contains('MB'), 'ram'].str.replace('MB', '').astype(float) / 1000
    # fix the ram column by removing the GB if the value contains GB and convert to float
    data.loc[data['ram'].str.contains('GB'), 'ram_gb'] = data.loc[data['ram'].str.contains('GB'), 'ram'].str.replace('GB', '').astype(float)
    data = data.drop('ram', axis=1)
    return data

def clean_graphics(data):
    data['graphics'] = data['graphics'].fillna('Unknown')
    data['graphics_coprocessor'] = data['graphics_coprocessor'].fillna('Unknown')
    data['graphics_type'] = 'integrated'
    # if the graphics column does not contain integrated, set the graphics_type column to discrete
    data.loc[~data['graphics'].str.contains('Integrated'), 'graphics_type'] = 'discrete'
    # if the graphics column contains Unknown, set the graphics_type column to nan
    data.loc[data['graphics'].str.contains('Unknown'), 'graphics_type'] = np.nan
    # set to integrated if the column contains Iris Xe
    data.loc[data['graphics'].str.contains('Iris Xe'), 'graphics_type'] = 'integrated'
    #set the graphics_type column to integrated if graphics_coprocessor contains Intel
    data.loc[data['graphics_coprocessor'].str.contains('Intel'), 'graphics_type'] = 'integrated'
    #set the graphics_type column to discrete if graphics_coprocessor contains discrete
    data.loc[data['graphics_coprocessor'].str.lower().str.contains('discrete'), 'graphics_type'] = 'discrete'

    # set the graphics_mfr column to the first word in the graphics column if the graphics_type is discrete
    data.loc[data['graphics_type'] == 'discrete', 'graphics_mfr'] = data.loc[data['graphics_type'] == 'discrete', 'graphics'].str.split(' ').str[0].str.lower()
    data['graphics_mfr'].fillna('unknown', inplace=True)
    data.loc[data['graphics_mfr'].str.contains('dedicated|shared'), 'graphics_mfr'] = 'unknown'
    data.loc[data['graphics_mfr'].str.contains('intel|iris|uhd|gt2'), 'graphics_mfr'] = 'intel'
    data.loc[data['graphics_mfr'].str.contains('rtx|geforce|t550|t500,hd|t1200|nvidia®|quadro|qn20-m1-r'), 'graphics_mfr'] = 'nvidia'
    data.loc[data['graphics_mfr'].str.contains('adreno'), 'graphics_mfr'] = 'qualcomm'
    data.loc[data['graphics_mfr'] == 'unknown', 'graphics_mfr'] = np.nan

    data = data.drop('graphics', axis=1)
    data = data.drop('graphics_coprocessor', axis=1)
    # data = data.drop('graphics_mfr', axis=1)
    # data = data.drop('graphics_type', axis=1)
    return data

def clean_brand(data):
    data['brand'].fillna('unknown', inplace=True)
    # set all values of ROKC in the brand column to nan
    data['brand'] = data['brand'].replace('ROKC', 'unknown')
    data.loc[data['brand'].str.contains('latitude'), 'brand'] = 'dell'
    data.loc[data['brand'].str.contains('toughbook'), 'brand'] = 'panasonic'

    data['brand'] = data['brand'].str.lower()

    #select all brands with only 1 or 2 values
    brands = data['brand'].value_counts()
    brands = brands[brands < 3].index.tolist()
    #set all brands with only 1 or 2 values to nan
    data.loc[data['brand'].isin(brands), 'brand'] = 'unknown'

    data.loc[data['brand'] == 'unknown', 'brand'] = np.nan
    # data = data.drop('brand', axis=1)
    return data

def clean_os(data):
    #set nan in OS to 'Unknown'
    data['OS'] = data['OS'].fillna('Unknown')
    data.loc[data['OS'].str.contains('10') & ~data['OS'].str.contains('pro', case=False), 'OS'] = 'Windows 10 Home'
    data.loc[data['OS'].str.contains('10') & data['OS'].str.contains('pro', case=False), 'OS'] = 'Windows 10 Pro'
    data.loc[data['OS'].str.contains('11') & ~data['OS'].str.contains('pro', case=False), 'OS'] = 'Windows 11 Home'
    data.loc[data['OS'].str.contains('11') & data['OS'].str.contains('pro', case=False), 'OS'] = 'Windows 11 Pro'
    # set OS to Windows if it contains Windows but not 10 or 11
    data.loc[data['OS'].str.lower().str.contains('windows') & ~data['OS'].str.contains('10|11'), 'OS'] = 'Windows(Other)'
    # set OS to other if it contains anything other than Windows or Chrome, Mac, or Linux
    data.loc[~data['OS'].str.lower().str.contains('windows|chrome|mac|linux'), 'OS'] = 'Other'
    data.loc[data['OS'].str.lower().str.contains('mac'), 'OS'] = 'Mac OS'
    # data = data.drop('OS', axis=1)
    return data

def clean_harddisk(data):
    data['harddisk'].fillna('unknown', inplace=True)
    #convert the harddisk_gb column to float
    data['harddisk_gb'] = pd.to_numeric(data['harddisk'], errors='coerce')
    data.loc[data['harddisk'].str.contains('MB'), 'harddisk_gb'] = data.loc[data['harddisk'].str.contains('MB'), 'harddisk'].str.replace('MB', '').astype(float) / 1000
    data.loc[data['harddisk'].str.contains('TB'), 'harddisk_gb'] = data.loc[data['harddisk'].str.contains('TB'), 'harddisk'].str.replace('TB', '').astype(float) * 1000
    data.loc[data['harddisk'].str.contains('GB'), 'harddisk_gb'] = data.loc[data['harddisk'].str.contains('GB'), 'harddisk'].str.replace('GB', '').astype(float)
    # set the harddisk_gb column to nan if the harddisk column contains unknown
    data.loc[data['harddisk'].str.contains('unknown'), 'harddisk_gb'] = np.nan
    data = data.drop('harddisk', axis=1)
    # data = data.drop('harddisk_gb', axis=1)
    return data
    
def clean_cpu(data):
    data['cpu'].fillna('unknown', inplace=True)
    data['cpu'] = data['cpu'].str.lower().str.replace('_', ' ')
    data.loc[data['cpu'].str.contains('athlon'), 'cpu'] = 'athalon' 
    mfr_model = {
        'intel': ['i3', 'i5', 'i7', 'i9', 'pentium', 'celeron', 'atom', 'xeon', 'core 2 duo', 'core m', 'core 2', 'core duo'],
        'amd': ['ryzen', 'athlon', 'sempron', 'opteron', 'amd r', 'amd a', 'a4', 'a6', 'a8', 'a9', 'a10', 'a12', 'a13', 'e2' , 'a-series'],
        'apple': ['m1 max', 'm1','m2 max', 'm2' ],
        'qualcomm': ['snapdragon'],
        'samsung': ['exynos'],
        'mediatek': ['helio', 'mt8183', 'mt8127'],
        'rockchip': ['rk'],
        'huawei': ['kirin'],
        'motorola': ['68000'],
    }
    # for each key in the mfr_model dictionary, if the cpu column contains the key, set the cpu_mfr column to the key
    for mfr, models in mfr_model.items():
        for model in models:
            data.loc[data['cpu'].str.contains(model), 'cpu_mfr'] = mfr
            data.loc[data['cpu'].str.contains(model), 'cpu'] = model
    data.loc[data['cpu'].str.contains('unknown|others'), 'cpu'] = np.nan

    # data['cpu_speed'] = data['cpu_speed'].str.replace('GHz', '')
    # data['cpu_speed'] = data['cpu_speed'].str.replace('Hz', '')
    # data['cpu_speed'] = data['cpu_speed'].str.replace('MHz', '')
    # data['cpu_speed'] = pd.to_numeric(data['cpu_speed'], errors='coerce')
    # # assume that any speeds above 100 are in MHz and divide by 1000 
    # data.loc[data['cpu_speed'] > 1000, 'cpu_speed'] = data.loc[data['cpu_speed'] > 1000, 'cpu_speed'] / 1000
    data = data.drop('cpu_speed', axis=1)

    return data

def clean_rating(data):
    # round ratings and convert them to a categorical value (string)
    # data['rating'] = data['rating'].round().astype(str)
    # data.loc[data['rating'].str.contains('nan'), 'rating'] = np.nan
    data = data.drop('rating', axis=1)
    return data

def clean_data(data):
    data = clean_brand(data)
    data = clean_cpu(data)
    data = clean_graphics(data)
    data = clean_harddisk(data)
    data = clean_os(data)
    data = clean_price(data)
    data = clean_ram(data)
    data = clean_rating(data)
    data = clean_screen_size(data)
    data = clean_special_features(data)
    #remove the color column since I don't feel like it contributes heavily to price
    data = data.drop('color', axis=1)
    #remove the model column since it is too distinct to generalize over 
    data = data.drop('model', axis=1)
    #remove duplicate rows
    # data = data.drop_duplicates()
    return data