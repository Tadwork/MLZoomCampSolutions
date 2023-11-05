import pandas as pd

from training_data_helpers import find_index_of_closest_price, get_categorical_parameters


def test_find_closest_price():
    prices = [
        {'price': 1000},
        {'price': 2000},
        {'price': 2200},
        {'price': 3000},
    ]
    # find 1000 which is the closest price between 1000 and 2000 for 1200
    assert find_index_of_closest_price(prices, 1200) == 0
    # find 2000 which is the closest price between 1000 and 2000 for 1800
    assert find_index_of_closest_price(prices, 1800) == 1
    # find 2000 which is the closest price between 2000 and 3000 for 2200 (choose lower price if equal distance)
    assert find_index_of_closest_price(prices, 2500) == 2
    # don't fail if the price is higher than the highest price
    assert find_index_of_closest_price(prices, 3500) == 3
    # don't fail if the price is lower than the lowest price
    assert find_index_of_closest_price(prices, 0) == 0
    # if the price is exactly the same as one of the prices, return that index
    assert find_index_of_closest_price(prices, 2000) == 1
    
def test_get_categorical_parameters():
    data = [
        {
            "brand": "Dell",
            "screen_size": '15.6',
            "cpu": "i7",
            "OS": "Windows 10",
            "cpu_mfr": "Intel",
            "graphics_type": "integrated",
            "graphics_mfr": "intel",
        },
        {
            'brand': 'HP',
            'screen_size': '17',
            'cpu': 'i5',
            'OS': 'Windows 10',
            'cpu_mfr': 'Intel',
            'graphics_type': 'discrete',
            'graphics_mfr': 'intel',
        },
        {
            'brand': 'Acer',
            'screen_size': '17',
            'cpu': 'i5',
            'OS': 'Windows 10',
            'cpu_mfr': 'Intel',
            'graphics_type': 'discrete',
            'graphics_mfr': 'intel',
        }
    ]
    options = get_categorical_parameters(pd.DataFrame(data))
    assert options == {
        'brand': ['Dell', 'HP', 'Acer'],
        'screen_size': ['15.6', '17'],
        'cpu': ['i7', 'i5'],
        'OS': ['Windows 10'],
        'cpu_mfr': ['Intel'],
        'graphics_type': ['integrated', 'discrete'],
        'graphics_mfr': ['intel']
    }
