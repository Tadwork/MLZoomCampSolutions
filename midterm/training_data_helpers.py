
CATEGORICAL_PARAMETER_NAMES = [
        "brand",
        "screen_size",
        "cpu",
        "OS",
        "cpu_mfr",
        "graphics_type",
        "graphics_mfr",
    ]

def get_categorical_parameters(training_data):
    options = {}
    for param in CATEGORICAL_PARAMETER_NAMES:
        options[param] = training_data[param].unique().tolist()
        options[param] = list(filter(lambda x: x == x, options[param]))
    return options


def find_index_of_closest_price(records, price):
    start = 0
    
    end = len(records) - 1
    while start < end:
        mid = start + (end - start) // 2
        mid_price = records[mid]["price"]
        if mid_price < price:
            start = mid
        else:
            end = mid
    if abs(price - records[start]["price"]) <= abs(records[end]["price"] - price):
        return start
    return end
