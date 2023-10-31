# Midterm Project

## Problem

There are thousands of laptops on the market with very similar specifications and prices. Figuring out which one is a good value for the money is difficult. My goal for this project would be to create a model that can return a price target based on specs passed to an ML model based on data scraped from Amazon in October 2023. 

## Dataset

https://www.kaggle.com/datasets/talhabarkaatahmad/laptop-prices-dataset-october-2023/data

I am using this data source because it is fairly recent, and comes from a reputable retailer (Amazon) so the prices should be in line with the market. It also contains the Amazon rating which might be helpful in determining price and overall quality.

This is not the highest quality dataset though and after some deeper analysis it seems to be missing the following data:
- Number of CPU cores (might be correlated with cpu but not always the case )
- Battery capacity
- Build Quality (might be correlated with rating)
- Keyboard type
- Touchscreen
- mass storage type (SSD/HDD) and throughput
- Screen resolution

There were also a significant amount of inconsistency in many of the columns and I had to make some assumptions to align the values and get a reasonable number of categorical values.


## Development Setup
- install poetry on your system `pip install poetry`
- `poetry shell` to create a new virtual env and activate it
- `poetry install` will install dev dependencies
- notebook.ipynb contains prototype code used to clean , run EDA, and test various models and parameter combinations
- run `train.py` to create or update the model and dv
- run `predict.py` to create a local Flask server


## Test

```
 curl http://0.0.0.0:9696/predict --data '{"brand":"dell","screen_size":"14 ","cpu":"i7","OS":"Windows 11 Home","cpu_mfr":"intel","graphics_type":"discrete","graphics_mfr":"nvidia","harddisk_gb":1000,"ram_gb":8}'
```

also deployed to https://midterm-laptop-price-prediction.onrender.com/

```
 curl https://midterm-laptop-price-prediction.onrender.com/predict --data '{"brand":"dell","screen_size":"14 ","cpu":"i7","OS":"Windows 11 Home","cpu_mfr":"intel","graphics_type":"discrete","graphics_mfr":"nvidia","harddisk_gb":1000,"ram_gb":8}'
```