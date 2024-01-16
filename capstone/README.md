# Capstone Project

Capstone Project 2	8 January 2024	22 January 2024	
Capstone Project 2 evaluation	22 January 2024	29 January 2024
https://docs.google.com/spreadsheets/d/e/2PACX-1vSkEwMv5OKwCdPfW6LgqQvKk48dZjPcFDrjDstBqZfq38UPadh0Nws1b57qOVYwzAjSufKnVf7umGWH/pubhtml

https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/projects

## Problem

Predicting the general age of a person based on their facial characteristics is a well-known, and still unsolved problem in machine learning. Predicting the age of a person can be useful for applications such as entertainment (providing age appropriate content), biometrics, or even just for providing a more customized user experience based on a target age.

## Dataset
https://www.kaggle.com/datasets/arashnic/faces-age-detection-dataset 

## Development Setup

- ensure you are in the capstone directory with `cd capstone`
- install poetry on your system `pip install poetry`
- `poetry shell` to create a new virtual env and activate it
- `poetry install` will install dev dependencies
- OPTIONAL: run `poetry export --without-hashes --format=requirements.txt > requirements.txt` to update requirements.txt dependencies if any changes
- notebook.ipynb contains prototype code used to clean , run EDA, and test various models and parameter combinations
- run `cd capstone` to set the current directory to the midterm folder
- run `python3 train.py` to create or update the model and dv
- run `python3 convert.py -i [INPUT_MODEL_FILE_NAME] -o model.tflite`
- run `python3 predict.py` to create a local Flask server

## Test

- ensure you are in the capstone directory with `cd capstone`
- `docker build . -f Dockerfile-lambda -t face-age-detection-capstone`
- `docker run -p 9696:9696 face-age-detection-capstone`
- from another terminal run `python3 test-lambda-locally.py`

Deployed to AWS at https://laptop-price-prediction.tzvi.dev/

```
 curl https://laptop-price-prediction.tzvi.dev/predict --data '{"brand":"dell","screen_size":"14 ","cpu":"i7","OS":"Windows 11 Home","cpu_mfr":"intel","graphics_type":"discrete","graphics_mfr":"nvidia","harddisk_gb":1000,"ram_gb":8}'
```

## Cloud deployment

### AWS Lambda using CDK

- install the aws CDK https://docs.aws.amazon.com/cdk/v2/guide/work-with.html#work-with-prerequisites
- in this directory and run `cdk bootstrap`
- run `cdk deploy` to deploy the LaptopPredictionStack
- somewhere in the middle of running it will send an email to the domain owner (tzvi.dev) asking permission to create a certificate for the subdomain which the owner must click "accept" on for the run to continue
- the stack will build the docker image, create a lambda, create an ssl certificate, and create a cloudfront deployment pointing to the lambda function url
- redirect Cloudflare (or other DNS provider) to the cloudfront distribution domain name XXXX.cloudfront.net using a CNAME record
- run `cdk destroy` to destroy the stack
