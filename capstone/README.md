# Capstone Project

Capstone Project 2	8 January 2024	22 January 2024	
Capstone Project 2 evaluation	22 January 2024	29 January 2024
https://docs.google.com/spreadsheets/d/e/2PACX-1vSkEwMv5OKwCdPfW6LgqQvKk48dZjPcFDrjDstBqZfq38UPadh0Nws1b57qOVYwzAjSufKnVf7umGWH/pubhtml

https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/projects

## Problem

Predicting the general age of a person based on their facial characteristics is a well-known, and still unsolved problem in machine learning. Predicting the age of a person can be useful for applications such as entertainment (providing age appropriate content), biometrics, or even just for providing a more customized user experience based on a target age.

Google recently deployed [age detection by scanning selfie images](https://www.telegraph.co.uk/business/2023/12/15/google-develops-selfie-scanning-block-children-porn/) as part of a solution to be compliant with upcoming UK laws that require services to block children from accessing adult websites.

## Dataset
https://www.kaggle.com/datasets/arashnic/faces-age-detection-dataset 

The dataset provided a total of 19906 images.The attributes of data are as follows:

ID – Unique ID of image
Class – Age bin of person in image

For simplicity, the problem has been converted to a multiclass problem with classes as Young, Middle and Old.

## Development Setup

- ensure you are in the capstone directory with `cd capstone`
- follow instructions in capstone/face-age-detection/README.md to download and unzip the training dataset
- install poetry on your system `pip install poetry`
- `poetry shell` to create a new virtual env and activate it
- `poetry install` will install dev dependencies
- OPTIONAL: run `poetry export --without-hashes --format=requirements.txt > requirements.txt` to update requirements.txt dependencies if any changes
- notebook.ipynb contains prototype code used to run EDA, and test various models and parameter combinations
- run `cd capstone` to set the current directory to the capstone folder
- run `python3 train.py` to create or update the model
- run `python3 convert.py -i [INPUT_MODEL_FILE_NAME] -o model.tflite` to convert the model to a tflite compatible model

## Test

- ensure you are in the capstone directory with `cd capstone`
- `docker build . --platform linux/amd64 -f Dockerfile-lambda -t face-age-detection-capstone`
- `docker run --platform linux/amd64 -p 8080:8080 face-age-detection-capstone`
- from another terminal run `python3 test-lambda-locally.py`
- the result should be similar to:
```json
{
    "results": {
        "prediction": "young",
        "raw": {
            "middle": 0.2964254915714264,
            "old": 0.026481281965970993,
            "young": 0.6770931482315063
        }
    }
}
```

Deployed to AWS at https://face-age-detection.tzvi.dev/

```shell
curl https://face-age-detection.tzvi.dev/predict -H "Content-Type: application/json" --data '{"url":"https://www.tzvi.dev/images/headshot_steve_friedman_circle_clear.png"}'
```

## Cloud deployment

### AWS Lambda using CDK

- install and set up the aws CDK https://docs.aws.amazon.com/cdk/v2/guide/work-with.html#work-with-prerequisites
- in this directory and run `cdk bootstrap`
- run `cdk deploy` to deploy the FaceAgeDetectionStack
- somewhere in the middle of running it will send an email to the domain owner (tzvi.dev) asking permission to create a certificate for the subdomain which the owner must click "accept" on for the run to continue
- the stack will build the docker image, create a lambda, create an ssl certificate, and create a cloudfront deployment pointing to the lambda function url
- redirect Cloudflare (or other DNS provider) to the cloudfront distribution domain name XXXX.cloudfront.net using a CNAME record
- run `cdk destroy` to destroy the stack
