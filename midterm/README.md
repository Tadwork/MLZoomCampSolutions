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
- run `cd midterm` to set the current directory to the midterm folder
- run `train.py` to create or update the model and dv
- run `predict.py` to create a local Flask server

## Final Model

The final model uses Linear Regression and has an RMSE of 402 on the test data set and an R2 score of 0.78
Validation numbers do vary but the model is pretty stable with the r2 score +- 0.032 on multiple folds (run train.py to see your own results)

## Test
- `cd midterm`
- `docker build . -t midterm`
- `docker run -p 9696:9696 midterm`
- from another terminal
```
 curl http://0.0.0.0:9696/predict --data '{"brand":"dell","screen_size":"14 ","cpu":"i7","OS":"Windows 11 Home","cpu_mfr":"intel","graphics_type":"discrete","graphics_mfr":"nvidia","harddisk_gb":1000,"ram_gb":8}'
```

Deployed to AWS at https://laptop-price-prediction.tzvi.dev/

```
 curl https://laptop-price-prediction.tzvi.dev/predict --data '{"brand":"dell","screen_size":"14 ","cpu":"i7","OS":"Windows 11 Home","cpu_mfr":"intel","graphics_type":"discrete","graphics_mfr":"nvidia","harddisk_gb":1000,"ram_gb":8}'
```


## Cloud deployment

### AWS Co-pilot
- [Install AWS Copilot](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Copilot.html) 
- `copilot init --app midterm --name laptop-price-prediction --type 'Request-Driven Web Service' --dockerfile './Dockerfile' --port 9696 --deploy`
- If making changes `copilot deploy`
- When done `copilot app delete`
- refer to the [AWS Copilot Documentation](https://aws.github.io/copilot-cli/)

### AWS Lambda using CDK

- install the aws CDK https://docs.aws.amazon.com/cdk/v2/guide/work-with.html#work-with-prerequisites
- in this directory and run `cdk bootstrap`
- run `cdk deploy` to deploy the LaptopPredictionStack
- somewhere in the middle of running it will send an email to the domain owner (tzvi.dev) asking permission to create a certificate for the subdomain which the owner must click "accept" on for the run to continue
- the stack will build the docker image, create a lambda, create an ssl certificate, and create a cloudfront deployment pointing to the lambda function url
- redirect Cloudflare (or other DNS provider) to the cloudfront distribution domain name XXXX.cloudfront.net using a CNAME record
- run `cdk destroy` to destroy the stack

