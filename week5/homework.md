2023.10.3
0c275a06c5190c5ce00af0acbb61c06374087949f643ef32d355ece12c4db043

0.92
147MB

``` shell
docker build . -t week5
docker run -p 9696:9696 week5
curl http://0.0.0.0:9696/predict --data '{"job": "retired", "duration": 445, "poutcome": "success"}'

# {"results":{"results":0.726936946355423}}
```