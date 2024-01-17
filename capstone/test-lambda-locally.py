import requests

data  = {
  "version": "2.0",
  "routeKey": "$default",
  "rawPath": "/predict",
  "headers": {},
  "rawQueryString": "",
  "requestContext": {
    "http": {
      "method": "POST",
      "path": "/predict",
      "protocol": "HTTP/1.1",
      "sourceIp": "192.168.0.1/32",
      "userAgent": "agent"
    },
  },
  "body": "{\"url\":\"https://www.tzvi.dev/images/headshot_steve_friedman_circle_clear.png\"}",
  "isBase64Encoded": False
}

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

result = requests.post(url, json=data).json()
print(result)