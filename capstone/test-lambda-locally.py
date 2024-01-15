import requests

data  = {
  "version": "2.0",
  "routeKey": "$default",
  "rawPath": "/parameters",
  "requestContext": {
    "http": {
      "method": "GET",
      "path": "/parameters",
      "protocol": "HTTP/1.1",
    },
  },
}

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

result = requests.post(url, json=data).json()
print(result)