import json
from urllib.request import Request, urlopen

url = "http://127.0.0.1:5000/predict"
payload = json.dumps({
    "resume": "Python SQL Pandas Power BI Excel dashboard data visualization"
}).encode()

request = Request(url, data=payload, headers={"Content-Type": "application/json"})
with urlopen(request, timeout=10) as response:
    print(response.status)
    print(json.loads(response.read().decode()))
