import requests
import json

url = 'http://127.0.0.1:5000/predict'
with open('plant_dataset/test/Tomato/20190919_171609.jpg','rb') as f:
    files = {'image': ('20190919_171609.jpg', f, 'image/jpeg')}
    resp = requests.post(url, files=files, data={'organ':'leaf'})

print(resp.status_code)
try:
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
except Exception:
    print(resp.text)
