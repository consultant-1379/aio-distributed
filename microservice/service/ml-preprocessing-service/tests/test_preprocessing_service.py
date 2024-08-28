import requests
import json

if __name__ == '__main__':
    headers = {
        'Content-Type': 'application/json',
    }
    with open('testrequest.json') as json_file:
        payload = json.load(json_file)
    url = "http://127.0.0.1:8000/preprocessing/createpipeline"
    response = requests.request("POST", url, headers=headers, json=payload,
                                verify=False)
    print(response.status_code)
    if response.status_code == 200:
        with open("preprocessingpipeline.sav", "wb") as f:
            f.write(response.content)

    print(response)
