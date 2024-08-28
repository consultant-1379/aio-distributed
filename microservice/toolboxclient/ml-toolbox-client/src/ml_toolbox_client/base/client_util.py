import requests
import json
import boto3


def post_request(url, request):
    payload = json.loads(request)
    print('payload ',payload)
    headers = {
        'Content-Type': 'application/json',
    }
    response = requests.request("POST", url, headers=headers, json=payload,
                                verify=False)
    return response

def push_file(file,bucket_name,file_path):
    try:
        client = boto3.client(
            's3',
            endpoint_url="http://<host>:<port>",
            aws_access_key_id="<accesskey>",
            aws_secret_access_key="<password>",
        )
        print(client.list_buckets())
        client.put_object(
            Bucket=bucket_name,
            Key=file_path,
            Body = file
        )
    except Exception as e:
        print("Error", e)
