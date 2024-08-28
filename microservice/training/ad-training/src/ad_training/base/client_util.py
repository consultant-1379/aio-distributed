import boto3
import io
from io import BytesIO

client = boto3.client(
    's3',
    endpoint_url="http://<host>:<port>",
    aws_access_key_id="<access>",
    aws_secret_access_key="<secret>",
)


def push_file(file, bucket_name, file_path):
    try:
        client.put_object(
            Bucket=bucket_name,
            Key=file_path,
            Body=file
        )
    except Exception as e:
        print("Error", e)


def download_file(bucket_name, file_path, local_file_path):
    try:
        client.download_file(bucket_name, file_path, local_file_path)
    except Exception as e:
        print("Error", e)


def get_file(bucket_name, file_path):
    with BytesIO() as f:
        client.download_fileobj(Bucket=bucket_name, Key=file_path,
                                Fileobj=f)
        file_content = f.seek(0)
    return file_content


def read_file(bucket_name, file_path):
    try:
        obj = client.get_object(Bucket=bucket_name, Key=file_path)
        file_content = io.BytesIO(obj['Body'].read())
        return file_content

    except Exception as e:
        print("Error", e)
