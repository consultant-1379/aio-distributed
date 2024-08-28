from ml_toolbox_client.base.client_util import post_request,push_file
import sys

thresholding_request = '{' \
                       '"step": "ZscoreEstimator",' \
                       '"params": [' \
                       '{"key": "threshold",' \
                       '"value": "4"}]}'


def create_threshold_trainer():
    print('inside ad thresholding client')
    url = "http://ad-service.adtoolbox.svc.cluster.local:8080/ad/createpipeline"
    response = post_request(url, thresholding_request)
    print(response.status_code)
    push_file(response.content, "artifacts",
              "trainable-pipeline/AD/AD-thresholding-trainer.sav")

    print(response.headers)

if __name__ == "__main__":
    sys.exit(create_threshold_trainer())