from ml_toolbox_client.base.client_util import post_request,push_file
import sys

forecasting_request = '{' \
                      '"step": "Prophet",' \
                      '"params": [' \
                      '{"key": "testparam1",' \
                        '"value": "value1"}]}'

def create_forecasting_trainer():
    print('inside ad forecasting client')

    url = "http://ad-service.adtoolbox.svc.cluster.local:8080/ad/createpipeline"
    response = post_request(url,forecasting_request)
    print(response.status_code)
    push_file(response.content, "artifacts",
              "trainable-pipeline/forecasting/forecasting-trainer.sav")

    print(response.headers)

if __name__ == "__main__":
    sys.exit(create_forecasting_trainer())