from ml_toolbox_client.base.client_util import post_request,push_file
import sys

preprocessing_request = '{' \
                        '"steps": [' \
                            ' {"step": "StandardScaler",' \
                                '"column": "kpivalue"}]' \
                        '}'

def create_feature_eng_trainer():
    print('inside feature eng client')
    url = "http://fe-service.adtoolbox.svc.cluster.local:8080/preprocessing/createpipeline"
    response = post_request(url,preprocessing_request)
    print(response.status_code)
    push_file(response.content,"artifacts",
              "trainable-pipeline/featureengineering/featureeng-trainer"
              ".sav")
    print(response.headers)

if __name__ == "__main__":
    sys.exit(create_feature_eng_trainer())