from unittest import TestCase
from ml_ad_lib.base.anomaly_detection import create_ad_pipeline
import pandas as pd
import joblib


class TestSpotThresholding(TestCase):

    def test_create_prophet_pipeline(self):
        pipeline_steps = {"step": "SPOT", "hello": "value1"}
        pr = create_ad_pipeline(pipeline_steps)
        joblib.dump(pr, "spotpipeline.sav")
        adPipeline = joblib.load('spotpipeline.sav')
        print(adPipeline)

        data = {
            "value": [1, 3, 7]
        }
        df = pd.DataFrame(data)

        print(df.head())
        print(df.dtypes)

        adPipeline.fit(df)

        data = {
            "value": [-1.039,-2.657,5.657,6.657,-2.657,-0.829,-0.829,
                      -0.829,-0.829,-0.817,-0.817,-0.817,-0.817,-1.449,-1.449,-1.449,-1.449,-1.67,-1.67,-1.67,-1.67,-2.977,-2.977,-2.977,-2.977,-3.763,-3.763,-3.763,-3.763,-4.738,-4.738,-4.738,-4.738,-4.742,-4.742,-4.742,-4.742,-4.051,-4.051,-4.051,
                      -4.051,-2.529,-2.529,-2.529,-2.529,
                      -3.376,-3.376,-3.376,-3.376,-3.38,-3.38,3.38,3.38,-3.827,-3.827,-3.827,-3.827,4.019,-4.019,-4.019,-4.019,-4.407,-4.407,-4.407,-4.407,-3.578,-3.578,-3.578,-3.578,-2.807,-2.807,-2.807,-2.807,-3.959,-3.959,-3.959,-3.959,-3.385,-3.385,-3.385,-3.385,-3.501,-3.501,-3.501,-3.501,-3.077,-3.077,-3.077,-3.077,-2.804,-2.804,-2.804,-2.804,-2.076,-2.076,-2.076,-2.076,-3.222,-3.222,-3.222,-3.222,-3.218,-3.218,-3.218,-3.218,-1.015,-1.015,-1.015,-1.015,-1.562,-1.562,-1.562,-1.562,-2.429,-2.429,-2.429,-2.429,-3.22,-3.22,-3.22,-3.22,-3.048,-3.048,-3.048,-3.048,-3.529,-3.529,-3.529,-3.529]
        }
        testdf = pd.DataFrame(data)

        print(adPipeline.predict(testdf, n_init=10))
        print(adPipeline.predict(testdf))
