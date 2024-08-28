from unittest import TestCase
from ml_ad_lib.base.anomaly_detection import create_ad_pipeline
import pandas as pd
import joblib


class TestZscoreThresholding(TestCase):


    def test_create_prophet_pipeline(self):
        pipeline_steps = {"step": "ZscoreEstimator","hello":"value1"}
        pr = create_ad_pipeline(pipeline_steps)
        joblib.dump(pr, "zscorepipeline.sav")
        adPipeline = joblib.load('zscorepipeline.sav')
        print(adPipeline)


        data = {
            "value": [1, 3, 7]
        }
        df = pd.DataFrame(data)

        print(df.head())
        print(df.dtypes)

        adPipeline.fit(df)

        data = {
            "value": [2, 25]
        }
        testdf = pd.DataFrame(data)

        print(adPipeline.predict(testdf,threshold=9))
        print(adPipeline.predict(testdf))

