from unittest import TestCase
from ml_ad_lib.base.anomaly_detection import create_ad_pipeline
import pandas as pd
import joblib
from datetime import datetime, timedelta


class TestProphetForecast(TestCase):


    def datetime_range(self,start, end, delta):
        current = start
        while current < end:
            yield current
            current += delta


    def test_create_prophet_pipeline(self):
        pipeline_steps = {"step": "Prophet","hello":"value1"}
        pr = create_ad_pipeline(pipeline_steps)
        print(pr)


        joblib.dump(pr, "adpipeline.sav")

        adPipeline = joblib.load('adpipeline.sav')
        print(adPipeline)
        print(type(adPipeline))
        df=pd.read_csv('traindata.csv')
        df.columns=['ds','y']
        df['ds']=pd.to_datetime(df.ds)
        print(df.head())
        print(df.dtypes)

        adPipeline.fit(df)

        #future = list()
        future = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in
               self.datetime_range(datetime(2022, 1, 8, 0),
                              datetime(2022, 1, 14, 23,55),
                              timedelta(minutes=5))]
        print(future)

        # for i in range(1, 13):
        #     date = '2022-%01d' % i
        #     future.append([date])
        # print(future)
        future = pd.DataFrame(future)
        future.columns = ['ds']
        future['ds'] = pd.to_datetime(future['ds'])
        print(adPipeline.predict(future,test='test'))

