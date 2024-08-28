from ad_training.base.client_util import push_file, download_file, \
    read_file
import sys
import joblib
import pandas as pd
import os
from io import StringIO
import tempfile
from datetime import datetime, timedelta


def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta


def forecast_current_week(trained_pipeline):
    time_stamp_list = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in
                       datetime_range(datetime(2022, 1, 1, 0),
                                      datetime(2022, 1, 7, 23, 55),
                                      timedelta(minutes=5))]

    current_week_df = pd.DataFrame(time_stamp_list)
    current_week_df.columns = ['ds']
    current_week_df['ds'] = pd.to_datetime(current_week_df['ds'])
    predicted_current_week_df = trained_pipeline.predict(current_week_df)
    predicted_current_week_df=predicted_current_week_df[['ds', 'yhat']]
    print(predicted_current_week_df)
    csv_buf = StringIO()
    predicted_current_week_df.to_csv(csv_buf, index=False)
    push_file(csv_buf.getvalue(), 'data',
              'training/forecasted/currentweek/forecasted-values.csv')


def forecast_future_week(trained_pipeline):
    time_stamp_list = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in
                       datetime_range(datetime(2022, 1, 8, 0),
                                      datetime(2022, 1, 14, 23, 55),
                                      timedelta(minutes=5))]

    future_week_df = pd.DataFrame(time_stamp_list)
    future_week_df.columns = ['ds']
    future_week_df['ds'] = pd.to_datetime(future_week_df['ds'])
    predicted_future_week_df = trained_pipeline.predict(future_week_df)
    predicted_future_week_df = predicted_future_week_df[['ds', 'yhat']]
    print(predicted_future_week_df)
    csv_buf = StringIO()
    predicted_future_week_df.to_csv(csv_buf, index=False)
    push_file(csv_buf.getvalue(), 'data',
              'training/forecasted/futureweek/forecasted-values.csv')


def train_and_forecast():
    print('inside forecasting training')

    download_file('artifacts',
                  'trainable-pipeline/forecasting/forecasting'
                  '-trainer.sav', "forecasting_training.sav")
    pipeline_obj = joblib.load("forecasting_training.sav")
    file_content = read_file('data',
                             'training/feature-engineered/transformed-features.csv')
    df = pd.read_csv(file_content)
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df.ds)
    print('training samples ', df.head())
    trained_pipeline = pipeline_obj.fit(df)

    forecast_current_week(trained_pipeline)
    forecast_future_week(trained_pipeline)
    with tempfile.TemporaryFile() as fp:
        joblib.dump(trained_pipeline, fp)
        fp.seek(0)
        push_file(fp.read(), 'artifacts',
                  'trained-pipeline/forecast/forecast_trained.sav')

    os.remove("forecasting_training.sav")


if __name__ == "__main__":
    sys.exit(train_and_forecast())
