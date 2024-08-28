from ad_training.base.client_util import push_file, download_file, \
    read_file
import sys
import joblib
import pandas as pd
import os
import tempfile


def train_thresholding():
    print('inside thresholding training')

    download_file('artifacts',
                  'trainable-pipeline/AD/AD-thresholding-trainer.sav',
                  "AD-thresholding-trainer.sav")
    pipeline_obj = joblib.load("AD-thresholding-trainer.sav")
    file_content = read_file('data',
                             'training/residual/residual-values.csv')
    df = pd.read_csv(file_content)
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df.ds)
    print('residual samples ', df['y'])
    residual_array = df['y'].to_numpy().reshape(-1,1)
    trained_pipeline = pipeline_obj.fit(residual_array)
    with tempfile.TemporaryFile() as fp:
        joblib.dump(trained_pipeline, fp)
        fp.seek(0)
        push_file(fp.read(), 'artifacts',
                  'trained-pipeline/AD-thresholding/thresholding_trained.sav')

    os.remove("AD-thresholding-trainer.sav")


if __name__ == "__main__":
    sys.exit(train_thresholding())
