from ad_training_featureeng.base.client_util import push_file,download_file,\
    read_file,get_file
import sys
import joblib
import pandas as pd
import os
from io import StringIO
import tempfile

def train_fe_pipeline():
    print('inside fe training')

    download_file('artifacts',
               'trainable-pipeline/featureengineering/featureeng'
               '-trainer.sav',"fe_training.sav")
    pipeline_obj=joblib.load("fe_training.sav")
    file_content=read_file('data',
               'training/input/train.csv')
    df = pd.read_csv(file_content)
    df.columns = ['ds', 'kpivalue']
    #df['ds'] = pd.to_datetime(df.ds)
    trained_pipeline = pipeline_obj.fit(df)
    transformed_array=trained_pipeline.transform(df)
    transformed_df = pd.DataFrame(transformed_array, columns=['kpivalue', 'ds'])
    transformed_df = transformed_df [['ds','kpivalue']]
    csv_buf = StringIO()
    transformed_df.to_csv(csv_buf,index=False)
    push_file(csv_buf.getvalue(),'data',
              'training/feature-engineered/transformed-features.csv')
    with tempfile.TemporaryFile() as fp:
        joblib.dump(trained_pipeline, fp)
        fp.seek(0)
        # use bucket_name and OutputFile - s3 location path in string format.
        push_file(fp.read(),'artifacts',
                  'trained-pipeline/featureengineering/fe_trained.sav')

    os.remove("fe_training.sav")

if __name__ == "__main__":
    sys.exit(train_fe_pipeline())