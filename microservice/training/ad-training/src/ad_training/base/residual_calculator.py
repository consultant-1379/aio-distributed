from ad_training.base.client_util import push_file, \
    read_file
import sys
import pandas as pd
from io import StringIO


def calculate_residual():
    print('inside residual calculator')
    actual_values_file = read_file('data',
                             'training/feature-engineered/transformed-features.csv')
    actual_df = pd.read_csv(actual_values_file)
    actual_df.columns = ['ds', 'y']
    actual_df['ds'] = pd.to_datetime(actual_df.ds)
    print('actual values \n', actual_df.head(),'\n')

    forecasted_values_file = read_file('data',
                             'training/forecasted/currentweek/forecasted-values.csv')
    forecasted_df = pd.read_csv(forecasted_values_file)
    forecasted_df.columns = ['ds', 'yhat']
    forecasted_df['ds'] = pd.to_datetime(forecasted_df.ds)
    print('forecasted values \n', forecasted_df.head(),'\n')

    residual_series = actual_df['y'] - forecasted_df['yhat']
    frame = {'residual': residual_series}
    residual_df = pd.DataFrame(frame)
    residual_df['ds'] = actual_df['ds']
    residual_df = residual_df[['ds', 'residual']]
    print('residual values \n',residual_df,'\n')
    csv_buf = StringIO()
    residual_df.to_csv(csv_buf, index=False)
    push_file(csv_buf.getvalue(), 'data',
              'training/residual/residual-values.csv')


if __name__ == "__main__":
    sys.exit(calculate_residual())
