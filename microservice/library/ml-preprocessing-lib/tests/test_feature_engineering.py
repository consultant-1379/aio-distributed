from unittest import TestCase
from ml_preprocessing_lib.base.feature_engineering import create_preprocessing_pipeline
import pandas as pd
import joblib

class TestFeatureEngineering(TestCase):

    def test_create_preprocessing_pipeline_with_fit(self):
        # pipeline_steps = [{"step": "StandardScaler", "column": "test",
        #                    "copy": False, "with_mean": False}
        #                   ]


        pipeline_steps = [{"step": "StandardScaler", "column": "test",
                           "copy": True, "with_mean": True
                           },
                          {"step": "StandardScaler", "column": "value",
                           "copy": True, "with_mean": True
                           }
                          ]

        pr = create_preprocessing_pipeline(pipeline_steps)
        print(pr)
        print('id', id(pr))
        joblib.dump(pr, "preprocessingpipeline.sav")

        preproesingpipeline = joblib.load('preprocessingpipeline.sav')
        print(preproesingpipeline)
        print(type(preproesingpipeline))
        data = {
            "id": [1, 2, 3],
            "test": [5, 7, 9],
            "value": [1, 3, 7]
        }
        df = pd.DataFrame(data)
        print(df)
        print(df.dtypes)
        print(df.columns)
        preproesingpipeline.fit(df)
        print(type(preproesingpipeline.transform(df)))
        print(preproesingpipeline.transform(df))
        self.assertIsNotNone(preproesingpipeline)
        #
        # from sklearn.preprocessing import StandardScaler
        # data = [[5], [7], [9]]
        # scaler = StandardScaler()
        # print(scaler.fit(data))
        # print(scaler.mean_)
        # print(scaler.transform(data))

        #

        # data = [[1], [3], [7]]
        # scaler = StandardScaler()
        # print(scaler.fit(data))
        # print(scaler.mean_)
        # print(scaler.transform(data))

        # from sklearn.svm import SVC
        # from sklearn.preprocessing import StandardScaler
        # from sklearn.datasets import make_classification
        # from sklearn.pipeline import Pipeline
        # from sklearn.model_selection import train_test_split
        #
        # x, y = make_classification(random_state=0)
        # x_train, x_test, y_train, y_test = train_test_split(x, y,
        #                                                     random_state=0)
        # pipeline = Pipeline([('scaler', StandardScaler(with_std=False)),
        #                      ('svc',
        #                                                             SVC())])
        #
        # pipeline.fit(x_train, y_train)
        # print(pipeline.score(x_test, y_test))

