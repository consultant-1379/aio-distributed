from sklearn.pipeline import make_pipeline
from ml_preprocessing_lib.base.pipeline_step_util import get_transformer, \
    get_column_transformer


def create_preprocessing_pipeline(pipeline_steps):
    """Construct a :class:`Pipeline` from the given configuration.

       Parameters
       ----------
       pipeline_steps : List of steps.
           step with transformers and parameters.


       Returns
       -------
       p : Pipeline
           Returns a scikit-learn :class:`Pipeline` object.

       See Also
       --------
       Pipeline : Class for creating a pipeline of transforms with list of
       transformers.

       Examples
       --------
           >>> from ml_preprocessing_lib import \
    create_preprocessing_pipeline
           create_preprocessing_pipeline([{"step": "StandardScaler",
           "column": "test",
                           "copy": True, "with_mean": True
                           }
                          ])
           """
    transformers = []
    if pipeline_steps:
        for step_params in pipeline_steps:
            preprocessing_step_name = step_params['step']
            feature_column = step_params['column']
            step_params.pop('step')
            step_params.pop('column')
            transformers.append(
                get_transformer(preprocessing_step_name,
                                feature_column,
                                **step_params))
        column_transformer = get_column_transformer(transformers)
    return make_pipeline(column_transformer)
