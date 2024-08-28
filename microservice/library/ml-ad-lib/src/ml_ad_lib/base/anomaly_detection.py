from sklearn.pipeline import make_pipeline
from ml_ad_lib.base.pipeline_step_util import get_estimator


def create_ad_pipeline(pipeline_steps):
    """Construct a :class:`Pipeline` from the given configuration.

    Parameters
    ----------
    pipeline_steps : Dictionary of step.
        step with estimator and parameters.


    Returns
    -------
    p : Pipeline
        Returns a scikit-learn :class:`Pipeline` object.

    See Also
    --------
    Pipeline : Class for creating a pipeline of transforms with a final
            estimator.

    Examples
    --------
        >>> from ml_ad_lib import create_ad_pipeline
        create_ad_pipeline('step': 'ZscoreEstimator','threshold':10)
        """
    if pipeline_steps:
        preprocessing_step_name = pipeline_steps['step']
        pipeline_steps.pop('step')
        estimator = get_estimator(preprocessing_step_name, **pipeline_steps)
        return make_pipeline(estimator)
