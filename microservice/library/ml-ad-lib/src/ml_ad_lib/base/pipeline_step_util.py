from ml_ad_lib.custom_estimator.prophet_estimator import ProphetEstimator
from ml_ad_lib.custom_estimator.zscore_threshold_estimator import \
    ZscoreEstimator
from ml_ad_lib.custom_estimator.spot_estimator import SpotEstimator

string_to_estimator_dictionary = {
    "Prophet": ProphetEstimator,
    "ZscoreEstimator": ZscoreEstimator,
    "SPOT": SpotEstimator
}


def get_estimator(step_name, **kwargs):
    """Construct a :Estimator from the given configuration.

    Parameters
    ----------
    step_name : Name of step.
        Name that identifies the estimator

    **kwargs : dict
            Parameters of this estimator.

    Returns
    -------
    p : Estimator
        Returns one of the supported estimator based on input step name.

    Examples
    --------
        >>> from ml_ad_lib import get_estimator
        get_estimator(preprocessing_step_name, **kwargs)
    """
    estimator = string_to_estimator_dictionary.get(step_name)(**kwargs)
    return estimator
