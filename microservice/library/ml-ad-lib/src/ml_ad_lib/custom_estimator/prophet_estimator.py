from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from prophet import Prophet


class ProphetEstimator(BaseEstimator):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._param_names = list(kwargs.keys())
        self.model = Prophet()

    def fit(self, X, y, **fit_params):
        """Fit the model.

        Fit the transformed data using the
        estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of
            first
            step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements
            for
            all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each
            step, where
            each parameter name is prefixed such that parameter
            ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : object
            Pipeline with fitted steps.
        """
        self.model.fit(X)
        self.X__=X
        self.y_=y
        return self

    def predict(self, X,**predict_params):
        """Apply `predict` with the estimator.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of
            first step
            of the pipeline.

        **predict_params : dict of string -> object
            Parameters to the ``predict``

        Returns
        -------
        y_pred : ndarray
            Result of calling `predict` on the estimator.
        """
        print(self.get_params())
        print(predict_params)
        check_is_fitted(self)
        #X = check_array(X)
        y_pred = self.model.predict(X)
        return y_pred
