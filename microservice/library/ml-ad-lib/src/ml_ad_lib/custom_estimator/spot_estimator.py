from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from ml_ad_lib.algorithm.spot import SPOT

class SpotEstimator(BaseEstimator):
    """
        This class is a custom estimator implementation of SPOT algorithm
        univariate dataset

        Attributes
        ----------
        **kwargs : dict
                Parameters of this estimator.
        """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._param_names = list(kwargs.keys())

    def fit(self, X, y, **fit_params):
        """Fit the model.

        Fit the transformed data using the
        estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first
            step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for
            all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter
            ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : object
            Pipeline with fitted steps.
        """
        self.model=SPOT(1e-4)
        self.X__ = X
        self.y_ = y
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
        n_init=predict_params.get('n_init',2)
        print(n_init)
        print(predict_params)
        X = check_array(X)
        X = check_array(X)
        init_data = X[:n_init]  # initial batch
        data = X[n_init:]
        y_pred =[]
        self.model.fit(init_data, data)  # data import
        self.model.initialize(level = 0.50)  # initialization step
        y_pred = self.model.run()
        return y_pred
