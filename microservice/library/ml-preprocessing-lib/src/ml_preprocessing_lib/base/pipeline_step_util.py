from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer

string_to_transformer_dictionary = {
    "StandardScaler": StandardScaler
}

def get_transformer(step_name, feature_column=None, **kwargs):
    """Construct a :Transformer from the given configuration.

        Parameters
        ----------
        step_name : Name of step.
            Name that identifies the transformer

        feature_column : feature column.
            Name of the feature on which transformation as to be applied

        **kwargs : dict
                Parameters of this transformer.

        Returns
        -------
        p : Transformer
            Returns one of the supported transformer based on input step name.

        Examples
        --------
            >>> from ml_preprocessing_lib import
            get_transformer
            get_transformer(preprocessing_step_name, **kwargs)
        """
    transformer = string_to_transformer_dictionary.get(step_name)(**kwargs)
    return (transformer, [feature_column])


def get_column_transformer(transformers):
    """Construct a :Column Transformer from the list of Transformers.

            Parameters
            ----------
            *transformers : Transformer.
                List of transformers

            Returns
            -------
            p : ColumnTransformer
                Returns a ColumnTransformer based on the input transformer
                list
                name.

            Examples
            --------
                >>> from ml_preprocessing_lib import
                get_column_transformer
                get_column_transformer(*transformers)
            """
    return make_column_transformer(*transformers, remainder='passthrough')
