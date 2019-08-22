"""
This module supports to run PAL functions in a pipeline manner.
"""

#pylint: disable=invalid-name
#pylint: disable=eval-used
#pylint: disable=unused-variable
#pylint: disable=line-too-long
class Pipeline(object):
    """
    Pipeline construction to run transformers and estimators sequentially.

    Parameters
    ----------

    step : List
        List of (name, transform) tuples that are chained. The last object should be an estimator.

    Examples
    --------

    >>> pipeline([
        ('pca', PCA(connection_context, scaling=True, scores=True)),
        ('imputer', Imputer(connection_context, strategy='mean')),
        ('hgbt', HybridGradientBoostingClassifier(conn_context = connection_context, \
        n_estimators = 4, split_threshold=0, learning_rate=0.5, fold_num=5, \
        max_depth=6, cross_validation_range=cv_range))
        ])

    """
    def __init__(self, steps):
        self.steps = steps

    def fit_transform(self, df, param):
        """
        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------

        df : DataFrame
            HANA dataframe to be transformed in the pipeline.
        param : dict
            Parameters corresponding to the transform name.

        Returns
        -------

        df_ : DataFrame
            Transformed HANA dataframe

        Examples
        --------

        >>> my_pipeline = Pipeline([
                ('pca', PCA(connection_context, scaling=True, scores=True)),
                ('imputer', Imputer(connection_context, strategy='mean'))
                ])
        >>> param = {'pca': [('key', 'ID'), ('label', 'CLASS')], 'imputer': []}
        >>> my_pipeline.fit_transform(train_df, param)

        """
        df_ = df
        for step in self.steps:
            param_str = ''
            if step[0] in param.keys():
                for name, val in param[step[0]]:
                    param_str = param_str + "," + name + "=" + "'" + val + "'"
            obj = step[1]
            exec_str = "obj.fit_transform(df_" + param_str + " )"
            df_ = eval(exec_str)
        return df_

    def fit(self, df, param):
        """
        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        df : DataFrame
            HANA dataframe to be transformed in the pipeline.
        param : dict
            Parameters corresponding to the transform name.

        Returns
        -------
        df_ : DataFrame
            Transformed HANA dataframe

        Examples
        --------

        >>> my_pipeline = Pipeline([
            ('pca', PCA(connection_context, scaling=True, scores=True)),
            ('imputer', Imputer(connection_context, strategy='mean')),
            ('hgbt', HybridGradientBoostingClassifier(conn_context = connection_context, \
            n_estimators = 4, split_threshold=0, learning_rate=0.5, fold_num=5, \
            max_depth=6, cross_validation_range=cv_range))
            ])
        >>> param = {
                        'pca': [('key', 'ID'), ('label', 'CLASS')],
                        'imputer': [],
                        'hgbt': [('key', 'ID'), ('label', 'CLASS'), ('categorical_variable', ['CLASS'])]
                    }
        >>> hgbt_model = my_pipeline.fit(train_df, param)
        """
        df_ = df
        fit = self.steps[-1]
        transform = self.steps[:-1]
        for step in transform:
            param_str = ''
            if step[0] in param.keys():
                for name, val in param[step[0]]:
                    param_str = param_str + "," + name + "=" + "'" + val + "'"
            obj = step[1]
            exec_str = "obj.fit_transform(df_" + param_str + " )"
            df_ = eval(exec_str)

        param_str = ''
        if fit[0] in param.keys():
            for name, val in param[fit[0]]:
                if isinstance(val, str):
                    param_str = param_str + "," + name + "=" + "'" + val + "'"
                if isinstance(val, list):
                    list_str = ''
                    for v in val:
                        list_str = list_str + "," + "'" + v + "'"
                    param_str = param_str + "," + name + "=" + "[" + list_str[1:] + "]"
        obj = fit[1]
        exec_str = "obj.fit(df_" + param_str + " )"
        eval(exec_str)
        return obj
