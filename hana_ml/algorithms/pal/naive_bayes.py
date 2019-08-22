"""
This module contains wrappers for PAL naive bayes classification.

The following classes are available:

    * :class:`NaiveBayes`
"""

#pylint: disable=relative-beyond-top-level,line-too-long
import logging
import uuid

from hdbcli import dbapi
from hana_ml.ml_exceptions import FitIncompleteError
from . import metrics
from .pal_base import (
    PALBase,
    ParameterTable,
    ListOfStrings,
)

logger = logging.getLogger(__name__) #pylint: disable=invalid-name


class NaiveBayes(PALBase):
    """
    A classification model based on Bayes' theorem.

    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA system.

    alpha : float, optional

        Laplace smoothing value. Set a positive value to enable Laplace smoothing
        for categorical variables and use that value as the smoothing parameter.
        Set value 0 to disable Laplace smoothing.

        Defaults to 0.

    discretization : {'no', 'supervised'}, optional

        Discretize continuous attributes. Case-insensitive.

          - 'no' or not provided: disable discretization.
          - 'supervised': use supervised discretization on all the continuous
            attributes.

        Defaults to 'no'.

    model_format : {'json', 'pmml'}, optional

        Controls whether to output the model in JSON format or PMML format.
        Case-insensitive.

          - 'json' or not provided: JSON format.
          - 'pmml': PMML format.

        Defaults to 'json'.

    categorical_variable : str or list of str, optional

        Specifies INTEGER columns that should be treated as categorical.
        Other INTEGER columns will be treated as continuous.

    thread_ratio : float, optional

        Controls the proportion of available threads to use.
        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads. Values between 0 and 1
        will use that percentage of available threads.Values outside this range
        tell PAL to heuristically determine the number of threads to use.

        Defaults to 0.

    Attributes
    ----------

    model_ : DataFrame

        Trained model content.

        .. note::
            The Laplace value (alpha) is only stored by JSON format models.
            If the PMML format is chosen, you may need to set the Laplace value (alpha)
            again in predict() and score().

    Examples
    --------

    Training data:

    >>> df1.collect()
      HomeOwner MaritalStatus  AnnualIncome DefaultedBorrower
    0       YES        Single         125.0                NO
    1        NO       Married         100.0                NO
    2        NO        Single          70.0                NO
    3       YES       Married         120.0                NO
    4        NO      Divorced          95.0               YES
    5        NO       Married          60.0                NO
    6       YES      Divorced         220.0                NO
    7        NO        Single          85.0               YES
    8        NO       Married          75.0                NO
    9        NO        Single          90.0               YES

    Training the model:

    >>> nb = NaiveBayes(cc, alpha=1.0, model_format='pmml')
    >>> nb.fit(df1)

    Prediction:

    >>> df2.collect()
       ID HomeOwner MaritalStatus  AnnualIncome
    0   0        NO       Married         120.0
    1   1       YES       Married         180.0
    2   2        NO        Single          90.0

    >>> nb.predict(df2, 'ID', alpha=1.0, verbose=True)
       ID CLASS  CONFIDENCE
    0   0    NO   -6.572353
    1   0   YES  -23.747252
    2   1    NO   -7.602221
    3   1   YES -169.133547
    4   2    NO   -7.133599
    5   2   YES   -4.648640
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes

    model_format_map = {'json': 0, 'pmml': 1}
    discretization_map = {'no':0, 'supervised': 1}
    def __init__(self,
                 conn_context,
                 alpha=None,
                 discretization=None,
                 model_format=None,
                 categorical_variable=None,
                 thread_ratio=None
                ):
        super(NaiveBayes, self).__init__(conn_context)
        self.alpha = self._arg('alpha', alpha, float)
        self.discretization = self._arg('discretization',
                                        discretization, self.discretization_map)
        self.model_format = self._arg('model_format',
                                      model_format, self.model_format_map)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        self.categorical_variable = self._arg('categorical_variable', categorical_variable, ListOfStrings)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)

    def fit(self, data, key=None, features=None, label=None, categorical_variable=None):
        """
        Fit classification model based on training data.

        Parameters
        ----------

        data : DataFrame

            Training data.

        key : str, optional

            Name of the ID column. If ``key`` is not provided, it is assumed
            that the input has no ID column.

        features : list of str, optional

            Names of the feature columns.
            If ``features`` is not provided, it defaults to all non-ID,
            non-label columns.

        label : str, optional

            Name of the dependent variable.

            Defaults to the last column.
        """
        # pylint: disable=too-many-locals
        key = self._arg('key', key, str)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        categorical_variable = self._arg('categorical_variable', categorical_variable, ListOfStrings)
        cols = data.columns
        if key is not None:
            id_col = [key]
            cols.remove(key)
        else:
            id_col = []
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols

        data = data[id_col + features + [label]]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        tables = ['MODEL', 'STATS', 'OPTIMAL_PARAM']
        tables = ['#PAL_NAIVE_BAYES_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                  for name in tables]
        model_tbl, stats_tbl, opt_param_tbl = tables#pylint:disable=unused-variable

        param_rows = [
            ('LAPLACE', None, self.alpha, None),
            ('DISCRETIZATION', self.discretization, None, None),
            ('MODEL_FORMAT', self.model_format, None, None),
            ('THREAD_RATIO', None, self.thread_ratio, None),
            ('HAS_ID', key is not None, None, None),
        ]
        if self.categorical_variable is not None:
            param_rows.extend(('CATEGORICAL_VARIABLE', None, None, variable)
                              for variable in self.categorical_variable)
        if categorical_variable is not None:
            param_rows.extend(('CATEGORICAL_VARIABLE', None, None, variable)
                              for variable in categorical_variable)

        try:
            self._call_pal_auto('PAL_NAIVE_BAYES',
                                data,
                                ParameterTable().with_data(param_rows),
                                *tables)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            self._try_drop(tables)
            raise

        # pylint: disable=attribute-defined-outside-init
        self.model_ = self.conn_context.table(model_tbl)

    def predict(self, data, key, features=None, alpha=None,#pylint:disable=too-many-locals
                verbose=None,):
        """
        Predict based on fitted model.

        Parameters
        ----------

        data : DataFrame

            Independent variable values to predict for.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.

            If ``features`` is not provided, it defaults to all non-ID columns.

        alpha : float, optional

            Laplace smoothing value. Set a positive value to enable Laplace smoothing
            for categorical variables and use that value as the smoothing parameter.
            Set value 0 to disable Laplace smoothing.

            Defaults to the alpha value in the JSON model, if there is one, or
            0 otherwise.

        verbose : bool, optional

            If true, output all classes and the corresponding confidences
            for each data point.

            Defaults to False.

        Returns
        -------

        DataFrame

            Predicted result, structured as follows:

              - ID column, with the same name and type as ``data`` 's ID column.
              - CLASS, type NVARCHAR, predicted class name.
              - CONFIDENCE, type DOUBLE, confidence for the prediction of the
                sample, which is a logarithmic value of the posterior
                probabilities.

            .. note::

                A non-zero Laplace value (alpha) is required if there exist discrete
                category values that only occur in the test set. It can be read from
                JSON models or from the parameter ``alpha`` in predict().
                The Laplace value you set here takes precedence over the values
                read from JSON models.
        """
        if getattr(self, 'model_', None) is None:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")

        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        #it examines if the discrete value in test set appears in each class in training
        #letting pal takes of the check
        alpha = self._arg('alpha', alpha, float)
        verbose = self._arg('verbose', verbose, bool)

        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols

        data = data[[key] + features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        result_tbl = '#PAL_NAIVE_BAYES_RESULT_TBL_{}_{}'.format(self.id, unique_id)

        param_rows = [
            ('THREAD_RATIO', None, self.thread_ratio, None),
            ('VERBOSE_OUTPUT', verbose, None, None),
            ('LAPLACE', None, alpha, None)
            ]

        try:
            self._call_pal_auto("PAL_NAIVE_BAYES_PREDICT",
                                data,
                                self.model_,
                                ParameterTable().with_data(param_rows),
                                result_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            self._try_drop(result_tbl)
            raise

        return self.conn_context.table(result_tbl)

    def score(self, data, key, features=None, label=None, alpha=None, ):#pylint:disable= too-many-locals
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------

        data : DataFrame

            Data on which to assess model performance.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.
            If `features` is not provided, it defaults to all non-ID,
            non-label columns.

        label : str, optional

            Name of the dependent variable.

            Defaults to the last column.

        alpha : float, optional

            Laplace smoothing value. Set a positive value to enable Laplace smoothing
            for categorical variables and use that value as the smoothing parameter.
            Set value 0 to disable Laplace smoothing.

            Defaults to the alpha value in the JSON model, if there is one, or
            0 otherwise.

        Returns
        -------

        float :

            Mean accuracy on the given test data and labels.
        """

        if getattr(self, 'model_', None) is None:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")

        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)

        cols = data.columns
        cols.remove(key)
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols

        prediction = self.predict(data=data, key=key, alpha=alpha,
                                  features=features, verbose=False)
        prediction = prediction.select(key, 'CLASS').rename_columns(['ID_P', 'PREDICTION'])

        actual = data.select(key, label).rename_columns(['ID_A', 'ACTUAL'])
        joined = actual.join(prediction, 'ID_P=ID_A').select('ACTUAL', 'PREDICTION')

        return metrics.accuracy_score(self.conn_context, joined,
                                      label_true='ACTUAL',
                                      label_pred='PREDICTION')
