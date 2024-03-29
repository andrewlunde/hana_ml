"""
This module includes decision tree-based models for classification and regression.

The following classes are available:

    * :class:`DecisionTreeClassifier`
    * :class:`DecisionTreeRegressor`
    * :class:`RandomForestClassifier`
    * :class:`RandomForestRegressor`
    * :class:`GradientBoostingClassifier`
    * :class:`GradientBoostingRegressor`
    * :class:`HybridGradientBoostingClassifier`
    * :class:`HybridGradientBoostingRegressor`
"""

#pylint: disable=too-many-lines,line-too-long,too-many-arguments,relative-beyond-top-level
import logging
import uuid
from hdbcli import dbapi
from hana_ml.ml_exceptions import FitIncompleteError
from .pal_base import (
    PALBase,
    ParameterTable,
    ListOfStrings,
    ListOfTuples
)

from . import metrics
logger = logging.getLogger(__name__) #pylint: disable=invalid-name


class _RandomForestBase(PALBase):#pylint: disable=too-many-instance-attributes, too-few-public-methods
    """
    Base Random forest model for classification and regression.
    """

    #pylint:disable=too-many-arguments
    def __init__(self,
                 conn_context,
                 n_estimators=100,
                 max_features=None,
                 max_depth=None,
                 min_samples_leaf=None,
                 split_threshold=None,
                 calculate_oob=True,
                 random_state=None,
                 thread_ratio=None,
                 allow_missing_dependent=True,
                 categorical_variable=None, #move to fit()
                 sample_fraction=None
                ):
        super(_RandomForestBase, self).__init__(conn_context)
        self.n_estimators = self._arg('n_estimators', n_estimators, int)
        self.max_features = self._arg('max_features', max_features, int)
        self.max_depth = self._arg('max_depth', max_depth, int)
        self.min_samples_leaf = self._arg('min_samples_leaf', min_samples_leaf, int)
        self.split_threshold = self._arg('split_threshold', split_threshold, float)
        self.calculate_oob = self._arg('calculate_oob', calculate_oob, bool)
        self.random_state = self._arg('random_state', random_state, int)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.allow_missing_dependent = self._arg('allow_missing_dependent',
                                                 allow_missing_dependent, bool)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        self.categorical_variable = self._arg('categorical_variable',
                                              categorical_variable, ListOfStrings)
        self.sample_fraction = self._arg('sample_fraction', sample_fraction, float)
        self.strata = None
        self.priors = None

    #has_id default value is inconsistent with document
    def fit(self, data, key=None, features=None, label=None, categorical_variable=None):
        #pylint:disable=too-many-locals, too-many-branches, too-many-statements
        """
        Train the model on input data.

        Parameters
        ----------

        data : DataFrame

            Training data.

        key : str, optional

            Name of the ID column. If ``key`` is not provided, it is assumed \
            that the input has no ID column.

        features : list of str, optional

            Names of the feature columns. \
            If ``features`` is not provided, it defaults to all non-ID, \
            non-label columns.

        label : str, optional

            Name of the dependent variable.

            Defaults to the last column.
        categorical_variable : str or list of str, optional

            Specify INTEGER column(s) that should be be treated as categorical \
            data. Other INTEGER columns will be treated as continuous.
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

        n_features = len(features)
        if self.max_features is not None and self.max_features > n_features:
            msg = ('max_features should not be larger than the number of features in the input.')
            logger.error(msg)
            raise ValueError(msg)

        data = data[id_col + features + [label]]

        tables = ['MODEL', 'VAR_IMPORTANCE', 'OOB_ERR', 'CM']
        tables = ['#PAL_RANDOM_FOREST_{}_TBL_{}'.format(name, self.id)
                  for name in tables]
        model_tbl, var_importance_tbl, oob_err_tbl, cm_tbl = tables

        param_rows = [
            ('TREES_NUM', self.n_estimators, None, None),
            ('TRY_NUM', self.max_features, None, None),
            ('MAX_DEPTH', self.max_depth, None, None),
            ('NODE_SIZE', self.min_samples_leaf, None, None),
            ('SPLIT_THRESHOLD', None, self.split_threshold, None),
            ('CALCULATE_OOB', self.calculate_oob, None, None),
            ('SEED', self.random_state, None, None),
            ('THREAD_RATIO', None, self.thread_ratio, None),
            ('ALLOW_MISSING_DEPENDENT', self.allow_missing_dependent, None, None),
            ('SAMPLE_FRACTION', None, self.sample_fraction, None),
            ('HAS_ID', key is not None, None, None),
            ]
        if self.categorical_variable is not None:
            param_rows.extend(('CATEGORICAL_VARIABLE', None, None, variable)
                              for variable in self.categorical_variable)
        if categorical_variable is not None:
            param_rows.extend(('CATEGORICAL_VARIABLE', None, None, variable)
                              for variable in categorical_variable)

        if self.strata is not None:
            label_t = data.dtypes([label])[0][1]
            if label_t == 'INT':
                param_rows.extend(('STRATA', class_type, fraction, None)
                                  for class_type, fraction in self.strata)
            elif label_t in {'VARCHAR', 'NVARCHAR'}:
                param_rows.extend(('STRATA', None, fraction, class_type)
                                  for class_type, fraction in self.strata)
        if self.priors is not None:
            label_t = data.dtypes([label])[0][1]
            if label_t == 'INT':
                param_rows.extend(('PRIORS', class_type, prior_prob, None)
                                  for class_type, prior_prob in self.priors)
            elif label_t in {'VARCHAR', 'NVARCHAR'}:
                param_rows.extend(('PRIORS', None, prior_prob, class_type,)
                                  for class_type, prior_prob in self.priors)

        #model_spec = [
        #    ("ROW_INDEX", INTEGER),
        #    ("TREE_INDEX", INTEGER),
        #    ("MODEL_CONTENT", NVARCHAR(5000))
        #    ]

        #var_importance_spec = [
        #    ("VARIABLE_NAME", NVARCHAR(200)),
        #    ("IMPORTANCE", DOUBLE)
        #    ]

        #oob_err_spec = [
        #    ("TREE_INDEX", INTEGER),
        #    ("ERROR", DOUBLE)
        #    ]

        #document is in consistent from the actual behavior
        #cm_spec = [
        #    ("ACTUAL_CLASS", NVARCHAR(1000)),
        #    ("PREDICTED_CLASS", NVARCHAR(1000)),
        #    ("COUNT", INTEGER)
        #    ]

        try:
            #self._materialize(data_tbl, data)
            #self._create(ParameterTable(param_tbl).with_data(param_rows))
            #self._create(Table(model_tbl, model_spec))
            #self._create(Table(var_importance_tbl, var_importance_spec))
            #self._create(Table(oob_err_tbl, oob_err_spec))
            #self._create(Table(cm_tbl, cm_spec))
            self._call_pal_auto("PAL_RANDOM_DECISION_TREES",
                                data,
                                ParameterTable().with_data(param_rows),
                                *tables)
        except dbapi.Error as db_err:
            #msg = 'HANA error while attempting to fit random forest model.'
            logger.exception(str(db_err))
            self._try_drop(tables)
            raise

        # pylint: disable=attribute-defined-outside-init
        conn = self.conn_context
        self.model_ = conn.table(model_tbl)
        self.feature_importances_ = conn.table(var_importance_tbl)
        self.oob_error_ = conn.table(oob_err_tbl) if self.calculate_oob else None
        self._confusion_matrix_ = conn.table(cm_tbl)
        #cm_tbl is empty when calculate_oob is False in PAL

    missing_replacement_map = {'feature_marginalized':1, 'instance_marginalized':2}
    def _predict(self, data, key, features=None, verbose=None,#pylint:disable=too-many-locals
                 block_size=None, missing_replacement=None):
        """
        Predict dependent variable values based on fitted model.

        Parameters
        ----------

        data : DataFrame

            Independent variable values to predict for.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns. \
            If ``features`` is not provided, it defaults to all non-ID columns.

        block_size : int, optional

            The number of rows loaded per time during prediction. \
            0 indicates load all data at once.

            Defaults to 0.

        missing_replacement : str, optional

            The missing replacement strategy:
                - 'feature_marginalized': marginalise each missing feature out \
                independently.
                - 'instance_marginalized': marginalise all missing features \
                in an instance as a whole corresponding to each category.

            Defaults to feature_marginalized.

        verbose : bool, optional

            If true, output all classes and the corresponding confidences \
            for each data point. \
            This parameter is valid only for classification.

            Defaults to False.

        Returns
        -------

        DataFrame

            DataFrame of score and confidence, structured as follows:
                - ID column, with same name and type as ``data`'s ID column.
                - SCORE, type DOUBLE, representing the predicted classes/values.
                - CONFIDENCE, type DOUBLE, representing the confidence of \
                a class, all 0s if for regression.
        """
        if getattr(self, 'model_', None) is None:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")

        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        verbose = self._arg('verbose', verbose, bool)
        block_size = self._arg('block_size', block_size, int)
        missing_replacement = self._arg('missing_replacement',
                                        missing_replacement,
                                        self.missing_replacement_map)

        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols

        data = data[[key] + features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()

        #tables = data_tbl, model_tbl, param_tbl, result_tbl = [
        result_tbl = '#PAL_RANDOM_FOREST_RESULT_TBL_{}_{}'.format(self.id, unique_id)
        #    for name in ['DATA', 'MODEL', 'PARAM', 'FITTED']]

        param_rows = [
            ('THREAD_RATIO', None, self.thread_ratio, None),
            ('VERBOSE', verbose, None, None),
            ('BLOCK_SIZE', block_size, None, None),
            ('MISSING_REPLACEMENT', missing_replacement, None, None)
            ]

        #result_spec = [
        #    (parse_one_dtype(data.dtypes([data.columns[0]])[0])),
        #    ("SCORE", NVARCHAR(100)),
        #    ("CONFIDENCE", DOUBLE)
        #]

        try:
            #self._materialize(data_tbl, data)
            #self._materialize(model_tbl, self.model_)
            #self._create(ParameterTable(param_tbl).with_data(param_rows))
            #self._create(Table(result_tbl, result_spec))
            self._call_pal_auto("PAL_RANDOM_DECISION_TREES_PREDICT",
                                data,
                                self.model_,
                                ParameterTable().with_data(param_rows),
                                result_tbl)
        except dbapi.Error as db_err:
            #msg = 'HANA error during random forest prediction.'
            logger.exception(str(db_err))
            self._try_drop(result_tbl)
            raise

        return self.conn_context.table(result_tbl)

class RandomForestClassifier(_RandomForestBase):#pylint: disable=too-many-instance-attributes
    """
    Random forest model for classification.

    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA system.

    n_estimators : int, optional

        Specifies the number of trees in the random forest.

        Defaults to 100.

    max_features : int, optional

        Specifies the number of randomly selected splitting variables. \
        Should not be larger than the number of input features.

        Defaults to sqrt(p) (for classification) or p/3(for regression), \
        where p is the number of input features.

    max_depth : int, optional

        The maximum depth of a tree.

        By default it is unlimited.

    min_samples_leaf : int, optional

        Specifies the minimum number of records in a leaf.

        Defaults to 1 for classification.

    split_threshold : float, optional

        Specifies the stop condition: if the improvement value of the best \
        split is less than this value, the tree stops growing.

        Defaults to 1e-5.

    calculate_oob : bool, optional

        If true, calculate the out-of-bag error.

        Defaults to True.

    random_state : {'0', 'Others'}, int, optional

        Specifies the seed for random number generator.

            0: Uses the current time (in seconds) as the seed.

            Others: Uses the specified value as the seed.

        Defaults to 0.

    thread_ratio : float, optional

        Controls the proportion of available threads to use. \
        The value range is from 0 to 1, where 0 indicates a single thread, \
        and 1 indicates up to all available threads. Values between 0 and 1 \
        will use that percentage of available threads. Values outside this \
        range tell PAL to heuristically determine the number of threads to use.

        Defaults to heuristically determined.

    allow_missing_dependent : bool, optional

        Specifies if a missing target value is allowed.

            - False: Not allowed. An error occurs if a missing target is present.
            - True: Allowed. The datum with the missing target is removed.

        Defaults to True.

    categorical_variable : str or list of str, optional

        Specifies INTEGER column(s) should be treated as categorical. \
        The default behavior is: \
        string: categorical, or integer and float: continuous. \
        VALID only for integer variables; omitted otherwise.

        Default value detected from input data.

    sample_fraction : float, optional

        The fraction of data used for training. \
        Assume there are n pieces of data, sample fraction is r, then n*r \
        data is selected for training.

        Defaults to 1.0.

    strata : List of tuples: (class, fraction), optional

        Strata proportions for stratified sampling. A (class, fraction) tuple \
        specifies that rows with that class should make up the specified \
        fraction of each sample. If the given fractions do not add up to 1, \
        the remaining portion is divided equally between classes with \
        no entry in ``strata``, or between all classes if all classes have \
        an entry in ``strata``. \
        If ``strata`` is not provided, bagging is used instead of stratified \
        sampling.

    priors : List of tuples: (class, prior_prob), optional

        Prior probabilities for classes. A (class, prior_prob) tuple \
        specifies the prior probability of this class. If the given \
        priors do not add up to 1, the remaining portion is divided equally \
        between classes with no entry in ``priors``, or between all classes \
        if all classes have an entry in 'priors'. \
        If ``priors`` is not provided, it is determined by the proportion of \
        every class in the training data.

    Attributes
    ----------

    model_ : DataFrame

        Trained model content.

    feature_importances_ : DataFrame

        The feature importance (the higher, the more important the feature).

    oob_error_ : DataFrame

        Out-of-bag error rate or mean squared error for random forest up \
        to indexed tree. \
        Set to None if ``calculate_oob`` is False.

    confusion_matrix_ : DataFrame

        Confusion matrix used to evaluate the performance of classification \
        algorithms.

    Examples
    --------

    Input dataframe for training:

    >>> df1.head(4).collect()
       OUTLOOK  TEMP  HUMIDITY WINDY       LABEL
    0    Sunny  75.0      70.0   Yes        Play
    1    Sunny   NaN      90.0   Yes Do not Play
    2    Sunny  85.0       NaN    No Do not Play
    3    Sunny  72.0      95.0    No Do not Play

    Creating RandomForestClassifier instance:

    >>> rfc = RandomForestClassifier(conn_context=cc, n_estimators=3,
    ...                              max_features=3, random_state=2,
    ...                              split_threshold=0.00001,
    ...                              calculate_oob=True,
    ...                              min_samples_leaf=1, thread_ratio=1.0)

    Performing fit() on given dataframe:

    >>> rfc.fit(df1, features=['OUTLOOK', 'TEMP', 'HUMIDITY', 'WINDY'],
    ...         label='LABEL')
    >>> rfc.feature_importances_.collect()
      VARIABLE_NAME  IMPORTANCE
    0       OUTLOOK    0.449550
    1          TEMP    0.216216
    2      HUMIDITY    0.208108
    3         WINDY    0.126126

    Input dataframe for predicting:

    >>> df2.collect()
       ID   OUTLOOK     TEMP  HUMIDITY WINDY
    0   0  Overcast     75.0  -10000.0   Yes
    1   1      Rain     78.0      70.0   Yes

    Performing predict() on given dataframe:

    >>> result = rfc.predict(df2, key='ID', verbose=False)
    >>> result.collect()
       ID SCORE  CONFIDENCE
    0   0  Play    0.666667
    1   1  Play    0.666667

    Input dataframe for scoring:

    >>> df3.collect()
       ID   OUTLOOK  TEMP  HUMIDITY WINDY LABEL
    0   0     Sunny    70      90.0   Yes  Play
    1   1  Overcast    81      90.0   Yes  Play
    2   2      Rain    65      80.0    No  Play

    Performing score() on given dataframe:

    >>> rfc.score(df3, key='ID')
    0.6666666666666666
    """


    #pylint:disable=too-many-arguments
    def __init__(self,
                 conn_context,
                 n_estimators=100,
                 max_features=None,
                 max_depth=None,
                 min_samples_leaf=1,
                 split_threshold=None,
                 calculate_oob=True,
                 random_state=None,
                 thread_ratio=None,
                 allow_missing_dependent=True,
                 categorical_variable=None,
                 sample_fraction=None,
                 strata=None,
                 priors=None):
        #pylint:disable=too-many-locals
        super(RandomForestClassifier, self).__init__(conn_context, n_estimators,
                                                     max_features, max_depth,
                                                     min_samples_leaf, split_threshold,
                                                     calculate_oob, random_state, thread_ratio,
                                                     allow_missing_dependent, categorical_variable,
                                                     sample_fraction)
        self.strata = self._arg('strata', strata, ListOfTuples)
        self.priors = self._arg('priors', priors, ListOfTuples)

    def fit(self, data, key=None, features=None, label=None, categorical_variable=None):
        """
        Train the model on input data.

        Parameters
        ----------

        data : DataFrame

            Training data.

        key : str, optional

            Name of the ID column. If ``key`` is not provided, it is assumed \
            that the input has no ID column.

        features : list of str, optional

            Names of the feature columns. \
            If ``features`` is not provided, it defaults to all non-ID, \
            non-label columns.

        label : str, optional

            Name of the dependent variable.

            Defaults to the last column.

        categorical_variable : str or list of str, optional

            Specifies INTEGER column(s) that should be treated as categorical. \
            Other INTEGER columns will be treated as continuous.
        """
        # SQLTRACE
        self.conn_context.sql_tracer.set_sql_trace(self, self.__class__.__name__, 'Fit')

        super(RandomForestClassifier, self).fit(data, key, features, label, categorical_variable)
        # pylint: disable=attribute-defined-outside-init
        self.confusion_matrix_ = self._confusion_matrix_

    def predict(self, data, key, features=None, verbose=None,
                block_size=None, missing_replacement=None):#pylint:disable=too-many-locals
        """
        Predict dependent variable values based on fitted model.

        Parameters
        ----------

        data : DataFrame

            Independent variable values to predict for.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns. \
            If ``features`` is not provided, it defaults to all non-ID columns.

        block_size : int, optional

            The number of rows loaded per time during prediction. \
            0 indicates load all data at once.

            Defaults to 0.

        missing_replacement : str, optional

            The missing replacement strategy:

                - 'feature_marginalized': marginalise each missing feature out\
                   independently.
                - 'instance_marginalized': marginalise all missing features \
                  in an instance as a whole corresponding to each category.

            Defaults to feature_marginalized.

        verbose : bool, optional

            If true, output all classes and the corresponding confidences \
            for each data point.

        Returns
        -------

        DataFrame

            DataFrame of score and confidence, structured as follows:
                - ID column, with same name and type as ``data`` 's ID column.
                - SCORE, type DOUBLE, representing the predicted classes.
                - CONFIDENCE, type DOUBLE, representing the confidence of \
                a class.
        """
         # SQLTRACE
        self.conn_context.sql_tracer.set_sql_trace(self, self.__class__.__name__, 'Predict')

        return super(RandomForestClassifier, self)._predict(data=data, key=key,
                                                            features=features,
                                                            verbose=verbose,
                                                            block_size=block_size,
                                                            missing_replacement=missing_replacement)

    def score(self, data, key, features=None, label=None,
              block_size=None, missing_replacement=None):#pylint:disable=too-many-locals
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------

        data : DataFrame

            Data on which to assess model performance.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns. \
            If ``features`` is not provided, it defaults to all non-ID, \
            non-label columns.

        label : str, optional

            Name of the dependent variable.

            Defaults to the last column.

        block_size : int, optional

            The number of rows loaded per time during prediction. \
            0 indicates load all data at once.

            Defaults to 0.

        missing_replacement : str, optional

            The missing replacement strategy:
                - 'feature_marginalized': marginalise each missing feature out \
                independently.
                - 'instance_marginalized': marginalise all missing features \
                in an instance as a whole corresponding to each category.

            Defaults to feature_marginalized.

        Returns
        -------

        float

            Mean accuracy on the given test data and labels.
        """
        if getattr(self, 'model_', None) is None:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")

        # SQLTRACE
        self.conn_context.sql_tracer.set_sql_trace(self, self.__class__.__name__, 'Score')

        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        #input check for block_size and missing_replacement is done in predict()

        cols = data.columns
        cols.remove(key)
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols

        prediction = self.predict(data=data, key=key, features=features,
                                  block_size=block_size,
                                  missing_replacement=missing_replacement)
        prediction = prediction.select(key, 'SCORE').rename_columns(['ID_P', 'PREDICTION'])

        actual = data.select(key, label).rename_columns(['ID_A', 'ACTUAL'])
        joined = actual.join(prediction, 'ID_P=ID_A').select('ACTUAL', 'PREDICTION')

        return metrics.accuracy_score(self.conn_context, joined,
                                      label_true='ACTUAL',
                                      label_pred='PREDICTION')


class RandomForestRegressor(_RandomForestBase):#pylint: disable=too-many-instance-attributes
    """
    Random forest model for regression.

    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA system.

    n_estimators : int, optional

        Specifies the number of trees in the random forest.

        Defaults to 100.

    max_features : int, optional

        Specifies the number of randomly selected splitting variables. \
        Should not be larger than the number of input features.

        Defaults to sqrt(p) (for classification) or p/3(for regression), \
        where p is the number of input features.

    max_depth : int, optional

        The maximum depth of a tree.

        By default it is unlimited.

    min_samples_leaf : int, optional

        Specifies the minimum number of records in a leaf.

        Defaults to 5 for regression.

    split_threshold : float, optional

        Specifies the stop condition: if the improvement value of the best \
        split is less than this value, the tree stops growing.

        Defaults to 1e-5.

    calculate_oob : bool, optional

        If true, calculate the out-of-bag error.

        Defaults to True.

    random_state : {'0', 'Others'}, int, optional

        Specifies the seed for random number generator.

            0: Uses the current time (in seconds) as the seed.

            Others: Uses the specified value as the seed.

        Defaults to 0.

    thread_ratio : float, optional

        Controls the proportion of available threads to use. \
        The value range is from 0 to 1, where 0 indicates a single thread, \
        and 1 indicates up to all available threads. Values between 0 and \
        1 will use up to that percentage of available threads. Values outside \
        this range tell PAL to heuristically determine the number of threads \
        to use.

        Defaults to heuristically determined.

    allow_missing_dependent : bool, optional

        Specifies if a missing target value is allowed.

            - False: Not allowed. An error occurs if a missing target is present.
            - True: Allowed. The datum with a missing target is removed.

        Defaults to True.

    categorical_variable : str or list of str, optional

        Specifies INTEGER column(s) that should be treated as categorical. \
        The default behavior is: \
        string: categorical, or integer and float: continuous. \
        VALID only for integer variables; omitted otherwise.

        Default value detected from input data.

    sample_fraction : float, optional

        The fraction of data used for training.\
        Assume there are n pieces of data, sample fraction is r, then n*r \
        data is selected for training.

        Defaults to 1.0.

    Attributes
    ----------

    model_ : DataFrame

        Trained model content.

    feature_importances_ : DataFrame

        The feature importance (the higher, the more important the feature).

    oob_error_ : DataFrame

        Out-of-bag error rate or mean squared error for random forest \
        up to indexed tree. \
        Set to None if ``calculate_oob`` is False.

    Examples
    --------

    Input dataframe for training:

    >>> df1.head(5).collect()
       ID         A         B         C         D       CLASS
    0   0 -0.965679  1.142985 -0.019274 -1.598807  -23.633813
    1   1  2.249528  1.459918  0.153440 -0.526423  212.532559
    2   2 -0.631494  1.484386 -0.335236  0.354313   26.342585
    3   3 -0.967266  1.131867 -0.684957 -1.397419  -62.563666
    4   4 -1.175179 -0.253179 -0.775074  0.996815 -115.534935

    Creating RandomForestRegressor instance:

    >>> rfr = RandomForestRegressor(conn_context=cc, random_state=3)

    Performing fit() on given dataframe:

    >>> rfr.fit(df1, key='ID')
    >>> rfr.feature_importances_.collect()
       VARIABLE_NAME  IMPORTANCE
    0             A    0.249593
    1             B    0.381879
    2             C    0.291403
    3             D    0.077125

    Input dataframe for predicting:

    >>> df2.collect()
       ID         A         B         C         D
    0   0  1.081277  0.204114  1.220580 -0.750665
    1   1  0.524813 -0.012192 -0.418597  2.946886

    Performing predict() on given dataframe:

    >>> result = rfr.predict(df2, key='ID')
    >>> result.collect()
       ID    SCORE  CONFIDENCE
    0   0    48.126   62.952884
    1   1  -10.9017   73.461039

    Input dataframe for scoring:

    >>> df3.head(5).collect()
        ID         A         B         C         D       CLASS
    0    0  1.081277  0.204114  1.220580 -0.750665   139.10170
    1    1  0.524813 -0.012192 -0.418597  2.946886    52.17203
    2    2 -0.280871  0.100554 -0.343715 -0.118843   -34.69829
    3    3 -0.113992 -0.045573  0.957154  0.090350    51.93602
    4    4  0.287476  1.266895  0.466325 -0.432323   106.63425

    Performing score() on given dataframe:

    >>> rfr.score(df3, key='ID')
    0.6530768858159514
    """
    def predict(self, data, key, features=None, block_size=None, missing_replacement=None):#pylint:disable=too-many-locals, too-many-arguments
        """
        Predict dependent variable values based on fitted model.

        Parameters
        ----------

        data : DataFrame

            Independent variable values to predict for.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns. \
            If ``features`` is not provided, it defaults to all non-ID columns.

        block_size : int, optional

            The number of rows loaded per time during prediction. \
            0 indicates load all data at once.

            Defaults to 0.

        missing_replacement : str, optional

            The missing replacement strategy:

                - 'feature_marginalized': marginalise each missing feature out
                   independently.
                - 'instance_marginalized': marginalise all missing features
                   in an instance as a whole corresponding to each category.

            Defaults to feature_marginalized.

        Returns
        -------

        DataFrame

            DataFrame of score and confidence, structured as follows:

                - ID column, with same name and type as ``data``'s ID column.
                - SCORE, type DOUBLE, representing the predicted values.
                - CONFIDENCE, all 0s.

                It is included due to the fact PAL uses the same table for \
                classification.
        """
        return super(RandomForestRegressor, self)._predict(data=data, key=key,
                                                           features=features,
                                                           block_size=block_size,
                                                           missing_replacement=missing_replacement)

    def score(self, data, key, features=None, label=None,#pylint:disable=too-many-locals, too-many-arguments
              block_size=None, missing_replacement=None):
        """
        Returns the coefficient of determination R^2 of the prediction.

        Parameters
        ----------

        data : DataFrame

            Data on which to assess model performance.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns. \
            If ``features`` is not provided, it defaults to all non-ID, \
            non-label columns.

        label : str, optional

            Name of the dependent variable.

            Defaults to the last column.

        block_size : int, optional

            The number of rows loaded per time during prediction. \
            0 indicates load all data at once.

            Defaults to 0.

        missing_replacement : str, optional

            The missing replacement strategy:

                - 'feature_marginalized': marginalise each missing feature out \
                independently.

                - 'instance_marginalized': marginalise all missing features \
                in an instance as a whole corresponding to each category.

            Defaults to feature_marginalized.

        Returns
        -------

        float

            The coefficient of determination R^2 of the prediction on the
            given data.
        """
        if getattr(self, 'model_', None) is None:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")

        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        #input check for block_size and missing_replacement is done in predict()

        cols = data.columns
        cols.remove(key)
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols

        prediction = self.predict(data=data, key=key, features=features,
                                  block_size=block_size,
                                  missing_replacement=missing_replacement)
        original = data[[key, label]]
        prediction = prediction.select([key, 'SCORE']).rename_columns(['ID_P', 'PREDICTION'])
        original = data[[key, label]].rename_columns(['ID_A', 'ACTUAL'])
        joined = original.join(prediction, 'ID_P=ID_A').select('ACTUAL', 'PREDICTION')
        return metrics.r2_score(self.conn_context,
                                joined,
                                label_true='ACTUAL',
                                label_pred='PREDICTION')


class _DecisionTreeBase(PALBase):#pylint: disable=too-many-instance-attributes
    """
    Base Decision tree model for classification and regression."""
    algorithm_map = {}
    model_format_map = {'json':1, 'pmml':2}
    discretization_type_map = {'mdlpc':0, 'equal_freq':1}
    #pylint:disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
    def __init__(self,
                 conn_context,
                 algorithm,
                 thread_ratio=None,
                 allow_missing_dependent=True,
                 percentage=None,
                 min_records_of_parent=None,
                 min_records_of_leaf=None,
                 max_depth=None,
                 categorical_variable=None,
                 split_threshold=None,
                 use_surrogate=None,
                 model_format=None,
                 output_rules=True
                ):
        super(_DecisionTreeBase, self).__init__(conn_context)
        self.algorithm = self._arg('algorithm', algorithm, self.algorithm_map, required=True)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.allow_missing_dependent = self._arg('allow_missing_dependent',
                                                 allow_missing_dependent, bool)
        self.percentage = self._arg('percentage', percentage, float)
        self.min_records_of_parent = self._arg('min_records_of_parent', min_records_of_parent, int)
        self.min_records_of_leaf = self._arg('min_records_of_leaf', min_records_of_leaf, int)
        self.max_depth = self._arg('max_depth', max_depth, int)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        self.categorical_variable = self._arg('categorical_variable',
                                              categorical_variable, ListOfStrings)
        self.split_threshold = self._arg('split_threshold', split_threshold, float)
        self.discretization_type = None
        self.bins = None
        self.max_branch = None
        self.merge_threshold = None
        self.use_surrogate = self._arg('use_surrogate', use_surrogate, bool)
        if use_surrogate is not None:
            if algorithm.lower() != 'cart':
                msg = ("use_surrogate is inapplicable. " +
                       "It is only applicable when algorithm is cart.")
                logger.error(msg)
                raise ValueError(msg)
        self.model_format = self._arg('model_format', model_format, self.model_format_map)
        self.output_rules = self._arg('output_rules', output_rules, bool)
        self.output_confusion_matrix = None
        self.priors = None

    #has_id default value is inconsistent with document
    def fit(self, data, key=None, features=None, label=None, categorical_variable=None):
        #pylint:disable=too-many-locals, too-many-branches, too-many-statements
        """
        Train the model on input data.

        Parameters
        ----------

        data : DataFrame

            Training data.

        key : str, optional

            Name of the ID column. If ``key`` is not provided, it is assumed \
            that the input has no ID column.

        features : list of str, optional

            Names of the feature columns. \
            If ``features`` is not provided, it defaults to all non-ID, \
            non-label columns.

        label : str, optional

            Name of the dependent variable.

            Defaults to the last column.

        categorical_variable : str or list of str, optional

            Specifies INTEGER column(s) that should be treated as categorical \
            data. Other INTEGER columns will be treated as continuous.
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
        tables = ['MODEL', 'RULES', 'CM', 'STATS', 'CV']
        tables = ['#PAL_DECISION_TREE_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                  for name in tables]
        model_tbl, rules_tbl, cm_tbl, stats_tbl, cv_tbl = tables#pylint:disable=unused-variable

        param_rows = [
            ('ALGORITHM', self.algorithm, None, None),
            ('THREAD_RATIO', None, self.thread_ratio, None),
            ('ALLOW_MISSING_DEPENDENT', self.allow_missing_dependent, None, None),
            ('PERCENTAGE', None, self.percentage, None),
            ('MIN_RECORDS_OF_PARENT', self.min_records_of_parent, None, None),
            ('MIN_RECORDS_OF_LEAF', self.min_records_of_leaf, None, None),
            ('MAX_DEPTH', self.max_depth, None, None),
            ('SPLIT_THRESHOLD', None, self.split_threshold, None),
            ('DISCRETIZATION_TYPE', self.discretization_type, None, None),
            ('MAX_BRANCH', self.max_branch, None, None),
            ('MERGE_THRESHOLD', None, self.merge_threshold, None),
            ('USE_SURROGATE', self.use_surrogate, None, None),
            ('MODEL_FORMAT', self.model_format, None, None),
            ('IS_OUTPUT_RULES', self.output_rules, None, None),
            ('IS_OUTPUT_CONFUSION_MATRIX', self.output_confusion_matrix, None, None),
            ('HAS_ID', key is not None, None, None),
            ]
        if self.categorical_variable is not None:
            param_rows.extend(('CATEGORICAL_VARIABLE', None, None, variable)
                              for variable in self.categorical_variable)
        if categorical_variable is not None:
            param_rows.extend(('CATEGORICAL_VARIABLE', None, None, variable)
                              for variable in categorical_variable)
        if self.priors is not None:
            param_rows.extend(('{}_PRIOR_'.format(class_type), None, prior_prob, None)
                              for class_type, prior_prob in self.priors)
        if self.bins is not None:
            param_rows.extend(('{}_BIN_'.format(col_name), num_bins, None, None)
                              for col_name, num_bins in self.bins)

        #model_spec = [
        #    ("ROW_INDEX", INTEGER),
        #    ("MODEL_CONTENT", NVARCHAR(5000))
        #    ]

        #rules_spec = [
        #    ("ROW_INDEX", INTEGER),
        #    ("RULES_CONTENT", NVARCHAR(5000))
        #    ]

        #cm_spec = [
        #    ("ACTUAL_CLASS", NVARCHAR(1000)),
        #    ("PREDICTED_CLASS", NVARCHAR(1000)),
        #    ("COUNT", INTEGER)
        #    ]

        #stats_spec = [
        #    ("STAT_NAME", NVARCHAR(1000)),
        #    ("STAT_VALUE", NVARCHAR(1000))
        #    ]

        try:
            #self._materialize(data_tbl, data)
            #self._create(ParameterTable(param_tbl).with_data(param_rows))
            #self._create(Table(model_tbl, model_spec))
            #self._create(Table(rules_tbl, rules_spec))
            #self._create(Table(cm_tbl, cm_spec))
            #self._create(Table(stats_tbl, stats_spec))
            #self._create(ParameterTable(cv_tbl))
            self._call_pal_auto("PAL_DECISION_TREE",
                                data,
                                ParameterTable().with_data(param_rows),
                                *tables)
        except dbapi.Error as db_err:
            #msg = 'HANA error while attempting to fit decision tree model.'
            logger.exception(str(db_err))
            self._try_drop(tables)
            raise
        # pylint: disable=attribute-defined-outside-init
        conn = self.conn_context
        self.model_ = conn.table(model_tbl)
        self.decision_rules_ = conn.table(rules_tbl) if self.output_rules else None
        self._confusion_matrix_ = conn.table(cm_tbl)

    def predict(self, data, key, features=None, verbose=False):#pylint:disable=too-many-locals
        """
        Predict dependent variable values based on fitted model.

        Parameters
        ----------

        data : DataFrame

            Independent variable values to predict for.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns. \
            If ``features`` is not provided, it defaults to all non-ID columns.

        verbose : bool, optional

            If true, output all classes and the corresponding confidences \
            for each data point. \
            This parameter is valid only for classification.

            Defaults to False.

        Returns
        -------

        DataFrame
            DataFrame of score and confidence, structured as follows:
                - ID column, with same name and type as ``data``'s ID column.
                - SCORE, type DOUBLE, representing the predicted classes/values.
                - CONFIDENCE, type DOUBLE, representing the confidence of \
                  a class, all 0s if for regression.
        """
        if getattr(self, 'model_', None) is None:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")

        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        verbose = self._arg('verbose', verbose, bool)

        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols

        data = data[[key] + features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()

        #tables = data_tbl, model_tbl, param_tbl, result_tbl = [
        result_tbl = '#PAL_DECISION_TREE_RESULT_TBL_{}_{}'.format(self.id, unique_id)
        #    for name in ['DATA', 'MODEL', 'PARAM', 'RESULT']]

        param_rows = [
            ('THREAD_RATIO', None, self.thread_ratio, None),
            ('VERBOSE', int(verbose), None, None),
            ]

        #result_spec = [
        #    (parse_one_dtype(data.dtypes([data.columns[0]])[0])),
        #    ("SCORE", NVARCHAR(100)),
        #    ("CONFIDENCE", DOUBLE)
        #]

        try:
            #self._materialize(data_tbl, data)
            #self._materialize(model_tbl, self.model_)
            #self._create(ParameterTable(param_tbl).with_data(param_rows))
            #self._create(Table(result_tbl, result_spec))
            self._call_pal_auto("PAL_DECISION_TREE_PREDICT",
                                data,
                                self.model_,
                                ParameterTable().with_data(param_rows),
                                result_tbl)
        except dbapi.Error as db_err:
            #msg = 'HANA error during decision tree prediction.'
            logger.exception(str(db_err))
            self._try_drop(result_tbl)
            raise

        return self.conn_context.table(result_tbl)


class DecisionTreeClassifier(_DecisionTreeBase):
    """
    Decision Tree model for classification.

    Parameters
    ----------

    conn_context : ConnectionContext

        Database connection object.

    algorithm : {'c45', 'chaid', 'cart'}

        Algorithm used to grow a decision tree. Case-insensitive.

            - 'c45': C4.5 algorithm.
            - 'chaid': Chi-square automatic interaction detection.
            - 'cart': Classification and regression tree.

    thread_ratio : float, optional

        Controls the proportion of available threads to use. \
        The value range is from 0 to 1, where 0 indicates a single thread, \
        and 1 indicates up to all available threads. Values between 0 and \
        1 will use up to that percentage of available threads. Values outside \
        this range tell PAL to heuristically determine the number of threads \
        to use.

        Defaults to 0.

    allow_missing_dependent : bool, optional

        Specifies if a missing target value is allowed.

            - False: Not allowed. An error occurs if a missing target is present.
            - True: Allowed. The datum with the missing target is removed.

        Defaults to True.

    percentage : float, optional

        Specifies the percentage of the input data that will be used to build \
        the tree model. The rest of the data will be used for pruning.

        Defaults to 1.0.

    min_records_of_parent : int, optional

        Specifies the stop condition: if the number of records in one node is \
        less than the specified value, the algorithm stops splitting.

        Defaults to 2.

    min_records_of_leaf : int, optional

        Promises the minimum number of records in a leaf.

        Defaults to 1.

    max_depth : int, optional

        The maximum depth of a tree.

        By default it is unlimited.

    categorical_variable : str or list of str, optional

        Specifies INTEGER column(s) that should be treated as categorical. \
        The default behavior is: \
        string: categorical, or integer and float: continuous. \
        VALID only for integer variables; omitted otherwise.

        Default value detected from input data.

    split_threshold : float, optional

        Specifies the stop condition for a node:

            - C45: The information gain ratio of the best split is less than \
              this value.
            - CHAID: The p-value of the best split is greater than or equal \
              to this value.
            - CART: The reduction of Gini index or relative MSE of the best \
              split is less than this value.

        The smaller the SPLIT_THRESHOLD value is, the larger a C45 or CART \
        tree grows. \
        On the contrary, CHAID will grow a larger tree with larger \
        SPLIT_THRESHOLD value.

        Defaults to 1e-5 for C45 and CART, 0.05 for CHAID.

    discretization_type : {'mdlpc', 'equal_freq'}, optional

        Strategy for discretizing continuous attributes. Case-insensitive.

            - 'mdlpc': Minimum description length principle criterion.
            - 'equal_freq': Equal frequency discretization.

        Valid only for C45 and CHAID.

        Defaults to mdlpc.

    bins : List of tuples: (column name, number of bins), optional

        Specifies the number of bins for discretization. \
        Only valid when discretizaition type is equal_freq.

        Defaults to 10 for each column.

    max_branch : int, optional

        Specifies the maximum number of branches.

        Valid only for CHAID.

        Defaults to 10.

    merge_threshold : float, optional

        Specifies the merge condition for CHAID: if the metric value is \
        greater than or equal to the specified value, the algorithm will \
        merge two branches.

        Only valid for CHAID.

        Defaults to 0.05.

    use_surrogate : bool, optional

        If true, use surrogate split when NULL values are encountered. \
        Only valid for CART.

        Defaults to True.

    model_format : {'json', 'pmml'}, optional

        Specifies the tree model format for store. Case-insensitive.

            - 'json': export model in json format.
            - 'pmml': export model in pmml format.

        Defaults to json.

    output_rules : bool, optional

        If true, output decision rules.

        Defaults to True.

    priors : List of tuples: (class, prior_prob), optional

        Specifies the prior probability of every class label.

        Default value detected from data.

    output_confusion_matrix : bool, optional

        If true, output the confusion matrix.

        Defaults to True.

    Attributes
    ----------

    model_ : DataFrame

        Trained model content.

    decision_rules_ : DataFrame

        Rules for decision tree to make decisions. \
        Set to None if ``output_rules`` is False.

    confusion_matrix_ : DataFrame

        Confusion matrix used to evaluate the performance of classification \
        algorithms. \
        Set to None if ``output_confusion_matrix`` is False.

    Examples
    --------

    Input dataframe for training:

    >>> df1.head(4).collect()
       OUTLOOK  TEMP  HUMIDITY WINDY        CLASS
    0    Sunny    75      70.0   Yes         Play
    1    Sunny    80      90.0   Yes  Do not Play
    2    Sunny    85      85.0    No  Do not Play
    3    Sunny    72      95.0    No  Do not Play

    Creating DecisionTreeClassifier instance:

    >>> dtc = DecisionTreeClassifier(conn_context=cc, algorithm='c45',
    ...                              min_records_of_parent=2,
    ...                              min_records_of_leaf=1,
    ...                              thread_ratio=0.4, split_threshold=1e-5,
    ...                              model_format='json', output_rules=True)

    Performing fit() on given dataframe:

    >>> dtc.fit(df1, features=['OUTLOOK', 'TEMP', 'HUMIDITY', 'WINDY'],
    ...         label='LABEL')
    >>> dtc.decision_rules_.collect()
       ROW_INDEX                                                  RULES_CONTENT
    0         0                                       (TEMP>=84) => Do not Play
    1         1                         (TEMP<84) && (OUTLOOK=Overcast) => Play
    2         2         (TEMP<84) && (OUTLOOK=Sunny) && (HUMIDITY<82.5) => Play
    3         3 (TEMP<84) && (OUTLOOK=Sunny) && (HUMIDITY>=82.5) => Do not Play
    4         4       (TEMP<84) && (OUTLOOK=Rain) && (WINDY=Yes) => Do not Play
    5         5               (TEMP<84) && (OUTLOOK=Rain) && (WINDY=No) => Play

    Input dataframe for predicting:

    >>> df2.collect()
       ID   OUTLOOK  HUMIDITY  TEMP WINDY
    0   0  Overcast      75.0    70   Yes
    1   1      Rain      78.0    70   Yes
    2   2     Sunny      66.0    70   Yes
    3   3     Sunny      69.0    70   Yes
    4   4      Rain       NaN    70   Yes
    5   5      None      70.0    70   Yes
    6   6       ***      70.0    70   Yes

    Performing predict() on given dataframe:

    >>> result = dtc.predict(df2, key='ID', verbose=False)
    >>> result.collect()
       ID        SCORE  CONFIDENCE
    0   0         Play    1.000000
    1   1  Do not Play    1.000000
    2   2         Play    1.000000
    3   3         Play    1.000000
    4   4  Do not Play    1.000000
    5   5         Play    0.692308
    6   6         Play    0.692308

    Input dataframe for scoring:

    >>> df3.collect()
       ID   OUTLOOK  HUMIDITY  TEMP WINDY        LABEL
    0   0  Overcast      75.0    70   Yes         Play
    1   1      Rain      78.0    70    No  Do not Play
    2   2     Sunny      66.0    70   Yes         Play
    3   3     Sunny      69.0    70   Yes         Play

    Performing score() on given dataframe:

    >>> rfc.score(df3, key='ID')
    0.75
    """
    algorithm_map = {'c45':1, 'chaid':2, 'cart':3}
    #pylint:disable=too-many-arguments, too-many-locals
    def __init__(self,
                 conn_context,
                 algorithm,
                 thread_ratio=None,
                 allow_missing_dependent=True,
                 percentage=None,
                 min_records_of_parent=None,
                 min_records_of_leaf=None,
                 max_depth=None,
                 categorical_variable=None,
                 split_threshold=None,
                 discretization_type=None,
                 bins=None,
                 max_branch=None,
                 merge_threshold=None,
                 use_surrogate=None,
                 model_format=None,
                 output_rules=True,
                 priors=None,
                 output_confusion_matrix=True
                ):
        super(DecisionTreeClassifier, self).__init__(conn_context,
                                                     algorithm,
                                                     thread_ratio,
                                                     allow_missing_dependent,
                                                     percentage,
                                                     min_records_of_parent,
                                                     min_records_of_leaf,
                                                     max_depth,
                                                     categorical_variable,
                                                     split_threshold,
                                                     use_surrogate,
                                                     model_format,
                                                     output_rules)
        self.discretization_type = self._arg('discretization_type',
                                             discretization_type,
                                             self.discretization_type_map)
        self.bins = self._arg('bins', bins, ListOfTuples)
        self.priors = self._arg('priors', priors, ListOfTuples)
        self.output_confusion_matrix = self._arg('output_confusion_matrix',
                                                 output_confusion_matrix, bool)
        self.max_branch = self._arg('max_branch', max_branch, int)
        self.merge_threshold = self._arg('merge_threshold', merge_threshold, float)
        if self.algorithm != 1 and self.algorithm != 2 and discretization_type is not None:
            msg = ("discretization_type is inapplicable, " +
                   "since algorithm is {} instead of c45 or chaid.".format(algorithm))
            logger.error(msg)
            raise ValueError(msg)

        if bins is not None and (discretization_type is None or
                                 discretization_type.lower() != 'equal_freq'):
            msg = ("bins is inapplicable. " +
                   "It is applicable when discretization_type is set to be equal_freq.")
            logger.error(msg)
            raise ValueError(msg)

        if max_branch is not None and self.algorithm != 2:
            msg = ("max_branch is inapplicable. " +
                   "It is applicable when algorithm is set to be chaid.")
            logger.error(msg)
            raise ValueError(msg)

        if merge_threshold is not None and self.algorithm != 2:
            msg = ("merge_threshold is inapplicable. " +
                   "It is applicable when algorithm is set to be chaid.")
            logger.error(msg)
            raise ValueError(msg)

    def fit(self, data, key=None, features=None, label=None, categorical_variable=None):
        """
        Function for building a decision tree classifier.

        Parameters
        ----------

        data : DataFrame

            Training data.

        key : str, optional

            Name of the ID column. If ``key`` is not provided, it is assumed \
            that the input has no ID column.

        features : list of str, optional

            Names of the feature columns. \
            If ``features`` is not provided, it defaults to all non-ID, \
            non-label columns.

        label : str, optional

            Name of the dependent variable.

            Defaults to the last column.

        categorical_variable : str or list of str, optional

            Specifies INTEGER column(s) that should be treated as categorical. \
            Other INTEGER columns will be treated as continuous.
        """
        super(DecisionTreeClassifier, self).fit(data, key, features, label, categorical_variable)
        # pylint: disable=attribute-defined-outside-init
        self.confusion_matrix_ = self._confusion_matrix_ if self.output_confusion_matrix else None

    def score(self, data, key, features=None, label=None):#pylint:disable= too-many-locals
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------

        data : DataFrame

            Data on which to assess model performance.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns. \
            If ``features`` is not provided, it defaults to all non-ID, \
            non-label columns.

        label : str, optional

            Name of the dependent variable.

            Defaults to the last column.

        Returns
        -------

        float

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

        prediction = self.predict(data=data, key=key,
                                  features=features)
        prediction = prediction.select(key, 'SCORE').rename_columns(['ID_P', 'PREDICTION'])

        actual = data.select(key, label).rename_columns(['ID_A', 'ACTUAL'])
        joined = actual.join(prediction, 'ID_P=ID_A').select('ACTUAL', 'PREDICTION')

        return metrics.accuracy_score(self.conn_context, joined,
                                      label_true='ACTUAL',
                                      label_pred='PREDICTION')


class DecisionTreeRegressor(_DecisionTreeBase):
    """
    Decision Tree model for regression.

    Parameters
    ----------

    conn_context : ConnectionContext

        Database connection object.

    algorithm : {'cart'}

        Algorithm used to grow a decision tree.

            - 'cart': Classification and Regression tree.

        Currently supports cart.

    thread_ratio : float, optional

        Controls the proportion of available threads to use.
        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads. Values between 0 and
        1 will use that percentage of available threads. Values outside this
        range tell PAL to heuristically determine the number of threads to use.

        Defaults to 0.

    allow_missing_dependent : bool, optional

        Specifies if a missing target value is allowed.

            - False: Not allowed. An error occurs if a missing target is present.
            - True: Allowed. The datum with the missing target is removed.

        Defaults to True.

    percentage : float, optional

        Specifies the percentage of the input data that will be used to build
        the tree model.
        The rest of the data will be used for pruning.

        Defaults to 1.0.

    min_records_of_parent : int, optional

        Specifies the stop condition: if the number of records in one node
        is less than the specified value, the algorithm stops splitting.

        Defaults to 2.

    min_records_of_leaf : int, optional

        Promises the minimum number of records in a leaf.

        Defaults to 1.

    max_depth : int, optional

        The maximum depth of a tree.

        By default it is unlimited.

    categorical_variable : str or list of str, optional

        Specifies INTEGER column(s) that should be treated as categorical.
        The default behavior is:
        string: categorical, or integer and float: continuous.

        VALID only for integer variables; omitted otherwise.

        Default value detected from input data.

    split_threshold : float, optional

        Specifies the stop condition for a node:

            - CART: The reduction of Gini index or relative MSE of the best
              split is less than this value.

        The smaller the SPLIT_THRESHOLD value is, the larger a CART tree grows.

        Defaults to 1e-5 for CART.

    use_surrogate : bool, optional

        If true, use surrogate split when NULL values are encountered.
        Only valid for cart.

        Defaults to True.

    model_format : {'json', 'pmml'}, optional

        Specifies the tree model format for store. Case-insensitive.

            - 'json': export model in json format.
            - 'pmml': export model in pmml format.

        Defaults to json.

    output_rules : bool, optional

        If true, output decision rules.

        Defaults to True.

    Attributes
    ----------

    model_ : DataFrame

        Trained model content.

    decision_rules_ : DataFrame

        Rules for decision tree to make decisions.
        Set to None if ``output_rules`` is False.

    Examples
    --------

    Input dataframe for training:

    >>> df1.head(5).collect()
       ID         A         B         C         D      CLASS
    0   0  1.764052  0.400157  0.978738  2.240893  49.822907
    1   1  1.867558 -0.977278  0.950088 -0.151357   4.877286
    2   2 -0.103219  0.410598  0.144044  1.454274  11.914875
    3   3  0.761038  0.121675  0.443863  0.333674  19.753078
    4   4  1.494079 -0.205158  0.313068 -0.854096  23.607000

    Creating DecisionTreeRegressor instance:

    >>>  dtr = DecisionTreeRegressor(conn_context=cc, algorithm='cart',
    ...                              min_records_of_parent=2, min_records_of_leaf=1,
    ...                              thread_ratio=0.4, split_threshold=1e-5,
    ...                              model_format='pmml', output_rules=True)

    Performing fit() on given dataframe:

    >>> dtr.fit(df1, key='ID')
    >>> dtr.decision_rules_.head(2).collect()
       ROW_INDEX                                      RULES_CONTENT
    0          0         (A<-0.495502) && (B<-0.663588) => -85.8762
    1          1        (A<-0.495502) && (B>=-0.663588) => -29.9827

    Input dataframe for predicting:

    >>> df2.collect()
       ID         A         B         C         D
    0   0  1.764052  0.400157  0.978738  2.240893
    1   1  1.867558 -0.977278  0.950088 -0.151357
    2   2 -0.103219  0.410598  0.144044  1.454274
    3   3  0.761038  0.121675  0.443863  0.333674
    4   4  1.494079 -0.205158  0.313068 -0.854096

    Performing predict() on given dataframe:

    >>> result = dtr.predict(df2, key='ID')
    >>> result.collect()
       ID    SCORE  CONFIDENCE
    0   0  49.8229         0.0
    1   1  4.87728         0.0
    2   2  11.9148         0.0
    3   3   19.753         0.0
    4   4   23.607         0.0

    Input dataframe for scoring:

    >>> df3.collect()
       ID         A         B         C         D      CLASS
    0   0  1.764052  0.400157  0.978738  2.240893  49.822907
    1   1  1.867558 -0.977278  0.950088 -0.151357   4.877286
    2   2 -0.103219  0.410598  0.144044  1.454274  11.914875
    3   3  0.761038  0.121675  0.443863  0.333674  19.753078
    4   4  1.494079 -0.205158  0.313068 -0.854096  23.607000

    Performing score() on given dataframe:

    >>> dtr.score(df3, key='ID')
    0.9999999999900131
    """
    algorithm_map = {'cart':3}
    def score(self, data, key, features=None, label=None):#pylint:disable=too-many-locals
        """
        Returns the coefficient of determination R^2 of the prediction.

        Parameters
        ----------

        data : DataFrame

            Data on which to assess model performance.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.
            If ``features`` is not provided, it defaults to all non-ID,
            non-label columns.

        label : str, optional

            Name of the dependent variable.

            Defaults to the last column.

        Returns
        -------

        float

            The coefficient of determination R^2 of the prediction on the
            given data.
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

        prediction = self.predict(data, key, features).select([key, 'SCORE'])
        prediction = prediction.rename_columns(['ID_P', 'PREDICTION'])
        original = data[[key, label]].rename_columns(['ID_A', 'ACTUAL'])
        joined = original.join(prediction, 'ID_P=ID_A').select('ACTUAL', 'PREDICTION')
        return metrics.r2_score(self.conn_context,
                                joined,
                                label_true='ACTUAL',
                                label_pred='PREDICTION')

#Wrapper for GBDT
GBDTALIAS = {'n_estimators':'iter_num',
             'max_tree_depth':'max_depth',
             'loss':'regression_type',
             'split_threshold':'min_split_loss',
             'subsample':'row_sample_rate',
             'lamb':'lambda',
             'learning_rate':'learning_rate',
             'min_sample_weight_leaf':'min_leaf_sample_weight',
             'max_w_in_split':'max_w_in_split',
             'col_subsample_split':'col_sample_rate_split',
             'col_subsample_tree':'col_sample_rate_tree',
             'alpha':'alpha', 'scale_pos_w':'scale_pos_w',
             'base_score':'base_score'}


#pylint:disable=too-few-public-methods, too-many-instance-attributes
class _GradientBoostingBase(PALBase):
    """
    Base Gradient Boosting tree model for classification and regression.
    """
    rangeparm = ('n_estimators', 'max_depth', 'learning_rate',
                 'min_sample_weight_leaf', 'max_w_in_split',
                 'col_subsample_split', 'col_subsample_tree',
                 'lamb', 'alpha', 'scale_pos_w', 'base_score',
                 'split_threshold', 'subsample')

    valid_loss = {'linear':'LINEAR', 'logistic':'LOGISTIC'}
    #pylint:disable=too-many-locals, too-many-arguments
    def __init__(self,
                 conn_context,
                 n_estimators=10,
                 subsample=None,
                 max_depth=None,
                 loss=None,
                 split_threshold=None,
                 learning_rate=None,
                 fold_num=None,
                 default_split_dir=None,
                 min_sample_weight_leaf=None,
                 max_w_in_split=None,
                 col_subsample_split=None,
                 col_subsample_tree=None,
                 lamb=None,
                 alpha=None,
                 scale_pos_w=None,
                 base_score=None,
                 cv_metric=None,
                 ref_metric=None,
                 categorical_variable=None,
                 allow_missing_label=None,
                 thread_ratio=None,
                 cross_validation_range=None
                ):
        super(_GradientBoostingBase, self).__init__(conn_context)
        self.n_estimators = self._arg('n_estimators', n_estimators, int)
        #self.random_state = self._arg('random_state', random_state, int)
        self.subsample = self._arg('subsample', subsample, float)
        self.max_depth = self._arg('max_depth', max_depth, int)
        self.split_threshold = self._arg('split_threshold', split_threshold, float)
        self.loss = self._arg('loss', loss, self.valid_loss)
        self.learning_rate = self._arg('learning_rate', learning_rate, float)
        self.default_split_dir = self._arg('default_split_dir', default_split_dir, int)
        self.fold_num = self._arg('fold_num', fold_num, int)
        self.min_sample_weight_leaf = self._arg('min_sample_weight_leaf',
                                                min_sample_weight_leaf, float)
        #self.min_samples_leaf = self._arg('min_samples_leaf', min_samples_leaf, int)
        self.max_w_in_split = self._arg('max_w_in_split', max_w_in_split, float)
        self.col_subsample_split = self._arg('col_subsample_split', col_subsample_split, float)
        self.col_subsample_tree = self._arg('col_subsample_tree', col_subsample_tree, float)
        self.lamb = self._arg('lamb', lamb, float)
        self.alpha = self._arg('alpha', alpha, float)
        self.scale_pos_w = self._arg('scale_pos_w', scale_pos_w, float)
        self.base_score = self._arg('base_score', base_score, float)
        self.cv_metric = self._arg('cv_metric', cv_metric, str)
        if isinstance(ref_metric, str):
            ref_metric = [ref_metric]
        self.ref_metric = self._arg('ref_metric', ref_metric, ListOfStrings)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        self.categorical_variable = self._arg('categorical_variable',
                                              categorical_variable, ListOfStrings)
        self.allow_missing_label = self._arg('allow_missing_label', allow_missing_label, bool)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.cross_validation_range = self._arg('cross_validation_range',
                                                cross_validation_range, ListOfTuples)
        if self.cross_validation_range is not None:
            if self.cross_validation_range:
                for prm in self.cross_validation_range:
                    if prm[0] not in self.rangeparm:
                        msg = ("Parameter name '{}' not supported ".format(prm[0]) +
                               "for cross-validation.")
                        logger.error(msg)
                        raise ValueError(msg)
        self.label_type = 'unknown'


    def fit(self, data, key=None, features=None, label=None, categorical_variable=None):
        #pylint:disable=too-many-locals, too-many-branches, too-many-statements
        """
        Train the tree-ensemble model on input data:

        Parameters
        ----------

        data : DataFrame

            Training data.

        key : str, optional

            Name of the ID column. If ``key`` is not provideed, it is assumed \
            that the input data has no ID column.

        features : list of str, optional

            Names of the feature columns. \
            If features is not provided, it defaults to all non-ID, \
            non-label columns.

        label : str, optional

            Name of the dependent variable.

            Defaults to the last column.

        categorical_variable : str or list of str, optional

            Specifies INTEGER columns that should be treated as categorical. \
            Other INTEGER columns will be treated as continuous.
        """
        key = self._arg('key', key, str)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        categorical_variable = self._arg('categorical_variable', categorical_variable, ListOfStrings)
        cols = data.columns
        #has_id input is process here
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
        #n_features = len(features)
        #Generate a temp view of the data so that label is always in the final column
        data = data[id_col + features + [label]]
        ##okay, now label is in the final column
        tables = ['MODEL', 'VAR_IMPORTANCE', 'CM', 'STATISTICS', 'CV']
        tables = ['#PAL_GRADIENT_BOOSTING_{}_TBL_{}'.format(name, self.id) for name in tables]
        model_tbl, var_importance_tbl, cm_tbl, stats_tbl, cv_tbl = tables
        param_rows = [
            ('ITER_NUM', self.n_estimators, None, None),
            ('ROW_SAMPLE_RATE', None, self.subsample, None),
            ('MAX_TREE_DEPTH', self.max_depth, None, None),
            ('MIN_SPLIT_LOSS', None, self.split_threshold, None),
            ('REGRESSION_TYPE', None, None, self.loss if self.loss is not None else None),
            ('FOLD_NUM', self.fold_num, None, None),
            ('DEFAULT_SPLIT_DIR', self.default_split_dir, None, None),
            ('LEARNING_RATE', None, self.learning_rate, None),
            ('MIN_LEAF_SAMPLE_WEIGHT', None, self.min_sample_weight_leaf, None),
            ('MAX_W_IN_SPLIT', None, self.max_w_in_split, None),
            ('COL_SAMPLE_RATE_SPLIT', None, self.col_subsample_split, None),
            ('COL_SAMPLE_RATE_TREE', None, self.col_subsample_tree, None),
            ('LAMBDA', None, self.lamb, None),
            ('ALPHA', None, self.alpha, None),
            ('SCALE_POS_W', None, self.scale_pos_w, None),
            ('BASE_SCORE', None, self.base_score, None),
            ('CV_METRIC', None, None,
             self.cv_metric.upper() if self.cv_metric is not None else None),
            #This line is left as intended
            ('ALLOW_MISSING_LABEL', self.allow_missing_label, None, None),
            ('THREAD_RATIO', None, self.thread_ratio, None),
            ('HAS_ID', key is not None, None, None)]
        #If categorical variable exists, do row extension
        if self.categorical_variable is not None:
            param_rows.extend(('CATEGORICAL_VARIABLE', None, None, variable)
                              for variable in self.categorical_variable)
        if categorical_variable is not None:
            param_rows.extend(('CATEGORICAL_VARIABLE', None, None, variable)
                              for variable in categorical_variable)
        if self.ref_metric is not None:
            param_rows.extend(('REF_METRIC', None, None, metric.upper())
                              for metric in self.ref_metric)
        #if cross-validation is triggered, do row extension
        if self.fold_num is not None:
            if self.fold_num > 1 and self.cross_validation_range is not None:
                param_rows.extend(('RANGE_{}'.format(GBDTALIAS[cvparam].upper()),
                                   None,
                                   None,
                                   '{}'.format(range_))
                                  for cvparam, range_ in self.cross_validation_range)

        #model_spec = [
        #    ("ROW_INDEX", INTEGER),
        #    ("KEY", NVARCHAR(1000)),
        #    ("VALUE", NVARCHAR(1000))]

        #var_importance_spec = [
        #    ("VARIABLE_NAME", NVARCHAR(256)),
        #    ("IMPORTANCE", DOUBLE)]
        #cm_spec = [
        #    ("ACTUAL_CLASS", NVARCHAR(1000)),
        #    ("PREDICT_CLASS", NVARCHAR(1000)),
        #    ("COUNT", INTEGER)]
        #stats_spec = [
        #    ("STAT_NAME", NVARCHAR(1000)),
        #    ("STAT_VALUE", NVARCHAR(1000))]
        #cv_spec = [
        #    ("PARM_NAME", NVARCHAR(1000)),
        #    ("INT_VALUE", INTEGER),
        #    ("DOUBLE_VALUE", DOUBLE),
        #    ("STRING_VALUE", NVARCHAR(1000))]
        try:
            #self._materialize(data_tbl, data)
            #self._create(ParameterTable(param_tbl).with_data(param_rows))
            #self._create(Table(model_tbl, model_spec))
            #self._create(Table(var_importance_tbl, var_importance_spec))
            #self._create(Table(cm_tbl, cm_spec))
            #self._create(Table(stats_tbl, stats_spec))
            #self._create(Table(cv_tbl, cv_spec))
            self._call_pal_auto("PAL_GBDT",
                                data,
                                ParameterTable().with_data(param_rows),
                                *tables)
        except dbapi.Error as db_err:
            #msg = ('HANA error while attempting to fit '+
            #       'gradient boosting tree model.')
            logger.exception(str(db_err))
            self._try_drop(tables)
            raise
        except Exception as err:
            msg = str(err)
            logger.exception(msg)
            self._try_drop(tables)
            raise
        conn = self.conn_context
        #pylint: disable=attribute-defined-outside-init
        #self.param_ = conn.table(param_tbl)
        self.model_ = conn.table(model_tbl)
        self.feature_importances_ = conn.table(var_importance_tbl)
        self._confusion_matrix_ = conn.table(cm_tbl)
        self.stats_ = conn.table(stats_tbl)
        if self.cross_validation_range is not None and self.fold_num > 1:
            self.cv_ = conn.table(cv_tbl)
        else:
            self.cv_ = None

    def _predict(self, key, data, features=None, verbose=None):
        #pylint:disable=too-many-locals, too-many-branches, too-many-statements
        """
        Predict dependent variable values based on fitted moel.

        Parameters
        ----------

        data : DataFrame

            Independent variable values to predict for.

        key : str
            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.
            If features is not provided, it defaults to all non-ID columns.

        verbose : bool, optional

            If true, output all classes and the corresponding confindences
            for each data point. Only valid classification.

            Default to False if not provided.

        Returns
        -------

        DataFrame

            DataFrame of score and confidence, structured as follows:
            1st column - ID column, with same as and type as ``data``'s
                         ID column.
            2nd column - SCORE, type NVARCHAR(1000), representing the predicted
                         classes/values.
            3rd column - CONFIDENCE, type DOUBLE, representing the confidence of
                         of a class. All Zero's for regression.
        """
        if getattr(self, 'model_', None) is None:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")

        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        verbose = self._arg('verbose', verbose, bool)
        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols
        data = data[[key] + features]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        #tables = data_tbl, model_tbl, param_tbl, result_tbl = [
        result_tbl = '#PAL_GRADIENT_BOOSTING_RESULT_TBL_{}_{}'.format(self.id, unique_id)
            #for name in ['DATA', 'MODEL', 'PARAM', 'FITTED']]
        param_rows = [
            ('THREAD_RATIO', None, self.thread_ratio, None),
            ('VERBOSE', verbose, None, None)]
        #result_spec = [
        #    (parse_one_dtype(data.dtypes([data.columns[0]])[0])),
        #    ("SCORE", NVARCHAR(100)),
        #    ("CONFIDENCE", DOUBLE)]

        try:
            #self._materialize(data_tbl, data)
            #self._materialize(model_tbl, self.model_)
            #self._create(ParameterTable(param_tbl).with_data(param_rows))
            #self._create(Table(result_tbl, result_spec))
            self._call_pal_auto("PAL_GBDT_PREDICT",
                                data,
                                self.model_,
                                ParameterTable().with_data(param_rows),
                                result_tbl)
        except dbapi.Error as db_err:
            #msg = 'HANA error during gradient boosting prediction.'
            logger.exception(str(db_err))
            self._try_drop(result_tbl)
            raise
	#except Exception as err:
	#    msg = str(err)
	#    logger.exception(msg)
        #    self._try_drop(tables)
        return self.conn_context.table(result_tbl)


class GradientBoostingClassifier(_GradientBoostingBase):
#pylint: disable=too-many-instance-attributes, line-too-long
    """
    Gradient Boosting model for classification.

    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA system.

    n_estimators : int, optional

        Specifies the number of trees in Gradient Boosting.

        Defaults to 10.

    loss : str, optional

        Type of loss function to be optimized. \
        Supported values are 'linear' and 'logistic'.

        Defaults to 'linear'.

    max_depth : int, optional

        The maximum depth of a tree.

        Defaults to 6.

    split_threshold : float, optional

        Specifies the stopping condition: if the improvement value of the best \
        split is less than this value, then the tree stops growing.

    learning_rate : float, optional.

        Learning rate of each iteration, must be within the range (0, 1].

        Defaults to 0.3.

    subsample : float, optional

        The fraction of samples to be used for fitting each base learner.

        Defaults to 1.0.

    fold_num : int, optional

        The k-value for k-fold cross-validation. \
        Effective only when ``cross_validation_range`` is not None nor empty.

    default_split_dir : int, optional.

        Default split direction for missing values. \
        Valid input values are 0, 1 and 2, where:

            0 - Automatically determined,
            1 - Left,
            2 - Right.

        Defaults to 0.

    min_sample_weight_leaf : float, optional

        The minimum sample weights in leaf node.

        Defaults to 1.0.

    max_w_in_split : float, optional

        The maximum weight constraint assigned to each tree node.

        Defaults to 0 (i.e. no constraint).

    col_subsample_split : float, optional

        The fraction of features used for each split, should be within range (0, 1].

        Defaults to 1.0.

    col_subsample_tree : float, optional

        The fraction of features used for each tree growth, should be within range (0, 1]

        Defaults to 1.0.

    lamb : float, optional

        L2 regularization weight for the target loss function. \
        Should be within range (0, 1].

        Defaults to 1.0.

    alpha : float, optional

        Weight of L1 regularization for the target loss function.

        Defaults to 1.0.

    scale_pos_w : float, optional

        The weight scaled to positive samples in regression.

        Defaults to 1.0.

    base_score : float, optional

        Intial prediction score for all instances. Global bias for sufficient number \
        of iterations(changing this value will not have too much effect).

    cv_metric : { 'rmse', 'mae', 'log_likelihood', 'multi_log_likelihood', \
        'error_rate', 'multi_error_rate', 'auc'}, optional

        The metric used for cross-validation.

        If multiple lines of metrics are provided, then only the first one is valid. \
        If not set, it takes the first value (in alphabetical order) of the parameter \
        'ref_metric' when the latter is set, otherwise it goes to default values.

        Defaults to
            1)'error_rate' for binary classification,

            2)'multi_error_rate' for \
            multi-class classification.

    ref_metric : str or list of str, optional

        Specifies a reference metric or a list of reference metrics. \
        Supported metrics same as cv_metric. \
        If not provided, defaults to

            1)['error_rate'] for binary classification,

            2)['multi_error_rate'] for multi-class classification.

    categorical_variable : str or list of str, optional

        Specifies which variable(s) should be treated as categorical. Otherwise default \
        behavior is followed:

            1) STRING - categorical,

            2) INTEGER and DOUBLE - continous.

        Only valid for INTEGER variables, omitted otherwise.

    allow_missing_label : {'0', '1'},  bool, optional

        Specifies whether missing label value is allowed.

            0: not allowed. In missing values presents in the input data, an error shall be \
            thrown.

            1: allowed. The datum with missing label will be removed automatically.

    thread_ratio : float, optional

        The ratio of available threads used for training:

            0: single thread;

            (0,1]: percentage of available threads;

            others : heuristically determined.

        Defaults to -1.

    cross_validation_range : list of tuples, optional

        Indicates the set of parameters involded for cross-validation. \
        Cross-validation is triggered only when this param is not None and the list \
        is not tempty, and fold_num is greater than 1.\
        Each tuple is a pair, with the first being parameter name of str type, and \
        the second being the a list of number of the following form: \
        [<begin-value>, <test-numbers>, <end-value>].

        Suppported parameters for cross-validation: ``n_estimators``, ``max_depth``, ``learning_rate``, \
        ``min_sample_weight_leaf``, ``max_w_in_split``, ``col_subsample_split``, ``col_subsample_tree``, \
        ``lamb``, ``alpha``, ``scale_pos_w``, ``base_score``.

        A simple example for illustration:

            [('n_estimators', [4, 3, 10]),

            ('learning_rate', [0.1, 3, 1.0]),

            ('split_threshold', [0.1, 3, 1.0])]


    Attributes
    ----------

    model_ : DataFrame

        Trained model content.

    feature_importances_ : DataFrame

        The feature importance (the higher, the more import the feature)

    confusion_matrix_ : DataFrame

        Confusion matrix used to evaluate the performance of classification algorithm.

    stats_ : DataFrame

        Statistics info for cross-validation.

    cv_ : DataFrame

        Best choice of parameter produced by cross-validation.

    Examples
    --------

    Input dataframe for training:

    >>> df.head(4).collect()
       ATT1  ATT2   ATT3  ATT4 LABEL
    0   1.0  10.0  100.0   1.0     A
    1   1.1  10.1  100.0   1.0     A
    2   1.2  10.2  100.0   1.0     A
    3   1.3  10.4  100.0   1.0     A

    Creating Gradient Boosting Classifier

    >>> gbc = GradientBoostingClassifier(conn_context = conn,
    ...     n_estimators = 4, split_threshold=0,
    ...     learning_rate=0.5, fold_num=5, max_depth=6,
    ...     cv_metric = 'error_rate', ref_metric=['auc'],
    ...     cross_validation_range=[('learning_rate',[0.1,1.0,3]), ('n_estimators', [4,10,3]), ('split_threshold', [0.1,1.0,3])])

    Performing fit() on given dataframe:

    >>> gbc.fit(df, features=['ATT1', 'ATT2', 'ATT3', 'ATT4'], label='LABEL')
    >>> gbc.stats_.collect()
             STAT_NAME STAT_VALUE
    0  ERROR_RATE_MEAN          0
    1   ERROR_RATE_VAR          0
    2         AUC_MEAN          1

    Input dataframe for predicting:

    >>> df1.head(4).collect()
       ID  ATT1  ATT2   ATT3  ATT4
    0   1   1.0  10.0  100.0   1.0
    1   2   1.1  10.1  100.0   1.0
    2   3   1.2  10.2  100.0   1.0
    3   4   1.3  10.4  100.0   1.0

    Performing predict() on given dataframe

    >>> result = gbc.fit(df1, key='ID', verbose=False)
    >>> result.head(4).collect()
       ID SCORE  CONFIDENCE
    0   1     A    0.825556
    1   2     A    0.825556
    2   3     A    0.825556
    3   4     A    0.825556
    """

    def fit(self, data, key=None, features=None, label=None, categorical_variable=None):
        """
        Train the model on input data.

        Parameters
        ----------


        data : DataFrame
            Training data.

        key : str, optional

            Name of the ID column. If ``key`` is not provided, it is assumed \
            that the input has no ID column.

        features : list of str, optional

            Names of the feature columns. \
            If ``features`` is not provided, it defaults to all non-ID, \
            non-label columns.

        label : str, optional

            Name of the dependent variable.

            Defaults to the last column.

        categorical_variable : str or list of str, optional

            Specifies INTEGER column(s) that should be treated as categorical. \
            Other INTEGER columns will be treated as continuous.
        """
        if label is None:
            label = data.columns[-1]
        self.label_type = data.dtypes([str(label)])[0][1]
        if self.categorical_variable is None:
            categorical_variable = []
        else:
            categorical_variable = self.categorical_variable
        if self.label_type not in ('VARCHAR', 'NVARCHAR'):
            if (label not in categorical_variable or self.label_type != 'INT'):
                msg = ("Label column data type \'{}\' is ".format(self.label_type) +
                       "not supported for classification.")
                logger.error(msg)
                raise ValueError(msg)

        super(GradientBoostingClassifier, self).fit(data, key, features, label, categorical_variable)
        #pylint: disable=attribute-defined-outside-init
        self.confusion_matrix_ = self._confusion_matrix_


    def predict(self, data, key, features=None, verbose=None):#pylint:disable=too-many-locals
        """
        Predict dependent variable values based on fitted model.

        Parameters
        ----------

        data : DataFrame

            Independent variable values to predict for.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns. \
            If ``features`` is not provided, it defaults to all non-ID columns.

        verbose : bool, optional

            If true, output all classes and the corresponding confidences \
            for each data point.

        Returns
        -------

        DataFrame

            DataFrame of score and confidence, structured as follows:

                - ID column, with same name and type as ``data`` 's ID column.
                - SCORE, type NVARCHAR, representing the predicted classes.
                - CONFIDENCE, type DOUBLE, representing the confidence of \
                  a class.
        """
        return super(GradientBoostingClassifier, self)._predict(data=data,
                                                                key=key,
                                                                features=features,
                                                                verbose=verbose)


class GradientBoostingRegressor(_GradientBoostingBase):
#pylint: disable=too-many-instance-attributes, line-too-long
    """
    Gradient Boosting Tree model for regression.

    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA system.

    n_estimators : int, optional

        Specifies the number of trees in Gradient Boosting.

        Defaults to 10.

    loss : str, optional

        Type of loss function to be optimized. Supported values are 'linear' and 'logistic'.

        Defaults to 'linear'.

    max_depth : int, optional

        The maximum depth of a tree.

        Defaults to 6.

    split_threshold : float, optional

        Specifies the stopping condition: if the improvement value of the best split is less than this value, then the tree stops growing.

    learning_rate : float, optional.

        Learning rate of each iteration, must be within the range (0, 1].

        Defaults to 0.3.

    subsample : float, optional

        The fraction of samples to be used for fitting each base learner.

        Defaults to 1.0.

    fold_num : int, optional

        The k-value for k-fold cross-validation.

    default_split_dir : int, optional.

        Default split direction for missing values. \
        Valid input values are 0, 1 and 2, where:

            0 - Automatically determined,

            1 - Left,

            2 - Right.

        Defaults to 0.

    min_sample_weight_leaf : float, optional

        The minimum sample weights in leaf node.

        Defaults to 1.0.

    max_w_in_split : float, optional

        The maximum weight constraint assigned to each tree node.

        Defaults to 0 (i.e. no constraint).

    col_subsample_split : float, optional

        The fraction of features used for each split, should be within range (0, 1].

        Defaults to 1.0.

    col_subsample_tree : float, optional

        The fraction of features used for each tree growth, should be within range (0, 1]

        Defaults to 1.0.

    lamb : float, optional

        L2 regularization weight for the target loss function. \
        Should be within range (0, 1].

        Defaults to 1.0.

    alpha : float, optional

        Weight of L1 regularization for the target loss function.

        Defaults to 1.0.

    scale_pos_w : float, optional

        The weight scaled to positive samples in regression.

        Defaults to 1.0.

    base_score : float, optional

        Intial prediction score for all instances. Global bias for sufficient number \
        of iterations(changing this value will not have too much effect).

    cv_metric : str, optional

        The metric used for cross-validation. \
        Supported metrics include: 'rmse', 'mae', 'log_likelihood', 'multi_log_likelihood', \
        'error_rate', 'multi_error_rate' and 'auc'. \
        If multiple lines of metrics are provided, then only the first one is valid. \
        If not set, it takes the first value (in alphabetical order) of the parameter \
        'ref_metric' when the latter is set, otherwise it goes to default values.

        Defaults to

            1)'error_rate' for binary classification,

            2)'multi_error_rate' for multi-class classification.

    ref_metric : str or list of str, optional

        Specifies a reference metric or a list of reference metrics. \
        Supported metrics same as cv_metric. \
        If not provided, defaults to

            1)['error_rate'] for binary classification,

            2)['multi_error_rate'] for multi-class classification.

    categorical_variable : str, optional

        Indicates which variables should be treated as categorical. Otherwise default behavior is followed:

            1) STRING - categorical,

            2) INTEGER and DOUBLE - continous.

        Only valid for INTEGER variables, omitted otherwise.

    allow_missing_label : {'0', '1'}, bool, optional

        Specifies whether missing label value is allowed.

            0: not allowed. In missing values presents in the input data, an error shall be thrown.

            1: allowed. The datum with missing label will be removed automatically.

    thread_ratio : float, optional

        The ratio of available threads used for training.

            0: single thread;

            (0,1]: percentage of available threads;

            others : heuristically determined.

        Defaults to -1.

    cross_validation_range : list of tuples, optional

        Indicates the set of parameters involded for cross-validation. \
        Each tuple is a pair, with the first being parameter name of str type, and \
        the second being the a list of number of the following form: \
            [<begin-value>, <test-numbers>, <end-value>].
        Suppported parameters for cross-validation: ``n_estimators``, ``max_depth``, ``learning_rate``, \
        ``min_sample_weight_leaf``, ``max_w_in_split``, ``col_subsample_split``, ``col_subsample_tree``, \
        ``lamb``, ``alpha``, ``scale_pos_w``, ``base_score``.

        A simple example for illustration:

            [('n_estimators', [4, 3, 10]),

            ('learning_rate', [0.1, 3, 1.0]),

            ('split_threshold', [0.1, 3, 1.0])]


    Attributes
    ----------

    model_ : DataFrame

        Trained model content.

    feature_importances_ : DataFrame

        The feature importance (the higher, the more import the feature)

    stats_ : DataFrame

        Statistics info for cross-validation.

    cv_ : DataFrame

        Best choice of parameter produced by cross-validation.

    Examples
    --------

    Input dataframe for training:

    >>> df.head(4).collect()
        ATT1     ATT2    ATT3    ATT4  TARGET
    0  19.76   6235.0  100.00  100.00   25.10
    1  17.85  46230.0   43.67   84.53   19.23
    2  19.96   7360.0   65.51   81.57   21.42
    3  16.80  28715.0   45.16   93.33   18.11

    Creating GradientBoostingRegressor instance:

    >>>  gbr = GradientBoostingRegressor(conn_context = conn,
    ...     n_estimators = 20, split_threshold=0.75,
    ...     learning_rate=0.75, fold_num=5, max_depth=6,
    ...     cv_metric = 'rmse', ref_metric=['mae'],
    ...     cross_validation_range=[('learning_rate',[0.0,5,1.0]), ('n_estimators', [10, 11, 20]), ('split_threshold', [0.0, 5, 1.0])])

    Performing fit() on given dataframe:

    >>> gbr.fit(df, features=['ATT1', 'ATT2', 'ATT3', 'ATT4'], label='TARGET')
    >>> gbr.stats_.collect()
       STAT_NAME STAT_VALUE
    0  RMSE_MEAN    1.83732
    1   RMSE_VAR   0.525622
    2   MAE_MEAN    1.44388

    Input dataframe for predicting:

    >>> df1.head(4).collect()
       ID   ATT1     ATT2    ATT3    ATT4
    0   1  19.76   6235.0  100.00  100.00
    1   2  17.85  46230.0   43.67   84.53
    2   3  19.96   7360.0   65.51   81.57
    3   4  16.80  28715.0   45.16   93.33

    Performing predict() on given dataframe:

    >>> result.head(4).collect()
       ID    SCORE CONFIDENCE
    0   1  24.1499       None
    1   2  19.2351       None
    2   3  21.8944       None
    3   4  18.5256       None
    """


    def fit(self, data, key=None, features=None, label=None, categorical_variable=None):
        """
        Train the model on input data.

        Parameters
        ----------

        data : DataFrame

            Training data.

        key : str, optional

            Name of the ID column. If ``key`` is not provided, it is assumed that the input has no ID column.

        features : list of str, optional

            Names of the feature columns. \
            If ``features`` is not provided, it defaults to all non-ID, non-label columns.

        label : str, optional

            Name of the dependent variable.

            Defaults to the last column.

        categorical_variable : str or list of str, optional

            Specifies INTEGER column(s) that should be treated as categorical. \
            Other INTEGER columns will be treated as continuous.
        """
        if label is None:
            label = data.columns[-1]
        self.label_type = data.dtypes([label])[0][1]
        if self.categorical_variable is None:
            categorical_variable = []
        else:
            categorical_variable = self.categorical_variable
        if self.label_type in ('VARCHAR', 'NVARCHAR') \
            or (self.label_type == 'INT' and label in categorical_variable):
            msg = "Label column is treated as categorical, not supported for regression."
            logger.error(msg)
            raise ValueError(msg)
        super(GradientBoostingRegressor, self).fit(data, key=key, features=features, label=label, categorical_variable=categorical_variable)

    def predict(self, data, key, features=None, verbose=None):
        """
        Predict dependent variable values based on fitted model.

        Parameters
        ----------

        data : DataFrame

            Independent variable values to predict for.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.

            If ``features`` is not provided, it defaults to all non-ID columns.

        verbose : bool, optional

            If true, output all classes and the corresponding confidences for each data point.

        Returns
        -------

        DataFrame

            DataFrame of score and confidence, structured as follows:

                - ID column, with same name and type as ``data``'s ID column.
                - SCORE, type DOUBLE, representing the predicted value.
                - CONFIDENCE, all None's for regression.
        """
        return super(GradientBoostingRegressor, self)._predict(data=data,
                                                               key=key,
                                                               features=features,
                                                               verbose=verbose)

#pylint:disable=too-few-public-methods, too-many-instance-attributes
class _HybridGradientBoostingBase(PALBase):
    """
    Hybrid Gradient Boosting Tree model for classification and regression.
    """
    #pylint:disable=too-many-arguments, too-many-locals
    rangeparm = ('n_estimators', 'max_depth', 'learning_rate',
                 'min_sample_weight_leaf', 'max_w_in_split',
                 'col_subsample_split', 'col_subsample_tree',
                 'lamb', 'alpha', 'scale_pos_w', 'base_score',
                 'split_threshold', 'subsample')

    hgbt_name_map = {
        'n_estimators':'ITER_NUM', 'random_state':'SEED',
        'subsample':'ROW_SAMPLE_RATE', 'max_depth':'MAX_DEPTH',
        'split_threshold':'GAMMA', 'learning_rate':'ETA',
        'sketch_eps':'SKETCH_EPS', 'fold_num':'FOLD_NUM',
        'min_sample_weight_leaf':'MIN_CHILD_HESSIAN',
        'min_samples_leaf':'NODE_SIZE',
        'max_w_in_split':'NODE_WEIGHT_CONSTRAINT',
        'col_subsample_split':'COL_SAMPLE_RATE_BYSPLIT',
        'col_subsample_tree':'COL_SAMPLE_RATE_BYTREE',
        'lamb':'LAMBDA', 'alpha':'ALPHA',
        'base_score':'BASE_SCORE',
        'cv_metric':'EVALUATION_METRIC',
        'ref_metric':'REF_METRIC',
        'calculate_importance':'CALCULATE_IMPORTANCE',
        'calculate_cm':'CALCULATE_CONFUSION_MATRIX'}

    split_method_map = {'exact':'exact', 'sketch':'sketch', 'sampling':'sampling'}
    missing_replace_map = {'feature_marginalized':1, 'instance_marginalized':2}

    def __init__(self,
                 conn_context,
                 n_estimators=10,
                 random_state=None,
                 subsample=None,
                 max_depth=None,
                 split_threshold=None,
                 learning_rate=None,
                 split_method=None,
                 sketch_eps=None,
                 fold_num=None,
                 min_sample_weight_leaf=None,
                 min_samples_leaf=None,
                 max_w_in_split=None,
                 col_subsample_split=None,
                 col_subsample_tree=None,
                 lamb=None,
                 alpha=None,
                 #base_score=None,
                 cv_metric=None,
                 ref_metric=None,
                 calculate_importance=None,
                 thread_ratio=None,
                 cross_validation_range=None
                ):
        super(_HybridGradientBoostingBase, self).__init__(conn_context)
        self.n_estimators = self._arg('n_estimators', n_estimators, int)
        self.random_state = self._arg('random_state', random_state, int)
        self.subsample = self._arg('subsample', subsample, float)
        self.max_depth = self._arg('max_depth', max_depth, int)
        self.split_threshold = self._arg('split_threshold', split_threshold, float)
        #self.loss = self._arg('loss', loss, str)
        self.learning_rate = self._arg('learning_rate', learning_rate, float)
        self.split_method = self._arg('split_method', split_method,
                                      self.split_method_map)
        self.sketch_eps = self._arg('sketch_eps', sketch_eps, float)
        self.fold_num = self._arg('fold_num', fold_num, int)
        self.min_sample_weight_leaf = self._arg('min_sample_weight_leaf',
                                                min_sample_weight_leaf, float)
        self.min_samples_leaf = self._arg('min_samples_leaf', min_samples_leaf, int)
        self.max_w_in_split = self._arg('max_w_in_split', max_w_in_split, float)
        self.col_subsample_split = self._arg('col_subsample_split',
                                             col_subsample_split, float)
        self.col_subsample_tree = self._arg('col_subsample_tree',
                                            col_subsample_tree, float)
        self.lamb = self._arg('lamb', lamb, float)
        self.alpha = self._arg('alpha', alpha, float)
        #self.scale_pos_w = self._arg('scale_pos_w', scale_pos_w, float)
        #self.base_score = self._arg('base_score', base_score, float)
        self.cv_metric = self._arg('cv_metric', cv_metric, str)
        if isinstance(ref_metric, str):
            ref_metric = [ref_metric]
        self.ref_metric = self._arg('ref_metric', ref_metric, ListOfStrings)
        #self.categorical_variable = self._arg('categorical_variable',
        #                                      categorical_variable, ListOfStrings)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.confusion_matrix_ = None
        self.cross_validation_range = self._arg('cross_validation_range',
                                                cross_validation_range, ListOfTuples)
        self.calculate_importance = self._arg('calculate_importance',
                                              calculate_importance, bool)
        if self.cross_validation_range is not None:
            if self.cross_validation_range:
                for prm in self.cross_validation_range:
                    if prm[0] not in self.rangeparm:
                        msg = ('Parameter name {} not supported '.format(prm[0]) +
                               'for cross-validation')
                        logger.error(msg)
                        raise ValueError(msg)

        self.label_type = 'unknown'
        self.calculate_cm = None
        self.base_score = None

    def fit(self, data,
            key=None,
            features=None,
            label=None,
            categorical_variable=None):
        #pylint:disable=too-many-locals, too-many-branches, too-many-statements
        """
        Train the tree-ensemble model on input data.
        """
        key = self._arg('key', key, str)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        #pylint:disable=attribute-defined-outside-init
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        self.categorical_variable = self._arg('categorical_variable',
                                              categorical_variable,
                                              ListOfStrings)

        cols = data.columns
        #has_id input is process here
        if key is not None:
            id_col = [key]
            cols.remove(key)
        else:
            id_col = []
        if label is None:
            label = cols[-1]
        cols.remove(label)
        #retrieve data type for  the label column
        #crucial for distinguish between regression and classification problems
        #and related error handling
        if features is None:
            features = cols
        #n_features = len(features)
        #Generate a temp view of the data so that label is always in the final column
        data = data[id_col + features + [label]]
        ##okay, now label is in the final column
        tables = ['PARAM', 'MODEL', 'IMP', 'CONFUSION', 'STATS', 'CV']
        tables = ['#PAL_HGBT_{}_TBL_{}'.format(name, self.id) for name in tables]
        param_tbl, model_tbl, var_importance_tbl, cm_tbl, stats_tbl, cv_tbl = tables
        out_tables = [model_tbl, var_importance_tbl, cm_tbl, stats_tbl, cv_tbl]
        param_rows = [
            ('ITER_NUM', self.n_estimators, None, None),
            ('SEED', self.random_state, None, None),
            ('ROW_SAMPLE_RATE', None, self.subsample, None),
            ('MAX_DEPTH', self.max_depth, None, None),
            ('GAMMA', None, self.split_threshold, None),
            ('FOLD_NUM', self.fold_num, None, None),
            ('ETA', None, self.learning_rate, None),
            ('SPLIT_METHOD', None, None, self.split_method),
            ('MIN_CHILD_HESSIAN', None, self.min_sample_weight_leaf, None),
            ('NODE_SIZE', self.min_samples_leaf, None, None),
            ('NODE_WEIGHT_CONSTRAINT', None, self.max_w_in_split, None),
            ('COL_SAMPLE_RATE_BYSPLIT', None, self.col_subsample_split, None),
            ('COL_SAMPLE_RATE_BYTREE', None, self.col_subsample_tree, None),
            ('LAMBDA', None, self.lamb, None),
            ('ALPHA', None, self.alpha, None),
            ('BASE_SCORE', None, self.base_score, None),
            ('EVALUATION_METRIC', None, None,
             (self.cv_metric).upper() if self.cv_metric is not None else None),
            ('CALCULATE_IMPORTANCE', self.calculate_importance, None, None),
            ('CALCULATE_CONFUSION_MATRIX', self.calculate_cm, None, None),
            ('THREAD_RATIO', None, self.thread_ratio, None),
            ('HAS_ID', key is not None, None, None)]
        #If categorical variable exists,
        #extend param rows to include the claim statement of categorical variables
        if self.categorical_variable is not None:
            param_rows.extend([('CATEGORICAL_VARIABLE', None, None, variable)
                               for variable in self.categorical_variable])
        if self.ref_metric is not None:
            param_rows.extend([('REF_METRIC', None, None, metric.upper())
                               for metric in self.ref_metric])
        #if cross-validation is triggered,
        # extend param rows to include the statement of ranges for cross-validation
        if self.fold_num is not None:
            if self.fold_num > 1 and self.cross_validation_range is not None:
                param_rows.extend(
                    ('RANGE_{}'.format(self.hgbt_name_map[cvparam]),
                     None,
                     None,
                     '{}'.format(range_))
                    for cvparam, range_ in self.cross_validation_range)
        #model_spec = [
        #    ("TREE_INDEX", INTEGER),
        #    ("MODEL_CONTENT", NCLOB)]
        #var_importance_spec = [
        #    ("VARIABLE_NAME", NVARCHAR(256)),
        #    ("IMPORTANCE", DOUBLE)]
        #cm_spec = [
        #    ("ACTUAL_CLASS", NVARCHAR(1000)),
        #    ("PREDICT_CLASS", NVARCHAR(1000)),
        #    ("COUNT", INTEGER)]
        #stats_spec = [
        #    ("STAT_NAME", NVARCHAR(1000)),
        #    ("STAT_VALUE", NVARCHAR(1000))]
        #cv_spec = [
        #    ("PARM_NAME", NVARCHAR(1000)),
        #    ("INT_VALUE", INTEGER),
        #    ("DOUBLE_VALUE", DOUBLE),
        #    ("STRING_VALUE", NVARCHAR(1000))]
        try:
            #self._materialize(data_tbl, data)
            #self._create(ParameterTable(param_tbl).with_data(param_rows))
            #self._create(Table(model_tbl, model_spec))
            #self._create(Table(var_importance_tbl, var_importance_spec))
            #self._create(Table(cm_tbl, cm_spec))
            #self._create(Table(stats_tbl, stats_spec))
            #self._create(Table(cv_tbl, cv_spec))
            #self._create(ParameterTable(param_tbl).with_data(param_rows))
            self._call_pal_auto("PAL_HGBT", data, ParameterTable(param_tbl).with_data(param_rows), *out_tables)
        except dbapi.Error as db_err:
            #msg = ("HANA error while attempting to " +
            #       "fit hybrid gradient boosting tree model.")
            logger.exception(str(db_err))
            self._try_drop(out_tables)
            raise
        conn = self.conn_context
        #pylint: disable=attribute-defined-outside-init
        #self.param_ = conn.table(param_tbl)
        self.model_ = conn.table(model_tbl)
        self.feature_importances_ = conn.table(var_importance_tbl)
        self._confusion_matrix_ = conn.table(cm_tbl)
        self.stats_ = conn.table(stats_tbl)
        if self.cross_validation_range is not None and self.fold_num > 1:
            self.cv_ = conn.table(cv_tbl)
        else:
            self.cv_ = None


    def _predict(self, key, data,
                 features=None,
                 verbose=None,
                 missing_replacement=None,
                 thread_ratio=None):
        #pylint:disable=too-many-locals, too-many-branches, too-many-statements
        if getattr(self, 'model_', None) is None:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        self.thread_ratio = thread_ratio
        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        missing_replacement = self._arg('missing_replacement',
                                        missing_replacement,
                                        self.missing_replace_map)
        verbose = self._arg('verbose', verbose, bool)
        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols
        data = data[[key] + features]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        param_tbl, result_tbl = [
            '#PAL_HGBT_{}_TBL_{}_{}'.format(name, self.id, unique_id)
            for name in ['PARAM', 'RESULT']]
        param_rows = [
            ('THREAD_RATIO', None, self.thread_ratio, None),
            ('VERBOSE', verbose, None, None),
            ('MISSING_REPLACEMENT', missing_replacement, None, None)
            ]

        #result_spec = [
        #    (parse_one_dtype(data.dtypes([data.columns[0]])[0])),
        #    ("SCORE", NVARCHAR(100)),
        #    ("CONFIDENCE", DOUBLE)
        #    ]
        try:
            #self._materialize(data_tbl, data)
            #self._materialize(model_tbl, self.model_)
            #self._create(ParameterTable(param_tbl).with_data(param_rows))
            #self._create(Table(result_tbl, result_spec))
            self._call_pal_auto("PAL_HGBT_PREDICT",
                                data,
                                self.model_,
                                ParameterTable(param_tbl).with_data(param_rows),
                                result_tbl)
        except dbapi.Error as db_err:
            #msg = 'HANA error during hybrid gradient boosting prediction.'
            logger.exception(str(db_err))
            self._try_drop(result_tbl)
            raise
        return self.conn_context.table(result_tbl)

class HybridGradientBoostingClassifier(_HybridGradientBoostingBase):
#pylint: disable=too-many-instance-attributes
    """
    Hybrid Gradient Boosting model for classification.

    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA system.

    n_estimators : int, optional

        Specifies the number of trees in Gradient Boosting.

        Defaults to 10.

    split_method : {'exact', 'sketch', 'sampling'}, optional

        The method to finding split point for numerical features.

        Defaults to 'exact'.

    random_state : int, optional

        The seed for random number generating.

            0 - current time as seed,

            Others - the seed.

    max_depth : int, optional

        The maximum depth of a tree.

        Defaults to 6.

    split_threshold : float, optional

        Specifies the stopping condition: if the improvement value of the best \
        split is less than this value, then the tree stops growing.

    learning_rate : float, optional.

        Learning rate of each iteration, must be within the range (0, 1].

        Defaults to 0.3.

    subsample : float, optional

        The fraction of samples to be used for fitting each base learner.

        Defaults to 1.0.

    fold_num : int, optional

        The k-value for k-fold cross-validation. \
        Effective only when cross_validation_range is not None nor empty.

    sketch_esp : float, optional

        The value of the sketch method which sets up an upper limit for the sum of \
        sample weights between two split points. Basically, the less this value is, the more \
        number of split points are tried.

    min_sample_weight_leaf : float, optional

        The minimum summation of ample weights in a leaf node.

        Defaults to 1.0.

    min_samples_leaf : int, optional

        The minimum number of data in a leaf node.

        Defaults to 1.

    max_w_in_split : float, optional

        The maximum weight constraint assigned to each tree node.

        Defaults to 0 (i.e. no constraint).

    col_subsample_split : float, optional

        The fraction of features used for each split, should be within range (0, 1].

        Defaults to 1.0.

    col_subsample_tree : float, optional

        The fraction of features used for each tree growth, should be within range (0, 1]

        Defaults to 1.0.

    lamb : float, optional

        L2 regularization weight for the target loss function. \
        Should be within range (0, 1].

        Defaults to 1.0.

    alpha : float, optional

        Weight of L1 regularization for the target loss function.

        Defaults to 1.0.

    base_score : float, optional

        Intial prediction score for all instances. Global bias for sufficient number \
        of iterations(changing this value will not have too much effect).

        Defaults to 0.5.

    cv_metric : {'rmse', 'mae', 'nll', 'error_rate', 'auc'}, optional

        The metric used for cross-validation.

        Defaults to 'error_rate'.

    ref_metric : str or list of str, optional

        Specifies a reference metric or a list of reference metrics. \
        Any reference metric must be a valid option of ``cv_metric``.

        Defaults to ['error_rate'].

    categorical_variable : str pr list of str, optional

        Specifies INTEGER variable(s) that should be treated as categorical. \
        Valid only for INTEGER variables, omitted otherwise.

            .. note::

                By default INTEGER variables are treated as numerical.

    thread_ratio : float, optional

        The ratio of available threads used for training.

            0: single thread;

            (0,1]: percentage of available threads;

            others : heuristically determined.

        Defaults to -1.

    calculate_importance : bool, optional

        Determines whether to calculate variable importance.

        Defaults to True.

    calculate_cm : bool, optional

        Determines whether to calculaet confusion matrix.

        Defaults to True.

    cross_validation_range : list of tuples, optional

        Indicates the set of parameters involded for cross-validation. \
        Cross-validation is triggered only when this param is not None and the list \
        is not tempty, and fold_num is greater than 1. \
        Each tuple is a pair, with the first being parameter name of str type, and \
        the second being the a list of numbers with the following strcture: \
        [<begin-value>, <end-value>, <test-numbers>].

        Suppported parameters for cross-validation: ``n_estimators``, ``max_depth``, ``learning_rate``, \
        ``min_sample_weight_leaf``, ``max_w_in_split``, ``col_subsample_split``, ``col_subsample_tree``, \
        ``lamb``, ``alpha``, ``scale_pos_w``, ``base_score``.

        A simple example for illustration:

            [('n_estimators', [4, 10, 3]),

            ('learning_rate', [0.1, 1.0, 3])]


    Attributes
    ----------

    model_ : DataFrame

        Trained model content.

    feature_importances_ : DataFrame

        The feature importance (the higher, the more import the feature)

    confusion_matrix_ : DataFrame

        Confusion matrix used to evaluate the performance of classification algorithm.

    stats_ : DataFrame

        Statistics info for cross-validation.

    cv_ : DataFrame

        Best choice of parameter produced by cross-validation.

    Examples
    --------

    Input dataframe for training:

    >>> df.head(7).collect()
       ATT1  ATT2   ATT3  ATT4 LABEL
    0   1.0  10.0  100.0   1.0     A
    1   1.1  10.1  100.0   1.0     A
    2   1.2  10.2  100.0   1.0     A
    3   1.3  10.4  100.0   1.0     A
    4   1.2  10.3  100.0   1.0     A
    5   4.0  40.0  400.0   4.0     B
    6   4.1  40.1  400.0   4.0     B

    Creating an instance of Hybrid Gradient Boosting classifier:

    >>> ghc = HybridGradientBoostingClassifier(conn_context = conn,
    ...           n_estimators = 4, split_threshold=0,
    ...           learning_rate=0.5, fold_num=5, max_depth=6,
    ...           cv_metric = 'error_rate', ref_metric=['auc'],
    ...           cross_validation_range=[('learning_rate',[0.1, 1.0, 3]),
    ...                                   ('n_estimators', [4, 10, 3]),
    ...                                   ('split_threshold', [0.1, 1.0, 3])])

    Performing fit() on given dataframe

    >>> gbc.fit(df, features=['ATT1', 'ATT2', 'ATT3', 'ATT4'], label='LABEL')
    >>> gbc.stats_.collect()
             STAT_NAME STAT_VALUE
    0  ERROR_RATE_MEAN   0.133333
    1   ERROR_RATE_VAR  0.0266666
    2         AUC_MEAN        0.9

    Input dataframe for predict:

    >>> df_predict.collect()
       ID  ATT1  ATT2   ATT3  ATT4
    0   1   1.0  10.0  100.0   1.0
    1   2   1.1  10.1  100.0   1.0
    2   3   1.2  10.2  100.0   1.0
    3   4   1.3  10.4  100.0   1.0
    4   5   1.2  10.3  100.0   3.0
    5   6   4.0  40.0  400.0   3.0
    6   7   4.1  40.1  400.0   3.0
    7   8   4.2  40.2  400.0   3.0
    8   9   4.3  40.4  400.0   3.0
    9  10   4.2  40.3  400.0   3.0


    Performing predict() on given dataframe

    >>> result = ghc.fit(df_predict, key='ID', verbose=False)
    >>> result.collect()
       ID SCORE  CONFIDENCE
    0   1     A    0.852674
    1   2     A    0.852674
    2   3     A    0.852674
    3   4     A    0.852674
    4   5     A    0.751394
    5   6     B    0.703119
    6   7     B    0.703119
    7   8     B    0.703119
    8   9     B    0.830549
    9  10     B    0.703119
    """
    valid_metric_list = ['rmse', 'mae', 'nll', 'error_rate', 'auc']
    #pylint:disable=too-many-locals
    def __init__(self,
                 conn_context,
                 n_estimators=None,
                 random_state=None,
                 subsample=None,
                 max_depth=None,
                 split_threshold=None,
                 learning_rate=None,
                 split_method=None,
                 sketch_eps=None,
                 fold_num=None,
                 min_sample_weight_leaf=None,
                 min_samples_leaf=None,
                 max_w_in_split=None,
                 col_subsample_split=None,
                 col_subsample_tree=None,
                 lamb=None,
                 alpha=None,
                 base_score=None,
                 cv_metric=None,
                 ref_metric=None,
                 calculate_importance=None,
                 calculate_cm=None,
                 thread_ratio=None,
                 cross_validation_range=None
                ):
        super(HybridGradientBoostingClassifier, self).__init__(conn_context,
                                                               n_estimators,
                                                               random_state,
                                                               subsample,
                                                               max_depth,
                                                               split_threshold,
                                                               learning_rate,
                                                               split_method,
                                                               sketch_eps,
                                                               fold_num,
                                                               min_sample_weight_leaf,
                                                               min_samples_leaf,
                                                               max_w_in_split,
                                                               col_subsample_split,
                                                               col_subsample_tree,
                                                               lamb,
                                                               alpha,
                                                               #base_score,
                                                               cv_metric,
                                                               ref_metric,
                                                               calculate_importance,
                                                               thread_ratio,
                                                               cross_validation_range)
        self.base_score = self._arg('base_score', base_score, float)
        self.calculate_cm = self._arg('calculate_cm', calculate_cm, bool)
        if self.cv_metric is not None:
            if self.cv_metric not in self.valid_metric_list:
                msg = ("'{}' is not a valid evaluation metric for ".format(cv_metric)+
                       "cross-validation in HGBT classification.")
                logger.error(msg)
                raise ValueError(msg)
        if self.ref_metric is not None:
            for metric in self.ref_metric:
                if metric not in self.valid_metric_list:
                    msg = ("'{}' is not a valid reference metric ".format(metric)+
                           "for cross-valiation in HGBT classification.")
                    logger.error(msg)
                    raise ValueError(msg)
    #confusion matrix becomes non-empty when fit finishes
    def fit(self, data,
            key=None,
            features=None,
            label=None,
            categorical_variable=None):
        """
        Train the model on input data.

        Parameters
        ----------

        data : DataFrame

            Training data.

        key : str, optional

            Name of the ID column. If ``key`` is not provided, it is assumed \
            that the input has no ID column.

        features : list of str, optional

            Names of the feature columns. \
            If ``features`` is not provided, it defaults to all non-ID, \
            non-label columns.

        label : str, optional

            Name of the dependent variable.

            Defaults to the last column.

        categorical_variable : str or list of str, optional

            Indicates INTEGER variable(s) that should be treated as categorical. \
            Valid only for INTEGER variables, omitted otherwise.

                .. note::

                    By default INTEGER variables are treated as numerical.
        """
        if label is None:
            label = data.columns[-1]
        self.label_type = data.dtypes([label])[0][1]
        if categorical_variable is None:
            cat_var = []
        else:
            cat_var = categorical_variable
        if self.label_type not in ('VARCHAR', 'NVARCHAR') and \
           (label not in cat_var or self.label_type != 'INT'):
            msg = ("Label column data type {} ".format(self.label_type) +
                   "is not supported for classification.")
            logger.error(msg)
            raise ValueError(msg)
        super(HybridGradientBoostingClassifier, self).fit(data=data,
                                                          key=key,
                                                          features=features,
                                                          label=label,
                                                          categorical_variable=categorical_variable)
        #pylint: disable=attribute-defined-outside-init
        self.confusion_matrix_ = self._confusion_matrix_


    def predict(self,#pylint:disable=too-many-arguments
                data,
                key,
                features=None,
                verbose=None,
                thread_ratio=None,
                missing_replacement=None):#pylint:disable=too-many-locals
        """
        Predict labels based on the trained HGBT classifier.

        Parameters
        ----------

        data : DataFrame

            Independent variable values to predict for.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns. \
            If ``features`` is not provided, it defaults to all non-ID columns.

        missing_replacement : str, optional

            The missing replacement strategy:

                - 'feature_marginalized': marginalise each missing feature out \
                independently.
                - 'instance_marginalized': marginalise all missing features \
                in an instance as a whole corr

        verbose : bool, optional

            If true, output all classes and the corresponding confidences \
            for each data point. \
            This parameter is valid only for classification.

            Defaults to False.

        Returns
        -------

        DataFrame

            DataFrame of score and confidence, structured as follows:

                - ID column, with same name and type as ``data``'s ID column.
                - SCORE, type DOUBLE, representing the predicted classes/values.
                - CONFIDENCE, type DOUBLE, representing the confidence of \
                a class label assignment.
        """
        return super(HybridGradientBoostingClassifier, self)._predict(
            data=data, key=key, features=features, verbose=verbose,
            thread_ratio=thread_ratio, missing_replacement=missing_replacement)


class HybridGradientBoostingRegressor(_HybridGradientBoostingBase):
#pylint: disable=too-many-instance-attributes
    """
    Hybrid Gradient Boosting model for regression.

    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA system.

    n_estimators : int, optional

        Specifies the number of trees in Gradient Boosting.

        Defaults to 10.

    split_method : {'exact', 'sketch', 'sampling'}, optional

        The method to find split point for numeric features.

        Defaults to 'exact'.

    random_state : int, optional

        The seed for random number generating.

            0 - current time as seed,

            Others - the seed.

    max_depth : int, optional

        The maximum depth of a tree.

        Defaults to 6.

    split_threshold : float, optional

        Specifies the stopping condition: if the improvement value of the best \
        split is less than this value, then the tree stops growing.

    learning_rate : float, optional.

        Learning rate of each iteration, must be within the range (0, 1].

        Defaults to 0.3.

    subsample : float, optional

        The fraction of samples to be used for fitting each base learner.

        Defaults to 1.0.

    fold_num : int, optional

        The k-value for k-fold cross-validation. \
        Effective only when ``cross_validation_range`` is not None nor empty.

    sketch_esp : float, optional

        The value of the sketch method which sets up an upper limit for the sum of \
        sample weights between two split points. Basically, the less this value is, the more \
        number of split points are tried.

    min_sample_weight_leaf : float, optional

        The minimum summation of ample weights in a leaf node.

        Defaults to 1.0.

    min_sample_leaf : int, optional

        The minimum number of data in a leaf node.

        Defaults to 1.

    max_w_in_split : float, optional

        The maximum weight constraint assigned to each tree node.

        Defaults to 0 (i.e. no constraint).

    col_subsample_split : float, optional

        The fraction of features used for each split, should be within range (0, 1].

        Defaults to 1.0.

    col_subsample_tree : float, optional

        The fraction of features used for each tree growth, should be within range (0, 1].

        Defaults to 1.0.

    lamb : float, optional

        Weight of L2 regularization for the target loss function. \
        Should be within range (0, 1].

        Defaults to 1.0.

    alpha : float, optional

        Weight of L1 regularization for the target loss function.

        Defaults to 1.0.

    cv_metric : {'rmse', 'mae'}, optional

        The metric used for cross-validation.

        Defaults to 'error_rate'.

    ref_metric : str or list of str, optional

        Specifies a reference metric or a list of reference metrics. \
        Any reference metric must be a valid option of ``cv_metric``.

        Defaults to ['rmse'].

    categorical_variable : str or list of str, optional

        Specifies INTEGER variable(s) that should be treated as categorical. \
        Valid only for INTEGER variables, omitted otherwise.

        .. note::

            By default INTEGER variables are treated as numerical.

    thread_ratio : float, optional

        The ratio of available threads used for training.

            0: single thread;

            (0,1]: percentage of available threads;

        others : heuristically determined.

        Defaults to -1.

    calculate_importance : bool, optional

        Determines whether to calculate variable importance.

        Defaults to True.

    calculate_cm : bool, optional

        Determines whether to calculaet confusion matrix.

        Defaults to True.

    cross_validation_range : list of tuples, optional

        Indicates the set of parameters involded for cross-validation. \
        Cross-validation is triggered only when this param is not None and the list \
        is not tempty, and fold_num is greater than 1. \
        Each tuple is a pair, with the first being parameter name of str type, and \
        the second being the a list of numbers with the following strcture: \
        [<begin-value>, <end-value>, <test-numbers>].

        Suppported parameters for cross-validation: n_estimators, max_depth, learning_rate, \
        min_sample_weight_leaf, max_w_in_split, col_subsample_split, col_subsample_tree,
        lamb, alpha, scale_pos_w, base_score.

        Simple example for illustration

        a list of two tuples
            [('n_estimators', [4, 10, 3]),

            ('learning_rate', [0.1, 1.0, 3])]

    Attributes
    ----------

    model_ : DataFrame

        Trained model content.

    feature_importances_ : DataFrame

        The feature importance (the higher, the more import the feature)

    confusion_matrix_ : DataFrame

        Confusion matrix used to evaluate the performance of classification algorithm.

    stats_ : DataFrame

        Statistics info for cross-validation.

    cv_ : DataFrame

        Best choice of parameter produced by cross-validation.

    Examples
    --------

    Input dataframe for training

    >>> df.head(7).collect()
        ATT1     ATT2    ATT3    ATT4  TARGET
    0  19.76   6235.0  100.00  100.00   25.10
    1  17.85  46230.0   43.67   84.53   19.23
    2  19.96   7360.0   65.51   81.57   21.42
    3  16.80  28715.0   45.16   93.33   18.11
    4  18.20  21934.0   49.20   83.07   19.24
    5  16.71   1337.0   74.84   94.99   19.31
    6  18.81  17881.0   70.66   92.34   20.07

    Creating an instance of HGBT regression and traing the model

    >>> hgr = HybridGradientBoostingRegressor(conn_context = conn,
    ...           n_estimators = 20, split_threshold=0.75,
    ...           split_method = 'exact', learning_rate=0.75,
    ...           fold_num=5, max_depth=6,
    ...           cv_metric = 'rmse', ref_metric=['mae'],
    ...           cross_validation_range=[('learning_rate',[0.0, 1.0, 5]),
    ...                                   ('n_estimators', [10, 20, 11]),
    ...                                   ('split_threshold', [0.0, 1.0, 5])])
    >>> hgr.fit(df, features=['ATT1','ATT2','ATT3', 'ATT4'], label='TARGET')

    Check the model content and feature importances

    >>> hgr.model_.head(4).collect()
       TREE_INDEX   MODEL_CONTENT
    0    -1           {"nclass":1,"param":{"bs":0.0,"obj":"reg:linea...
    1    0            {"height":0,"nnode":1,"nodes":[{"ch":[],"gn":9...
    2    1            {"height":0,"nnode":1,"nodes":[{"ch":[],"gn":5...
    3    2            {"height":0,"nnode":1,"nodes":[{"ch":[],"gn":3...
    >>> hgr.feature_importances_.collect()
      VARIABLE_NAME  IMPORTANCE
    0          ATT1    0.744019
    1          ATT2    0.164429
    2          ATT3    0.078935
    3          ATT4    0.012617

    The trained model can be used for prediction.
    Input data for prediction, i.e. with missing target values.

    >>> df_predict.collect()
       ID   ATT1     ATT2    ATT3    ATT4
    0   1  19.76   6235.0  100.00  100.00
    1   2  17.85  46230.0   43.67   84.53
    2   3  19.96   7360.0   65.51   81.57
    3   4  16.80  28715.0   45.16   93.33
    4   5  18.20  21934.0   49.20   83.07
    5   6  16.71   1337.0   74.84   94.99
    6   7  18.81  17881.0   70.66   92.34
    7   8  20.74   2319.0   63.93   95.08
    8   9  16.56  18040.0   14.45   61.24
    9  10  18.55   1147.0   68.58   97.90

    Predict the target values and view the results

    >>> result = hgr.predict(df_predict, key='ID', verbose=False)
    >>> result.collect()
       ID               SCORE CONFIDENCE
    0   1   23.79109147050638       None
    1   2   19.09572889593064       None
    2   3   21.56501359501561       None
    3   4  18.622664075787082       None
    4   5   19.05159916592106       None
    5   6  18.815530665858763       None
    6   7  19.761714911364443       None
    7   8   23.79109147050638       None
    8   9   17.84416828725911       None
    9  10  19.915574945518465       None
    """
    valid_metric_list = ['rmse', 'mae']
    #pylint:disable=too-many-locals
    def __init__(self,
                 conn_context,
                 n_estimators=None,
                 random_state=None,
                 subsample=None,
                 max_depth=None,
                 split_threshold=None,
                 learning_rate=None,
                 split_method=None,
                 sketch_eps=None,
                 fold_num=None,
                 min_sample_weight_leaf=None,
                 min_samples_leaf=None,
                 max_w_in_split=None,
                 col_subsample_split=None,
                 col_subsample_tree=None,
                 lamb=None,
                 alpha=None,
                 #base_score=None,
                 cv_metric=None,
                 ref_metric=None,
                 calculate_importance=None,
                 thread_ratio=None,
                 cross_validation_range=None
                ):
        super(HybridGradientBoostingRegressor, self).__init__(conn_context,
                                                              n_estimators,
                                                              random_state,
                                                              subsample,
                                                              max_depth,
                                                              split_threshold,
                                                              learning_rate,
                                                              split_method,
                                                              sketch_eps,
                                                              fold_num,
                                                              min_sample_weight_leaf,
                                                              min_samples_leaf,
                                                              max_w_in_split,
                                                              col_subsample_split,
                                                              col_subsample_tree,
                                                              lamb,
                                                              alpha,
                                                              #base_score,
                                                              cv_metric,
                                                              ref_metric,
                                                              calculate_importance,
                                                              thread_ratio,
                                                              cross_validation_range)
        if self.cv_metric is not None:
            if self.cv_metric not in self.valid_metric_list:
                msg = ("'{}' is not a valid evaluation metric for ".format(cv_metric)+
                       "cross-validation in HGBT regression.")
                logger.error(msg)
                raise ValueError(msg)
        if self.ref_metric is not None:
            for metric in self.ref_metric:
                if metric not in self.valid_metric_list:
                    msg = ("'{}' is not a valid reference metric ".format(metric)+
                           "for cross-valiation in HGBT regression.")
                    logger.error(msg)
                    raise ValueError(msg)

    #@override
    def fit(self, data,
            key=None,
            features=None,
            label=None,
            categorical_variable=None):
        """
        Train an HGBT regressor on the input data.

        Parameters
        ----------

        data : DataFrame

            Training data.

        key : str, optional

            Name of the ID column. If ``key`` is not provided, it is assumed \
            that the input has no ID column.

        features : list of str, optional

            Names of the feature columns. \
            If ``features`` is not provided, it defaults to all non-ID, \
            non-label columns.

        label : str, optional

            Name of the dependent variable.

            Defaults to the last column.

        categorical_variable : str or list of str, optional

            Specifies INTEGER variable(s) that should be treated as categorical. \
            Valid only for INTEGER variables, omitted otherwise.

            .. note::

                By default INTEGER variables are treated as numerical.
        """
        if label is None:
            label = data.columns[-1]
        self.label_type = data.dtypes([label])[0][1]
        if categorical_variable is None:
            cat_var = []
        else:
            cat_var = categorical_variable
        if self.label_type in ('VARCHAR', 'NVARCHAR') or \
            (self.label_type == 'INT' and label in cat_var):
            msg = "Label column is treated as categorical, not supported for regression."
            logger.error(msg)
            raise ValueError(msg)
        super(HybridGradientBoostingRegressor, self).fit(data,
                                                         key=key,
                                                         features=features,
                                                         label=label,
                                                         categorical_variable=categorical_variable)

    def predict(self, data, key,#pylint:disable=too-many-arguments
                features=None,
                verbose=None,
                thread_ratio=None,
                missing_replacement=None):
        """
        Predict dependent variable values based on fitted model.

        Parameters
        ----------

        data : DataFrame

            Independent variable values to predict for.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns. \
            If not provided, it defaults to all non-ID columns.

        missing_replacement : str, optional

            The missing replacement strategy:

                - 'feature_marginalized': marginalise each missing feature out \
                independently.
                - 'instance_marginalized': marginalise all missing features \
                in an instance as a whole corresponding to each category.

            Defaults to 'feature_marginalized'.

        verbose : bool, optional

            If true, output all classes and the corresponding confidences \
            for each data point.

        Returns
        -------

        DataFrame

            DataFrame of score and confidence, structured as follows:

                - ID column, with same name and type as ``data`` 's ID column.
                - SCORE, type DOUBLE, representing the predicted classes.
                - CONFIDENCE, type DOUBLE, all None for regression prediction.
        """
        return super(HybridGradientBoostingRegressor, self)._predict(
            data=data, key=key, features=features, verbose=verbose,
            thread_ratio=thread_ratio, missing_replacement=missing_replacement)
