"""
This module contains PAL wrapper for Support Vector Machine algorithms.

The following classes are available:

    * :class:`OneClassSVM`
    * :class:`SVC`
    * :class:`SVR`
    * :class:`SVRanking`
"""
#pylint: disable=too-many-lines, line-too-long, invalid-name, relative-beyond-top-level
import logging
import uuid
import itertools

from hdbcli import dbapi
from hana_ml.ml_exceptions import FitIncompleteError
from .pal_base import (
    PALBase,
    ParameterTable,
    ListOfStrings
)

from . import metrics
logger = logging.getLogger(__name__) #pylint: disable=invalid-name

class _SVMBase(PALBase):#pylint:disable=too-many-instance-attributes, too-few-public-methods
    """Base class for Support Vector Machine algorithms."""
    type_map = {1:'SVC', 2:'SVR', 3:'SVRANKING', 4:'ONECLASSSVM'}
    kernel_map = {'linear':0, 'poly':1, 'rbf':2, 'sigmoid':3}
    scale_info_map = {'no':0, 'standardization':1, 'rescale':2}
    #pylint:disable=too-many-arguments
    def  __init__(self, conn_context, svm_type, c=None, kernel=None,
                  degree=None, gamma=None, coef_lin=None, coef_const=None,
                  probability=None, shrink=None, error_tol=None, evaluation_seed=None,
                  thread_ratio=None, nu=None, scale_info=None, scale_label=None, handle_missing=True,
                  categorical_variable=None, category_weight=None, regression_eps=None):
        #pylint:disable=too-many-locals
        super(_SVMBase, self).__init__(conn_context)
        self.type = svm_type
        self.svm_c = self._arg('c', c, float)
        self.kernel_type = self._arg('kernel', kernel, self.kernel_map)
        self.poly_degree = self._arg('degree', degree, int)
        self.rbf_gamma = self._arg('gamma', gamma, float)
        self.coef_lin = self._arg('coef_lin', coef_lin, float)
        self.coef_const = self._arg('coef_const', coef_const, float)
        self.probability = self._arg('probability', probability, bool)
        self.shrink = self._arg('shrink', shrink, bool)
        self.error_tol = self._arg('error_tol', error_tol, float)
        self.evaluation_seed = self._arg('evaluation_seed', evaluation_seed, int)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.nu = self._arg('nu', nu, float)#pylint: disable=invalid-name
        self.scale_info = self._arg('scale_info', scale_info, self.scale_info_map)
        self.scale_label = self._arg('scale_label', scale_label, bool)
        self.handle_missing = self._arg('handle_missing', handle_missing, bool)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        self.categorical_variable = self._arg('categorical_variable', categorical_variable,
                                              ListOfStrings)
        self.category_weight = self._arg('category_weight', category_weight, float)
        self.regression_eps = self._arg('regression_eps', regression_eps, float)
        self.model_type = _SVMBase.type_map[self.type]

    #pylint:disable=too-many-statements, too-many-locals, too-many-branches
    def _fit(self, data, key=None, features=None, label=None, qid=None, categorical_variable=None):
        key = self._arg('key', key, str)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        qid = self._arg('qid', qid, str, self.type == 3)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        categorical_variable = self._arg('categorical_variable', categorical_variable, ListOfStrings)
        model_type = self.model_type
        cols_left = data.columns
        if key is not None:
            cols_left.remove(key)
        if self.type != 4:
            #all algorithms except OneClassSVM need label
            if label is None:
                label = cols_left[-1]
            cols_left.remove(label)
        if self.type == 3:
            cols_left.remove(qid)
        if features is None:
            features = cols_left
        used_cols = [col for col in itertools.chain([key], features, [qid], [label])
                     if col is not None]
        ##Output warning messages if qid is set when type is not svranking
        if self.type != 3 and qid is not None:
            error_msg = "Qid will only be valid when type is SVRanking."
            logger.error(error_msg)
            raise ValueError(error_msg)
        training_data = data[used_cols]
        #data_tbl = "#PAL_{}_DATA_TBL_{}".format(model_type, self.id)
        ## param table manipulation
        param_array = [('TYPE', self.type, None, None), ('SVM_C', None, self.svm_c, None),
                       ('KERNEL_TYPE', self.kernel_type, None, None),
                       ('POLY_DEGREE', self.poly_degree, None, None),
                       ('RBF_GAMMA', None, self.rbf_gamma, None),
                       ('THREAD_RATIO', None, self.thread_ratio, None),
                       ('NU', None, self.nu, None),
                       ('REGRESSION_EPS', None, self.regression_eps, None),
                       ('COEF_LIN', None, self.coef_lin, None),
                       ('COEF_CONST', None, self.coef_const, None),
                       ('PROBABILITY', self.probability, None, None),
                       ('HAS_ID', key is not None, None, None),
                       ('SHRINK', self.shrink, None, None),
                       ('ERROR_TOL', None, self.error_tol, None),
                       ('EVALUATION_SEED', self.evaluation_seed, None, None),
                       ('SCALE_INFO', self.scale_info, None, None),
                       ('SCALE_LABEL', self.scale_label, None, None),
                       ('HANDLE_MISSING', self.handle_missing, None, None),
                       ('CATEGORY_WEIGHT', None, self.category_weight, None)]
        #for categorical variable
        if self.categorical_variable is not None:
            param_array.extend([('CATEGORICAL_VARIABLE', None, None, variable)
                                for variable in self.categorical_variable])
        if categorical_variable is not None:
            param_array.extend([('CATEGORICAL_VARIABLE', None, None, variable)
                                for variable in categorical_variable])
        #param_tbl = "#{}_PARAMETER_TBL_{}".format(model_type, self.id)
        #result table definition
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        model_tbl = "#PAL_{}_MODEL_TBL_{}_{}".format(model_type, self.id, unique_id)
        #model_tbl_spec = [("ROW_INDEX", INTEGER),
        #                  ("MODEL_CONTENT", NVARCHAR(5000))]
        stat_tbl = "#PAL_{}_STATISTIC_TBL_{}_{}".format(model_type, self.id, unique_id)
        #stat_tbl_spec = [("STAT_NAME", NVARCHAR(256)),
        #                 ("STAT_VALUE", NVARCHAR(1000))]
        ph_tbl = "#PAL_{}_PLACEHOLDER_TBL_{}_{}".format(model_type, self.id, unique_id)
        #ph_tbl_spec = [("PARAM_NAME", NVARCHAR(256)),
        #               ("INT_VALUE", INTEGER),
        #               ("DOUBLE_VALUE", DOUBLE),
        #               ("STRING_VALUE", NVARCHAR(1000))]
        try:
            #self._materialize(data_tbl, training_data)
            #self._create(ParameterTable(param_tbl).with_data(param_array))
            #self._create(Table(model_tbl, model_tbl_spec))
            #self._create(Table(stat_tbl, stat_tbl_spec))
            #self._create(Table(ph_tbl, ph_tbl_spec))
            self._call_pal_auto('PAL_SVM',
                                training_data,
                                ParameterTable().with_data(param_array),
                                model_tbl,
                                stat_tbl,
                                ph_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            tables = [model_tbl, stat_tbl, ph_tbl]
            self._try_drop(tables)
            raise
        self.model_ = self.conn_context.table(model_tbl)#pylint:disable=attribute-defined-outside-init
        self.stat_ = self.conn_context.table(stat_tbl)#pylint:disable=attribute-defined-outside-init

    def _predict(self, data, key, features=None, qid=None, verbose=False):#pylint:disable=too-many-locals
        #check for fit table existence
        if not hasattr(self, 'model_'):
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        key = self._arg('key', key, str, True)
        features = self._arg('features', features, ListOfStrings)
        qid = self._arg('qid', qid, str, self.type == 3)
        verbose = self._arg('verbose', verbose, bool)
        unique_id = str(uuid.uuid1())
        unique_id = unique_id.replace('-', '_').upper()
        cols_left = data.columns
        cols_left.remove(key)
        if self.type == 3:
            cols_left.remove(qid)
        index_col = key
        model_type = self.model_type
        if features is None:
            features = cols_left
        used_cols = [col for col in itertools.chain([index_col], features, [qid])
                     if col is not None]
        predict_set = data[used_cols]
        #data_tbl = "#{}_PREDICT_DATA_TBL_{}_{}".format(model_type, self.id, unique_id)
        #model_tbl = "#{}_PREDICT_MODEL_TBL_{}_{}".format(model_type, self.id, unique_id)
        thread_ratio = 0.0 if self.thread_ratio is None else self.thread_ratio
        #param_tbl = "#{}_PREDICT_CONTROL_TBL_{}_{}".format(model_type,
        #                                                   self.id,
        #                                                   unique_id)
        param_array = [('THREAD_RATIO', None, thread_ratio, None)]
        param_array.append(('VERBOSE_OUTPUT', verbose, None, None))
        # result table
        result_tbl = "#{}_PREDICT_RESULT_TBL_{}_{}".format(model_type, self.id, unique_id)
        #index_name, index_type = parse_one_dtype(data.dtypes([index_col])[0])
        #result_specs = [(index_name, index_type),
        #                ('SCORE', NVARCHAR(100)),
        #                ('PROBABILITY', DOUBLE)]
        try:
            #self._materialize(data_tbl, predict_set)
            #self._materialize(model_tbl, self.model_)
            #self._create(ParameterTable(param_tbl).with_data(param_array))
            #self._create(Table(result_tbl, result_specs))
            self._call_pal_auto("PAL_SVM_PREDICT",
                                predict_set,
                                self.model_,
                                ParameterTable().with_data(param_array),
                                result_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            #tables = [data_tbl, model_tbl, param_tbl, result_tbl]
            self._try_drop(result_tbl)
            raise
        return self.conn_context.table(result_tbl)

class SVC(_SVMBase):
    """
    Support Vector Classification.

    Parameters
    ----------
    conn_context : ConnectionContext
        Connection to the HANA system.
    c : float, optional
        Trade-off between training error and margin.
        Value range > 0.

        Defaults to 100.0.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, optional

        Defaults to rbf.
    degree : int, optional
        Coefficient for the poly kernel type.
        Value range >= 1.

        Defaults to 3.
    gamma : float, optional
        Coefficient for the rbf kernel type.

        Defaults to 1.0/number of features in the dataset.

        Only valid for when ``kernel`` is rbf.
    coef_lin : float, optional
        Coefficient for the poly/sigmoid kernel type.

        Defaults to 0.
    coef_const : float, optional
        Coefficient for the poly/sigmoid kernel type.

        Defaults to 0.
    probability : bool, optional
        If true, output probability during prediction.

        Defaults to False.
    shrink : bool, optional
        If true, use shrink strategy.

        Defaults to True.
    tol : float, optional
        Specifies the error tolerance in the training process.
        Value range > 0.

        Defaults to 0.001.
    evaluation_seed : int, optional
        The random seed in parameter selection.
        Value range >= 0.

        Defaults to 0.
    thread_ratio : float, optional
        Controls the proportion of available threads to use.
        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads. Values between
        0 and 1 will use up to that percentage of available threads. Values
        outside this range tell PAL to heuristically determine the number of
        threads to use.

        Defaults to 0.0.
    scale_info : {'no', 'standardization', 'rescale'}, optional
        Options:

          - 'no' : No scale.
          - 'standardization' : Transforms the data to have zero mean
            and unit variance.
          - 'rescale' : Rescales the range of the features to scale the
            range in [-1,1].

        Defaults to standardization.

    handle_missing : bool, optional
        Whether to handle missing values:
            False: No,

            True: Yes.

        Defaults to True.

    categorical_variable : str or list of str, optional
        Specifies INTEGER column(s) in the data that should be treated categorical.
    category_weight : float, optional
        Represents the weight of category attributes.
        Value range > 0.

        Defaults to 0.707.

    Attributes
    ----------
    model_ : DataFrame
        Model content.

    stat_ : DataFrame
        Statistics content.

    Examples
    --------
    Training data:

    >>> df_fit.head(10).collect()
       ID  ATTRIBUTE1  ATTRIBUTE2  ATTRIBUTE3 ATTRIBUTE4  LABEL
    0   0         1.0        10.0       100.0          A      1
    1   1         1.1        10.1       100.0          A      1
    2   2         1.2        10.2       100.0          A      1
    3   3         1.3        10.4       100.0          A      1
    4   4         1.2        10.3       100.0         AB      1
    5   5         4.0        40.0       400.0         AB      2
    6   6         4.1        40.1       400.0         AB      2
    7   7         4.2        40.2       400.0         AB      2
    8   8         4.3        40.4       400.0         AB      2
    9   9         4.2        40.3       400.0         AB      2

    Create SVC instance and call fit:

    >>> svc = svm.SVC(connection_context, gamma=0.005, handle_missing=False)
    >>> svc.fit(df_fit, 'ID', ['ATTRIBUTE1', 'ATTRIBUTE2',
    ...                        'ATTRIBUTE3', 'ATTRIBUTE4'])
    >>> df_predict = connection_context.table("SVC_PREDICT_DATA_TBL")
    >>> df_predict.collect()
       ID  ATTRIBUTE1  ATTRIBUTE2  ATTRIBUTE3 ATTRIBUTE4
    0   0         1.0        10.0       100.0          A
    1   1         1.2        10.2       100.0          A
    2   2         4.1        40.1       400.0         AB
    3   3         4.2        40.3       400.0         AB
    4   4         9.1        90.1       900.0          A
    5   5         9.2        90.2       900.0          A
    6   6         4.0        40.0       400.0          A

    Call predict:

    >>> res = svc.predict(df_predict, 'ID', ['ATTRIBUTE1', 'ATTRIBUTE2',
    ...                                      'ATTRIBUTE3', 'ATTRIBUTE4'])
    >>> res.collect()
       ID SCORE PROBABILITY
    0   0     1        None
    1   1     1        None
    2   2     2        None
    3   3     2        None
    4   4     3        None
    5   5     3        None
    6   6     2        None
    """
    #pylint:disable=too-many-arguments
    def  __init__(self, conn_context, c=None, kernel='rbf', degree=None,
                  gamma=None, coef_lin=None, coef_const=None, probability=False, shrink=True,
                  tol=None, evaluation_seed=None, thread_ratio=None, scale_info=None, handle_missing=True,
                  categorical_variable=None, category_weight=None):
        #pylint:disable=too-many-locals
        super(SVC, self).__init__(conn_context, 1, c, kernel, degree, gamma, coef_lin,
                                  coef_const, probability, shrink, tol, evaluation_seed,
                                  thread_ratio, None, scale_info, None, handle_missing, categorical_variable,
                                  category_weight, None)

    def fit(self, data, key=None, features=None, label=None, categorical_variable=None):
        """
        Fit the model when given training dataset and other attributes.

        Parameters
        ----------
        data : DataFrame
            DataFrame containing the data.
        key : str
            Name of the ID column. If ``key`` is not provided, it is assumed
            that the input has no ID column.
        features : list of str, optional
            Names of the feature columns.
            If ``features`` is not provided, it defaults to
            all the non-ID, non-label columns.
        label : str, optional
            Name of the label column.
            If ``label`` is not provided, it defaults to the last column.
        categorical_variable : str or list of str, optional
            Specifies INTEGER column(s) that should be treated as categorical.
            Other INTEGER columns will be treated as continuous.
        """
        self._fit(data=data, key=key, features=features, label=label, categorical_variable=categorical_variable)

    def predict(self, data, key, features=None, verbose=False):#pylint:disable=too-many-locals
        """
        Predict the dataset using the trained model.

        Parameters
        ----------
        data : DataFrame
            DataFrame containing the data.
        key : str
            Name of the ID column.
        features : list of str, optional
            Names of the feature columns.
            If ``features`` is not provided, it defaults to all the
            non-ID, non-label columns.
        verbose : bool, optional
            If true, output scoring probabilities for each class.
            It is only applicable when probability is true during instance
            creation.

            Defaults to False.

        Returns
        -------
        DataFrame
            Predict result, structured as follows:
              - ID column, with the same name and type as ``data`` 's ID column.
              - SCORE, type NVARCHAR(100), prediction value.
              - PROBABILITY, type DOUBLE, prediction probability.
                It is NULL when ``probability`` is False during
                instance creation.
        """
        return self._predict(data=data, key=key, features=features, qid=None, verbose=verbose)

    def score(self, data, key, features=None, label=None):
        """
        Returns the accuracy on the given test data and labels.

        Parameters
        ----------
        data : DataFrame
            DataFrame containing the data.
        key : str.
            Name of the ID column.
        features : list of str, optional
            Names of the feature columns.
            If ``features`` is not provided, it defaults to all the non-ID,
            non-label columns.
        label : str, optional
            Name of the label column.
            If ``label`` is not provided, it defaults to the last column.

        Returns
        -------
        accuracy : float
            Scalar accuracy value comparing the predicted result and
            original label.
        """
        if not hasattr(self, 'model_'):
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        key = self._arg('key', key, str, True)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        #return scalar value of accuracy after calling predict
        unique_id = str(uuid.uuid1())
        unique_id = unique_id.replace('-', '_').upper()
        cols = data.columns
        cols.remove(key)
        if label is None:
            label = data.columns[-1]
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

class SVR(_SVMBase):
    """
    Support Vector Regression.

    Parameters
    ----------
    conn_context : ConnectionContext
        Connection to the HANA system.
    c : float, optional
        Trade-off between training error and margin.
        Value range > 0.

        Defaults to 100.0.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, optional

        Defaults to rbf.
    degree : int, optional
        Coefficient for the poly kernel type.
        Value range >= 1.

        Defaults to 3.
    gamma : float, optional
        Coefficient for the rbf kernel type.

        Defaults to 1.0/number of features in the dataset

        Only valid when ``kernel`` is rbf.
    coef_lin : float, optional
        Coefficient for the poly/sigmoid kernel type.

        Defaults to 0.
    coef_const : float, optional
        Coefficient for the poly/sigmoid kernel type.

        Defaults to 0.
    shrink : bool, optional
        If true, use shrink strategy.

        Defaults to True.
    tol : float, optional
        Specifies the error tolerance in the training process.
        Value range > 0.

        Defaults to 0.001.
    evaluation_seed : int, optional
        The random seed in parameter selection.
        Value range >= 0.

        Defaults to 0.
    thread_ratio : float, optional
        Controls the proportion of available threads to use.
        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads. Values between 0 and
        1 will use that percentage of available threads. Values outside this
        range tell PAL to heuristically determine the number of threads to use.

        Defaults to 0.0.
    scale_info : {'no', 'standardization', 'rescale'}, optional
        Options:

          - 'no' : No scale.
          - 'standardization' : Transforms the data to have zero mean
            and unit variance.
          - 'rescale' : Rescales the range of the features to scale the
            range in [-1,1].

        Defaults to standardization.
    scale_label : bool, optional
        If true, standardize the label for SVR.
        It is only applicable when the ``scale_info`` is
        standardization.

        Defaults to True.

    handle_missing : bool, optional
        Whether to handle missing values:
            False: No,

            True: Yes.

        Defaults to True.

    categorical_variable : str or list of str, optional
        Specifies INTEGER column(s) in the data that should be treated as categorical.

    category_weight : float, optional
        Represents the weight of category attributes.
        Value range > 0.

        Defaults to 0.707.
    regression_eps : float, optional
        Epsilon width of tube for regression.

        Defaults to 0.1.

    Attributes
    ----------
    model_ : DataFrame
        Model content.

    stat_ : DataFrame
        Statistics content.

    Examples
    --------
    Training data:

    >>> df_fit.collect()
        ID  ATTRIBUTE1  ATTRIBUTE2  ATTRIBUTE3  ATTRIBUTE4  ATTRIBUTE5       VALUE
    0    0    0.788606    0.787308   -1.301485    1.226053   -0.533385   95.626483
    1    1    0.414869   -0.381038   -0.719309    1.603499    1.557837  162.582000
    2    2    0.236282   -1.118764    0.233341   -0.698410    0.387380  -56.564303
    3    3   -0.087779   -0.462372   -0.038412   -0.552897    1.231209  -32.241614
    4    4   -0.476389    1.836772   -0.292337   -1.364599    1.326768 -143.240878
    5    5    0.523326    0.065154   -1.513822    0.498921   -0.590686   -5.237827
    6    6   -1.425838   -0.900437   -0.672299    0.646424    0.508856  -43.005837
    7    7   -1.601836    0.455530    0.438217   -0.860707   -0.338282 -126.389824
    8    8    0.266698   -0.725057    0.462189    0.868752   -1.542683   46.633594
    9    9   -0.772496   -2.192955    0.822904   -1.125882   -0.946846 -175.356260
    10  10    0.492364   -0.654237   -0.226986   -0.387156   -0.585063  -49.213910
    11  11    0.378409   -1.544976    0.622448   -0.098902    1.437910   34.788276
    12  12    0.317183    0.473067   -1.027916    0.549077    0.013483   32.845141
    13  13    1.340660   -1.082651    0.730509   -0.944931    0.351025   -6.500411
    14  14    0.736456    1.649251    1.334451   -0.530776    0.280830   87.451863

    Create SVR instance and call fit:

    >>> svr = svm.SVR(conn, kernel='linear', scale_info='standardization',
    ...               scale_label=True, handle_missing=False)
    >>> svr.fit(df_fit, 'ID', ['ATTRIBUTE1', 'ATTRIBUTE2', 'ATTRIBUTE3',
    ...                        'ATTRIBUTE4', 'ATTRIBUTE5'])
    """
    #pylint:disable=too-many-arguments
    def  __init__(self, conn_context, c=None, kernel='rbf', degree=None,
                  gamma=None, coef_lin=None, coef_const=None, shrink=True,
                  tol=None, evaluation_seed=None, thread_ratio=None, scale_info=None,
                  scale_label=None, handle_missing=True, categorical_variable=None,
                  category_weight=None, regression_eps=None):
        #pylint:disable=too-many-locals
        super(SVR, self).__init__(conn_context, 2, c, kernel, degree, gamma,
                                  coef_lin, coef_const, False, shrink, tol, evaluation_seed,
                                  thread_ratio, None, scale_info, scale_label,
                                  handle_missing, categorical_variable, category_weight,
                                  regression_eps)

    def fit(self, data, key, features=None, label=None, categorical_variable=None):
        """
        Fit the model when given training dataset and other attributes.

        Parameters
        ----------
        data : DataFrame
            DataFrame containing the data.
        key : str
            Name of the ID column. If ``key`` is not provided, it is assumed
            that the input has no ID column.
        features : list of str, optional
            Names of the feature columns.
            If ``features`` is not provided, it defaults to all the
            non-ID, non-label columns.
        label : str, optional
            Name of the label column.
            If ``label`` is not provided, it defaults to the
            last column.
        categorical_variable : str or list of str, optional
            Specifies INTEGER column(s) that should be be treated as categorical.
            Other INTEGER columns will be treated as continuous.
        """
        self._fit(data=data, key=key, features=features, label=label, categorical_variable=categorical_variable)

    def predict(self, data, key, features=None):#pylint:disable=too-many-locals
        """
        Predict the dataset using the trained model.

        Parameters
        ----------

        data : DataFrame

            DataFrame containing the data.
        key : str

            Name of the ID column.
        features : list of str, optional

            Names of the feature columns.
            If ``features`` is not provided, it defaults to
            all the non-ID columns.

        Returns
        -------

        DataFrame
            Predict result, structured as follows:

              - ID column, with the same name and type as ``data1`` 's ID column.
              - SCORE, type NVARCHAR(100), prediction value.
              - PROBABILITY, type DOUBLE, prediction probability.
                Always NULL. This column is only used for SVC and SVRanking.
        """
        return self._predict(data=data, key=key, features=features, qid=None)

    def score(self, data, key, features=None, label=None):
        """
        Returns the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        data : DataFrame
            DataFrame containing the data.
        key : str
            Name of the ID column.
        features : list of str, optional
            Names of the feature columns.
            If ``features`` is not provided, it defaults to all the non-ID
            and non-label columns.
        label : str, optional
            Name of the label column.
            If ``label`` is not provided, it defaults to the last column.

        Returns
        -------
        accuracy : float
            Returns the coefficient of determination R^2 of the prediction.
        """

        if not hasattr(self, 'model_'):
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        #return scalar value of accuracy after calling predict
        key = self._arg('key', key, str, True)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        unique_id = str(uuid.uuid1())
        unique_id = unique_id.replace('-', '_').upper()
        cols = data.columns
        cols.remove(key)
        if label is None:
            label = data.columns[-1]
        cols.remove(label)
        if features is None:
            features = cols
        prediction = self.predict(data, key, features)
        prediction = prediction.select([key, 'SCORE']).rename_columns(['ID_P', 'PREDICTION'])
        original = data[[key, label]].rename_columns(['ID_A', 'ACTUAL'])
        joined = original.join(prediction, 'ID_P=ID_A').select('ACTUAL', 'PREDICTION')
        return metrics.r2_score(self.conn_context,
                                joined,
                                label_true='ACTUAL',
                                label_pred='PREDICTION')

class SVRanking(_SVMBase):
    r"""
    Support Vector Ranking

    Parameters
    ----------
    conn_context : ConnectionContext
        Connection to the HANA system.
    c : float, optional
        Trade-off between training error and margin.
        Value range > 0.

        Defaults to 100.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, optional
        Defaults to rbf.
    degree : int, optional
        Coefficient for the poly kernel type.
        Value range >= 1.

        Defaults to 3.
    gamma : float, optional
        Coefficient for the rbf kernel type.

        Defaults to to 1.0/number of features in the dataset.

        Only valid when ``kernel`` is rbf.
    coef_lin : float, optional
        Coefficient for the poly/sigmoid kernel type.

        Defaults to 0.
    coef_const : float, optional
        Coefficient for the poly/sigmoid kernel type.

        Defaults to 0.
    probability : bool, optional
        If true, output probability during prediction.

        Defaults to False.
    shrink : bool, optional
        If true, use shrink strategy.

        Defaults to True.
    tol : float, optional
        Specifies the error tolerance in the training process.
        Value range > 0.

        Defaults to 0.001.
    evaluation_seed : int, optional
        The random seed in parameter selection.
        Value range >= 0.

        Defaults to 0.
    thread_ratio : float, optional
        Controls the proportion of available threads to use.
        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads. Values between 0 and 1
        will use that percentage of available threads. Values outside this
        range tell PAL to heuristically determine the number of threads to use.

        Defaults to 0.0.
    scale_info : {'no', 'standardization', 'rescale'}, optional
        Options:

        - 'no' : No scale.
        - 'standardization' : Transforms the data to have zero mean
          and unit variance.
        - 'rescale' : Rescales the range of the features to scale the
          range in [-1,1].

        Defaults to standardization.

    handle_missing : bool, optional
        Whether to handle missing values:
            * False: No,

            * True: Yes.

        Defaults to True.

    categorical_variable : str or list of str, optional
        Specifies INTEGER column(s) in the data that should be treated as categorical.
    category_weight : float, optional
        Represents the weight of category attributes.
        Value range > 0.

        Defaults to 0.707.

    Attributes
    ----------
    model_ : DataFrame
        Model content.

    stat_ : DataFrame
        Statistics content.

    Notes
    -----
    PAL will throw an error if probability=True is provided to the SVRanking
    constructor and verbose=True is not provided to predict(). This is a
    known bug.

    Examples
    --------
    Training data:

    >>> df_fit.head(10).collect()
       ID  ATTRIBUTE1  ATTRIBUTE2  ATTRIBUTE3  ATTRIBUTE4  ATTRIBUTE5    QID  LABEL
    0   0         1.0         1.0         0.0         0.2         0.0  qid:1      3
    1   1         0.0         0.0         1.0         0.1         1.0  qid:1      2
    2   2         0.0         0.0         1.0         0.3         0.0  qid:1      1
    3   3         2.0         1.0         1.0         0.2         0.0  qid:1      4
    4   4         3.0         1.0         1.0         0.4         1.0  qid:1      5
    5   5         4.0         1.0         1.0         0.7         0.0  qid:1      6
    6   6         0.0         0.0         1.0         0.2         0.0  qid:2      1
    7   7         1.0         0.0         1.0         0.4         0.0  qid:2      2
    8   8         0.0         0.0         1.0         0.2         0.0  qid:2      1
    9   9         1.0         1.0         1.0         0.2         0.0  qid:2      3

    Create SVRanking instance and call fit:

    >>> svranking = svm.SVRanking(conn, gamma=0.005)
    >>> features = ['ATTRIBUTE1', 'ATTRIBUTE2', 'ATTRIBUTE3', 'ATTRIBUTE4',
    ...             'ATTRIBUTE5']
    >>> svranking.fit(df_fit, 'ID', 'QID', features, 'LABEL')

    Call predict:

    >>> df_predict = conn.table("DATA_TBL_SVRANKING_PREDICT")
    >>> df_predict.head(10).collect()
       ID  ATTRIBUTE1  ATTRIBUTE2  ATTRIBUTE3  ATTRIBUTE4  ATTRIBUTE5    QID
    0   0         1.0         1.0         0.0         0.2         0.0  qid:1
    1   1         0.0         0.0         1.0         0.1         1.0  qid:1
    2   2         0.0         0.0         1.0         0.3         0.0  qid:1
    3   3         2.0         1.0         1.0         0.2         0.0  qid:1
    4   4         3.0         1.0         1.0         0.4         1.0  qid:1
    5   5         4.0         1.0         1.0         0.7         0.0  qid:1
    6   6         0.0         0.0         1.0         0.2         0.0  qid:4
    7   7         1.0         0.0         1.0         0.4         0.0  qid:4
    8   8         0.0         0.0         1.0         0.2         0.0  qid:4
    9   9         1.0         1.0         1.0         0.2         0.0  qid:4
    >>> svranking.predict(df_predict, key='ID',
    ...                   features=features, qid='QID').head(10).collect()
        ID     SCORE PROBABILITY
    0    0  -9.85138        None
    1    1  -10.8657        None
    2    2  -11.6741        None
    3    3  -9.33985        None
    4    4  -7.88839        None
    5    5   -6.8842        None
    6    6  -11.7081        None
    7    7  -10.8003        None
    8    8  -11.7081        None
    9    9  -10.2583        None
    """
    #pylint:disable=too-many-arguments
    def  __init__(self, conn_context, c=None, kernel='rbf', degree=None,
                  gamma=None, coef_lin=None, coef_const=None, probability=False, shrink=True,
                  tol=None, evaluation_seed=None, thread_ratio=None,
                  scale_info=None, handle_missing=True, categorical_variable=None,
                  category_weight=None):
        #pylint:disable=too-many-locals
        super(SVRanking, self).__init__(conn_context, 3, c, kernel, degree, gamma, coef_lin,
                                        coef_const, probability, shrink, tol, evaluation_seed,
                                        thread_ratio, None, scale_info, None, handle_missing, categorical_variable,
                                        category_weight, None)

    def fit(self, data, key, qid, features=None, label=None, categorical_variable=None):
        """
        Fit the model when given training dataset and other attributes.

        Parameters
        ----------
        data : DataFrame
            DataFrame containing the data.
        key : str
            Name of the ID column. If ``key`` is not provided, it is assumed
            that the input has no ID column.
        qid : str
            Name of the qid column.
        features : list of str, optional
            Names of the feature columns.
            If ``features`` is not provided, it defaults to all
            the non-ID, non-label, non-qid columns.
        label : str, optional
            Name of the label column.
            If ``label`` is not provided, it defaults to the last column.
        categorical_variable : str or list of str, optional
            INTEGER columns specified in this list will be treated as categorical
            data. Other INTEGER columns will be treated as continuous.
        """
        self._fit(data=data, key=key, features=features, label=label, qid=qid, categorical_variable=categorical_variable)

    def predict(self, data, key, qid, features=None, verbose=False):#pylint:disable=too-many-locals
        """
        Predict the dataset using the trained model.

        Parameters
        ----------
        data : DataFrame
            DataFrame containing the data.
        key : str
            Name of the ID column.
        qid : str
            Name of the qid column.
        features : list of str, optional
            Names of the feature columns.
            If ``features`` is not provided, it defaults to all
            the non-ID, non-qid columns.
        verbose : bool, optional
            If true, output scoring probabilities for each class.

            Defaults to False.

        Returns
        -------
        DataFrame
            Predict result, structured as follows:
                - ID column, with the same name and type as ``data``'s ID column.
                - Score, type NVARCHAR(100), prediction value.
                - PROBABILITY, type DOUBLE, prediction probability.
                  It is NULL when ``probability`` is False during
                  instance creation.
        """
        return self._predict(data=data, key=key, features=features, verbose=verbose, qid=qid)

class OneClassSVM(_SVMBase):
    """
    One Class SVM

    Parameters
    ----------
    conn_context : ConnectionContext
        Connection to the HANA system.
    c : float, optional
        Trade-off between training error and margin.
        Value range > 0.

        Defaults to 100.0.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, optional

        Defaults to rbf.
    degree : int, optional
        Coefficient for the poly kernel type.
        Value range >= 1.

        Defaults to 3.
    gamma : float, optional
        Coefficient for the rbf kernel type.

        Defaults to to 1.0/number of features in the dataset.

        Only valid when ``kernel`` is rbf.
    coef_lin : float, optional
        Coefficient for the poly/sigmoid kernel type.

        Defaults to 0.
    coef_const : float, optional
        Coefficient for the poly/sigmoid kernel type.

        Defaults to 0.
    shrink : bool, optional
        If true, use shrink strategy.

        Defaults to True.
    tol : float, optional
        Specifies the error tolerance in the training process.

        Value range > 0.

        Defaults to 0.001.
    evaluation_seed : int, optional
        The random seed in parameter selection.

        Value range >= 0.

        Defaults to 0.
    thread_ratio : float, optional
        Controls the proportion of available threads to use.
        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads. Values between 0 and 1
        will use that percentage of available threads. Values outside this
        range tell PAL to heuristically determine the number of threads to use.

        Defaults to 0.0.
    nu : float, optional
        The value for both the upper bound of the fraction of training errors
        and the lower bound of the fraction of support vectors.

        Defaults to 0.5.
    scale_info : {'no', 'standardization', 'rescale'}, optional
        Options:

          - 'no' : No scale.
          - 'standardization' : Transforms the data to have zero mean
            and unit variance.
          - 'rescale' : Rescales the range of the features to scale the
            range in [-1,1].

        Defaults to standardization.

    handle_missing : bool, optional
        Whether to handle missing values:

            False: No,

            True: Yes.

        Defaults to True.

    categorical_variable : str or list of str, optional
        Specifies INTEGER column(s) in the data that should be treated as categorical.
    category_weight : float, optional
        Represents the weight of category attributes.
        Value range > 0.

        Defaults to 0.707.

    Attributes
    ----------
    model_ : DataFrame
        Model content.

    stat_ : DataFrame
        Statistics content.

    Examples
    --------
    Training data:

    >>> df_fit.head(10).collect()
       ID  ATTRIBUTE1  ATTRIBUTE2  ATTRIBUTE3 ATTRIBUTE4
    0   0         1.0        10.0       100.0          A
    1   1         1.1        10.1       100.0          A
    2   2         1.2        10.2       100.0          A
    3   3         1.3        10.4       100.0          A
    4   4         1.2        10.3       100.0         AB
    5   5         4.0        40.0       400.0         AB
    6   6         4.1        40.1       400.0         AB
    7   7         4.2        40.2       400.0         AB
    8   8         4.3        40.4       400.0         AB
    9   9         4.2        40.3       400.0         AB

    Create OneClassSVM instance and call fit:

    >>> svc_one = svm.OneClassSVM(conn, scale_info='no', category_weight=1)
    >>> svc_one.fit(df_fit, 'ID', ['ATTRIBUTE1', 'ATTRIBUTE2', 'ATTRIBUTE3',
    ...                            'ATTRIBUTE4'])
    >>> df_predict = conn.table("DATA_TBL_SVC_ONE_PREDICT")
    >>> df_predict.head(10).collect()
       ID  ATTRIBUTE1  ATTRIBUTE2  ATTRIBUTE3 ATTRIBUTE4
    0   0         1.0        10.0       100.0          A
    1   1         1.1        10.1       100.0          A
    2   2         1.2        10.2       100.0          A
    3   3         1.3        10.4       100.0          A
    4   4         1.2        10.3       100.0         AB
    5   5         4.0        40.0       400.0         AB
    6   6         4.1        40.1       400.0         AB
    7   7         4.2        40.2       400.0         AB
    8   8         4.3        40.4       400.0         AB
    9   9         4.2        40.3       400.0         AB
    >>> features = ['ATTRIBUTE1', 'ATTRIBUTE2', 'ATTRIBUTE3',
    ...             'ATTRIBUTE4']

    Call predict:

    >>> svc_one.predict(df_predict, 'ID', features).head(10).collect()
       ID SCORE PROBABILITY
    0   0    -1        None
    1   1     1        None
    2   2     1        None
    3   3    -1        None
    4   4    -1        None
    5   5    -1        None
    6   6    -1        None
    7   7     1        None
    8   8    -1        None
    9   9    -1        None
    """
    #pylint:disable=too-many-arguments
    def  __init__(self, conn_context, c=None, kernel='rbf', degree=None, gamma=None,
                  coef_lin=None, coef_const=None, shrink=True, tol=None,
                  evaluation_seed=None, thread_ratio=None, nu=None, scale_info=None,
                  handle_missing=True, categorical_variable=None, category_weight=None):
        #pylint:disable=too-many-locals
        super(OneClassSVM, self).__init__(conn_context, 4, c, kernel, degree, gamma, coef_lin,
                                          coef_const, False, shrink, tol, evaluation_seed,
                                          thread_ratio, nu, scale_info, None, handle_missing, categorical_variable,
                                          category_weight, None)

    def fit(self, data, key=None, features=None, categorical_variable=None):
        """
        Fit the model when given training dataset and other attributes.

        Parameters
        ----------
        data : DataFrame
            DataFrame containing the data.
        key : str
            Name of the ID column. If ``key`` is not provided, it is assumed
            that the input has no ID column.
        features : list of str, optional
            Names of the feature columns.
            If ``features`` is not provided, it defaults to all
            the non-ID columns.
        categorical_variable : str or list of str, optional
            Specifies INTEGER column(s) specified that should be treated as categorical.
            Other INTEGER columns will be treated as continuous.
        """
        self._fit(data=data, key=key, features=features, label=None, categorical_variable=categorical_variable)

    def predict(self, data, key, features=None):#pylint:disable=too-many-locals
        """
        Predict the dataset using the trained model.

        Parameters
        ----------
        data : DataFrame
            DataFrame containing the data.
        key : str
            Name of the ID column.
        features : list of str, optional
            Names of the feature columns.
            If ``features`` is not provided, it defaults to all
            the non-ID columns.

        Returns
        -------
        DataFrame
            Predict result, structured as follows:
              - ID column, with the same name and type as ``data``'s ID column.
              - Score, type NVARCHAR(100), prediction value.
              - PROBABILITY, type DOUBLE, prediction probability.
                Always NULL. This column is only used for SVC and SVRanking.
        """
        return self._predict(data=data, key=key, features=features)
