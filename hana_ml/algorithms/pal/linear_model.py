"""
This module contains PAL wrapper and helper functions for linear model algorithms.

The following classes are available:

    * :class:`LinearRegression`
    * :class:`LogisticRegression`
"""

#pylint: disable=too-many-lines
#pylint: disable=line-too-long
#pylint: disable=relative-beyond-top-level
import itertools
import logging
import uuid

from hdbcli import dbapi

from hana_ml.ml_exceptions import FitIncompleteError
#from hana_ml.dataframe import quotename
from .pal_base import (
    PALBase,
    ParameterTable,
    parse_one_dtype,
    ListOfStrings
)
from . import metrics

logger = logging.getLogger(__name__) #pylint: disable=invalid-name

class LinearRegression(PALBase):
    r"""
    Linear regression is an approach to model the linear relationship between a variable,
    usually referred to as dependent variable, and one or more variables, usually referred to as independent variables, denoted as predictor vector .

    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA system.

    solver : {'QR', 'SVD', 'CD', 'Cholesky', 'ADMM'}, optional

        Algorithms to use to solve the least square problem. Case-insensitive.

              - 'QR': QR decomposition.
              - 'SVD': singular value decomposition.
              - 'CD': cyclical coordinate descent method.
              - 'Cholesky': Cholesky decomposition.
              - 'ADMM': alternating direction method of multipliers.

        'CD' and 'ADMM' are supported only when ``var_select`` is 'all'.

        Defaults to QR decomposition.

    var_select : {'all', 'forward', 'backward'}, optional

        Method to perform variable selection.

            - 'all': all variables are included.
            - 'forward': forward selection.
            - 'backward': backward selection.

        'forward' and 'backward' selection are supported only when ``solver``
        is 'QR', 'SVD' or 'Cholesky'.

        Defaults to 'all'.

    intercept : bool, optional

        If true, include the intercept in the model.

        Defaults to True.

    alpha_to_enter : float, optional

        P-value for forward selection.
        Valid only when ``var_select`` is 'forward'.

        Defaults to 0.05.

    alpha_to_remove : float, optional

        P-value for backward selection.
        Valid only when ``var_select`` is 'backward'.

        Defaults to 0.1.

    enet_lambda : float, optional

        Penalized weight. Value should be greater than or equal to 0.
        Valid only when ``solver`` is 'CD' or 'ADMM'.

    enet_alpha : float, optional

        Elastic net mixing parameter.
        Ranges from 0 (Ridge penalty) to 1 (LASSO penalty) inclusively.
        Valid only when ``solver`` is 'CD' or 'ADMM'.

        Defaults to 1.0.

    max_iter : int, optional

        Maximum number of passes over training data.
        If convergence is not reached after the specified number of
        iterations, an error will be generated.
        Valid only when ``solver`` is 'CD' or 'ADMM'.

        Defaults to 1e5.

    tol : float, optional

        Convergence threshold for coordinate descent.
        Valid only when ``solver`` is 'CD'.

        Defaults to 1.0e-7.

    pho : float, optional

        Step size for ADMM. Generally, it should be greater than 1.
        Valid only when ``solver`` is 'ADMM'.

        Defaults to 1.8.

    stat_inf : bool, optional
        If true, output t-value and Pr(>|t|) of coefficients.

        Defaults to False.

    adjusted_r2 : bool, optional

        If true, include the adjusted R\ :sup:`2` \ value in statistics.

        Defaults to False.

    dw_test : bool, optional

        If true, conduct Durbin-Watson test under null hypothesis that errors do not follow a first order autoregressive process.
        Not available if elastic net regularization is enabled or intercept is ignored.

        Defaults to False.

    reset_test : int, optional

        Specifies the order of Ramsey RESET test.
        Ramsey RESET test with power of variables ranging from 2 to this value (greater than 1) will be conducted.
        Value 1 means RESET test will not be conducted. Not available if elastic net regularization is enabled or intercept is ignored.

        Defaults to 1.

    bp_test : bool, optional

        If true, conduct Breusch-Pagan test under null hypothesis that homoscedasticity is satisfied.
        Not available if elastic net regularization is enabled or intercept is ignored.

        Defaults to False.

    ks_test : bool, optional

        If true, conduct Kolmogorov-Smirnov normality test under null hypothesis that errors follow a normal distribution.
        Not available if elastic net regularization is enabled or intercept is ignored.

        Defaults to False.

    thread_ratio : float, optional

        Controls the proportion of available threads to use.
        The value range is from 0 to 1, where 0 means only using 1 thread, and 1 means using at most all the currently available threads.
        Values outside this range tell PAL to heuristically determine the number of threads to use.
        Valid only when ``solver`` is 'QR', 'CD', 'Cholesky' or 'ADMM'.

        Defaults to 0.0.

    categorical_variable : str or ist of str, optional

        Specifies INTEGER columns specified that should be be treated as categorical.
        Other INTEGER columns will be treated as continuous.

    pmml_export : {'no', 'multi-row'}, optional

        Controls whether to output a PMML representation of the model, and how to format the PMML. Case-insensitive.

            - 'no' or not provided: No PMML model.
            - 'multi-row': Exports a PMML model, splitting it
              across multiple rows if it doesn't fit in one.

        Prediction does not require a PMML model.

    Attributes
    ----------

    coefficients_ : DataFrame

        Fitted regression coefficients.

    pmml_ : DataFrame

        PMML model. Set to None if no PMML model was requested.

    fitted_ : DataFrame

        Predicted dependent variable values for training data.
        Set to None if the training data has no row IDs.

    statistics_ : DataFrame

        Regression-related statistics, such as mean squared error.

    Examples
    --------

    Training data:

    >>> df.collect()
      ID       Y    X1 X2  X3
    0  0  -6.879  0.00  A   1
    1  1  -3.449  0.50  A   1
    2  2   6.635  0.54  B   1
    3  3  11.844  1.04  B   1
    4  4   2.786  1.50  A   1
    5  5   2.389  0.04  B   2
    6  6  -0.011  2.00  A   2
    7  7   8.839  2.04  B   2
    8  8   4.689  1.54  B   1
    9  9  -5.507  1.00  A   2

    Training the model:

    >>> lr = LinearRegression(cc,
    ...                       thread_ratio=0.5,
    ...                       categorical_variable=["X3"])
    >>> lr.fit(df, key='ID', label='Y')

    Prediction:

    >>> df2.collect()
       ID     X1 X2  X3
    0   0  1.690  B   1
    1   1  0.054  B   2
    2   2  0.123  A   2
    3   3  1.980  A   1
    4   4  0.563  A   1
    >>> lr.predict(df2, key='ID').collect()
       ID      VALUE
    0   0  10.314760
    1   1   1.685926
    2   2  -7.409561
    3   3   2.021592
    4   4  -3.122685
    """
    # pylint: disable=too-many-arguments,too-many-instance-attributes, too-many-locals, too-many-statements, too-many-branches
    solver_map = {'qr': 1, 'svd': 2, 'cd': 4, 'cholesky': 5, 'admm': 6}
    var_select_map = {'all': 0, 'forward': 1, 'backward': 2}
    pmml_export_map = {'no': 0, 'multi-row': 2}
    def __init__(self,
                 conn_context,
                 solver=None,
                 var_select=None,
                 intercept=True,
                 alpha_to_enter=None,
                 alpha_to_remove=None,
                 enet_lambda=None,
                 enet_alpha=None,
                 max_iter=None,
                 tol=None,
                 pho=None,
                 stat_inf=False,
                 adjusted_r2=False,
                 dw_test=False,
                 reset_test=None,
                 bp_test=False,
                 ks_test=False,
                 thread_ratio=None,
                 categorical_variable=None,
                 pmml_export=None):
        super(LinearRegression, self).__init__(conn_context)
        self.solver = self._arg('solver', solver, self.solver_map)
        self.var_select = self._arg('var_select', var_select, self.var_select_map)
        self.intercept = self._arg('intercept', intercept, bool)
        self.alpha_to_enter = self._arg('alpha_to_enter', alpha_to_enter, float)
        self.alpha_to_remove = self._arg('alpha_to_remove', alpha_to_remove, float)
        self.enet_lambda = self._arg('enet_lambda', enet_lambda, float)
        self.enet_alpha = self._arg('enet_alpha', enet_alpha, float)
        self.max_iter = self._arg('max_iter', max_iter, int)
        self.tol = self._arg('tol', tol, float)
        self.pho = self._arg('pho', pho, float)
        self.stat_inf = self._arg('stat_inf', stat_inf, bool)
        self.adjusted_r2 = self._arg('adjusted_r2', adjusted_r2, bool)
        self.dw_test = self._arg('dw_test', dw_test, bool)
        self.reset_test = self._arg('reset_test', reset_test, int)
        self.bp_test = self._arg('bp_test', bp_test, bool)
        self.ks_test = self._arg('ks_test', ks_test, bool)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        self.categorical_variable = self._arg('categorical_variable',
                                              categorical_variable, ListOfStrings)
        self.pmml_export = self._arg('pmml_export', pmml_export, self.pmml_export_map)

        if solver is not None:
            if solver.lower() == 'cd' or solver.lower() == 'admm':
                if var_select is not None and var_select.lower() != 'all':
                    msg = ('var_select cannot be {} when solver ' +
                           'is {}.').format(var_select.lower(), solver.lower())
                    logger.error(msg)
                    raise ValueError(msg)

        if solver is None or (solver.lower() != 'cd' and solver.lower() != 'admm'):
            if enet_lambda is not None:
                msg = ('enet_lambda is applicable only when solver is ' +
                       'coordinate descent or admm.')
                logger.error(msg)
                raise ValueError(msg)
            if enet_alpha is not None:
                msg = ('enet_alpha is applicable only when solver is ' +
                       'coordinate descent or admm.')
                logger.error(msg)
                raise ValueError(msg)
            if max_iter is not None:
                msg = ('max_iter is applicable only when solver is ' +
                       'coordinate descent or admm.')
                logger.error(msg)
                raise ValueError(msg)

        if (solver is None or solver.lower() != 'cd') and tol is not None:
            msg = 'tol is applicable only when solver is coordinate descent.'
            logger.error(msg)
            raise ValueError(msg)

        if (solver is None or solver.lower() != 'admm') and pho is not None:
            msg = 'pho is applicable only when solver is admm.'
            logger.error(msg)
            raise ValueError(msg)

        if var_select is None or var_select.lower() != 'forward':
            if alpha_to_enter is not None:
                msg = 'alpha_to_enter is applicable only when var_select is forward.'
                logger.error(msg)
                raise ValueError(msg)

        if var_select is None or var_select.lower() != 'backward':
            if alpha_to_remove is not None:
                msg = 'alpha_to_remove is applicable only when var_select is backward.'
                logger.error(msg)
                raise ValueError(msg)

        if enet_lambda is not None or enet_alpha is not None or intercept is False:
            if dw_test is not None and dw_test is not False:
                msg = ('dw_test is applicable only when elastic net regularization ' +
                       'is disabled and the model includes an intercept.')
                logger.error(msg)
                raise ValueError(msg)

            if reset_test is not None and reset_test != 1:
                msg = ('reset_test is applicable only when elastic net regularization ' +
                       'is disabled and the model includes an intercept.')
                logger.error(msg)
                raise ValueError(msg)

            if bp_test is not None and bp_test is not False:
                msg = ('bp_test is applicable only when elastic net regularization ' +
                       'is disabled and the model includes an intercept.')
                logger.error(msg)
                raise ValueError(msg)

            if ks_test is not None and ks_test is not False:
                msg = ('ks_test is applicable only when elastic net regularization ' +
                       'is disabled and the model includes an intercept.')
                logger.error(msg)
                raise ValueError(msg)

        if isinstance(reset_test, bool):
            msg = ('reset_test should be an integer, not a boolean ' +
                   'indicating whether or not to conduct the Ramsey RESET test.')
            logger.error(msg)
            raise TypeError(msg)

        if enet_alpha is not None and not 0 <= enet_alpha <= 1:
            msg = 'enet_alpha {!r} is out of bounds.'.format(enet_alpha)
            logger.error(msg)
            raise ValueError(msg)

    def fit(self,
            data,
            key=None,
            features=None,
            label=None,
            categorical_variable=None):
        r"""
        Fit regression model based on training data.

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
            If ``label`` is not provided, it defaults to the last column.

        categorical_variable : str or list of str, optional

            Specifies INTEGER column(s) that should be treated as categorical.
            Other INTEGER columns will be treated as continuous.
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

        #label is the first non-ID column in pal_linear_regression function
        data = data[id_col + [label] + features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()

        outputs = ['COEF', 'PMML', 'FITTED', 'STATS', 'OPTIMAL_PARAM']
        outputs = ['#PAL_LINEAR_REGRESSION_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                   for name in outputs]
        coef_tbl, pmml_tbl, fitted_tbl, stats_tbl, opt_param_tbl = outputs

        param_rows = [
            ('ALG', self.solver, None, None),
            ('VARIABLE_SELECTION', self.var_select, None, None),
            ('NO_INTERCEPT', not self.intercept, None, None),
            ('ALPHA_TO_ENTER', None, self.alpha_to_enter, None),
            ('ALPHA_TO_REMOVE', None, self.alpha_to_remove, None),
            ('ENET_LAMBDA', None, self.enet_lambda, None),
            ('ENET_ALPHA', None, self.enet_alpha, None),
            ('MAX_ITERATION', self.max_iter, None, None),
            ('THRESHOLD', None, self.tol, None),
            ('PHO', None, self.pho, None),
            ('STAT_INF', self.stat_inf, None, None),
            ('ADJUSTED_R2', self.adjusted_r2, None, None),
            ('DW_TEST', self.dw_test, None, None),
            ('RESET_TEST', self.reset_test, None, None),
            ('BP_TEST', self.bp_test, None, None),
            ('KS_TEST', self.ks_test, None, None),
            ('PMML_EXPORT', self.pmml_export, None, None),
            ('HAS_ID', key is not None, None, None)
        ]

        if self.solver != 2:
            param_rows.append(('THREAD_RATIO', None, self.thread_ratio, None))

        if self.categorical_variable is not None:
            param_rows.extend(('CATEGORICAL_VARIABLE', None, None, variable)
                              for variable in self.categorical_variable)
        if categorical_variable is not None:
            param_rows.extend(('CATEGORICAL_VARIABLE', None, None, variable)
                              for variable in categorical_variable)
        #coef_spec = [
        #    ('VARIABLE_NAME', NVARCHAR(1000)),
        #    ('COEFFICIENT_VALUE', DOUBLE),
        #    ('T_VALUE', DOUBLE),
        #    ('P_VALUE', DOUBLE)
        #]

        #pmml_spec = [
        #    ('ROW_INDEX', INTEGER),
        #    ('MODEL_CONTENT', NVARCHAR(5000)),
        #]

        #fitted_spec = [
        #    parse_one_dtype(data.dtypes()[0]),
        #    ('VALUE', DOUBLE),
        #]
        #statistics_spec = [
        #    ('STAT_NAME', NVARCHAR(256)),
        #    ('STAT_VALUE', NVARCHAR(1000)),
        #]

        try:
            self._call_pal_auto('PAL_LINEAR_REGRESSION',
                                data,
                                ParameterTable().with_data(param_rows),
                                coef_tbl,
                                pmml_tbl,
                                fitted_tbl,
                                stats_tbl,
                                opt_param_tbl)
        except dbapi.Error as db_err:
            #msg = 'HANA error while attempting to fit linear regression model.'
            logger.exception(str(db_err))
            raise

        # pylint: disable=attribute-defined-outside-init
        conn = self.conn_context
        self.coefficients_ = conn.table(coef_tbl)
        self.pmml_ = conn.table(pmml_tbl) if self.pmml_export else None
        self.fitted_ = conn.table(fitted_tbl) if key is not None else None
        self.statistics_ = conn.table(stats_tbl)

    def predict(self, data, key, features=None):
        r"""
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

        Returns
        -------

        DataFrame

            Predicted values, structured as follows:

                - ID column: with same name and type as ``data`` 's ID column.
                - VALUE: type DOUBLE, representing predicted values.

            .. note  ::

                predict() will pass the ``pmml_`` table to PAL as the model
                representation if there is a ``pmml_`` table, or the ``coefficients_``
                table otherwise.
        """
        if getattr(self, 'pmml_', None) is not None:
            model = self.pmml_
        elif hasattr(self, 'coefficients_'):
            model = self.coefficients_
        else:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")

        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)

        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols

        data = data[[key] + features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()

        fitted_tbl = '#PAL_LINEAR_REGRESSION_FITTED_TBL_{}_{}'.format(self.id, unique_id)

        param_rows = [
            ('THREAD_RATIO', None, self.thread_ratio, None)
        ]

        #fitted_spec = [
        #    parse_one_dtype(data.dtypes()[0]),
        #    ('VALUE', DOUBLE),
        #]

        try:
            self._call_pal_auto('PAL_LINEAR_REGRESSION_PREDICT',
                                data,
                                model,
                                ParameterTable().with_data(param_rows),
                                fitted_tbl)
        except dbapi.Error as db_err:
            #msg = 'HANA error during linear regression prediction.'
            logger.exception(str(db_err))
            raise

        return self.conn_context.table(fitted_tbl)

    def score(self, data, key, features=None, label=None):
        r"""
        Returns the coefficient of determination R: \sup:`2`\ of the prediction.

        Parameters
        ----------

        data : DataFrame

            Data on which to assess model performance.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.

            If ``features`` is not provided, it defaults all non-ID,
            non-label columns.

        label : str, optional

            Name of the dependent variable.
            If ``label`` is not provided, it defaults to the last column.

        Returns
        -------

        accuracy : float

            Returns the coefficient of determination R^2 of the prediction.

            .. note ::

                score() will pass the ``pmml_`` table to PAL as the model
                representation if there is a ``pmml_`` table, or the ``coefficients_``
                table otherwise.
        """

        if getattr(self, 'pmml_', None) is None or not hasattr(self, 'coefficients_'):
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
        prediction = self.predict(data, key=key, features=features)
        prediction = prediction.rename_columns(['ID_P', 'PREDICTION'])
        original = data[[key, label]].rename_columns(['ID_A', 'ACTUAL'])
        joined = original.join(prediction, 'ID_P=ID_A').select('ACTUAL', 'PREDICTION')
        return metrics.r2_score(self.conn_context,
                                joined,
                                label_true='ACTUAL',
                                label_pred='PREDICTION')


class LogisticRegression(PALBase):#pylint:disable=too-many-instance-attributes
    r"""
    Logistic regression model that handles binary-class and multi-class
    classification problems.

    Parameters
    ----------

    conn_context :  ConnectionContext

        Connection to the HANA system.

    multi_class : bool, optional

        If true, perform multi-class classification. Otherwise, there must be only two classes.

        Defaults to False.

    max_iter : int, optional

        Maximum number of iterations taken for the solvers to converge. If convergence is not reached after this number, an error will be generated.

                - multi-class: Defaults to 100.
                - binary-class: Defaults to 100000 when ``solver`` is cyclical, 1000 when ``solver`` is proximal, otherwise 100.

    pmml_export : {'no', 'single-row', 'multi-row'}, optional

        Controls whether to output a PMML representation of the model, and how to format the PMML. Case-insensitive.

          - multi-class:

            - 'no' or not provided: No PMML model.
            - 'multi-row': Exports a PMML model, splitting it across multiple rows if it doesn't fit in one.

          - binary-class:

            - 'no' or not provided: No PMML model.
            - 'single-row': Exports a PMML model in a maximum of one row. Fails if the model doesn't fit in one row.
            - 'multi-row': Exports a PMML model, splitting it across multiple rows if it doesn't fit in one.

        Defaults to 'no'.

    categorical_variable : str or list of str, optional(deprecated)

        Specifies INTEGER column(s) in the data that should be treated category variable.

    standardize : bool, optional

        If true, standardize the data to have zero mean and unit
        variance.

        Defaults to True.

    stat_inf : bool, optional

        If true, proceed with statistical inference.

        Defaults to False.

    solver : {'auto', 'newton', 'cyclical', 'lbfgs', 'stochastic', 'proximal'}, optional

        Optimization algorithm.

            - 'auto' : automatically determined by system based on input data and parameters.
            - 'newton': Newton iteration method.
            - 'cyclical': Cyclical coordinate descent method to fit elastic net regularized logistic regression.
            - 'lbfgs': LBFGS method (recommended when having many independent variables).
            - 'stochastic': Stochastic gradient descent method (recommended when dealing with very large dataset).
            - 'proximal': Proximal gradient descent method to fit elastic net regularized logistic regression.

        Only valid when ``multi_class`` is False.

        Defaults to 'auto'.

    alpha : float, optional

        Elastic net mixing parameter.
        Only valid when ``multi_class`` is False and ``solver`` is newton, cyclical, lbfgs or proximal.

        Defaults to 1.0.

    lamb : float, optional

        Penalized weight.
        Only valid when ``multi_class`` is False and ``solver`` is newton, cyclical, lbfgs or proximal.

        Defaults to 0.0.

    tol : float, optional

        Convergence threshold for exiting iterations.
        Only valid when ``multi_class`` is False.

        Defaults to 1.0e-7 when ``solver`` is cyclical, 1.0e-6 otherwise.

    epsilon : float, optional

        Determines the accuracy with which the solution is to
        be found.

        Only valid when ``multi_class`` is False and the ``solver`` is newton or lbfgs.

        Defaults to 1.0e-6 when ``solver`` is newton, 1.0e-5 when ``solver`` is lbfgs.

    thread_ratio : float, optional

        Controls the proportion of available threads to use for fit() method.
        The value range is from 0 to 1, where 0 indicates a single thread, and 1 indicates up to all available threads.
        Values between 0 and 1 will use that percentage of available threads. Values outside this range tell PAL to heuristically determine the number of threads to use.

        Defaults to 1.0.

    max_pass_number : int, optional

        The maximum number of passes over the data.
        Only valid when ``multi_class`` is False and ``solver`` is 'stochastic'.

        Defaults to 1.

    sgd_batch_number : int, optional

        The batch number of Stochastic gradient descent.
        Only valid when ``multi_class`` is False and ``solver`` is 'stochastic'.

        Defaults to 1.

    precompute : bool, optional

        Whether to pre-compute the Gram matrix.
        Only valid when ``solver`` is 'cyclical'.

        Defaults to True.

    handle_missing : bool, optional

        Whether to handle missing values.

        Defaults to True.

    categorical_variable : str or list of str, optional

        Specifies INTEGER column(s) in the data that should be treated as categorical.
        By default, string is categorical, while int and double are numerical.

    lbfgs_m : int, optional

        Number of previous updates to keep.
        Only applicable when ``multi_class`` is False and ``solver`` is 'lbfgs'.

        Defaults to 6.

    resampling_method : {'cv', 'stratified_cv', 'bootstrap', 'stratified_bootstrap'}, optional

        The resampling method for model evaluation and parameter selection.
        If no value specified, neither model evaluation nor parameter selection is activated.

    metric : {'accuracy', 'f1_score', 'auc', 'nll'}, optional

        The evaluation metric used for model evaluation/parameter selection.

    fold_num : int, optional

        The number of folds for cross-validation.
        Mandatory and valid only when ``resampling_method`` is 'cv' or 'stratified_cv'.

    repeat_times : int, optional

        The number of repeat times for resampling.

        Defaults to 1.

    search_strategy : {'grid', 'random'}, optional

        The search method for parameter selection.

    random_search_times : int, optional

        The number of times to randomly select candidate parameters for selection.
        Mandatory and valid when ``search_strategy`` is 'random'.

    random_state : int, optional

        The seed for random generation. 0 indicates using system time as seed.

        Defaults to 0.

    progress_indicator_id : str, optional

        The ID of progress indicator for model evaluation/parameter selection.
        Progress indicator deactivated if no value provided.

    lamb_values : list of float, optional

        The values of ``lamb`` for parameter selection.

        Only valid when ``search_strategy`` is specified.

    lamb_range : list of float, optional

        The range of ``lamb`` for parameter selection, including a lower limit and an upper limit.

        Only valid when ``search_strategy`` is specified.

    alpha_values : list of float, optional

        The values of ``alpha`` for parameter selection.

        Only valid when ``search_strategy`` is specified.

    alpha_range : list of float, optional

        The range of ``alpha`` for parameter selection, including a lower limit and an upper limit.

        Only valid when ``search_strategy`` is specified.

    class_map0 : str, optional (deprecated)

        Categorical label to map to 0.
        ``class_map0`` is mandatory when ``label`` column type is VARCHAR or NVARCHAR

        Only valid when ``multi_class`` is False during binary class fit and score.

    class_map1 : str, optional (deprecated)

        Categorical label to map to 1.

        ``class_map1`` is mandatory when ``label`` column type is VARCHAR or NVARCHAR during binary class fit and score.

        Only valid when ``multi_class`` is False.

    Attributes
    ----------

    coef_ : DataFrame

        Values of the coefficients.

    result_ : DataFrame

        Model content.

    optim_param_ : DataFrame

        The optimal parameter set selected via cross-validation.
        Empty if cross-validation is not activated.

    stat_ : DataFrame

        Statistics info for the trained model, structured as follows:

            - 1st column: 'STAT_NAME', NVARCHAR(256)
            - 2nd column: 'STAT_VALUE', NVARCHAR(1000)

    pmml_ : DataFrame

        PMML model. Set to None if no PMML model was requested.

    Examples
    --------

    Training data:

    >>> df.collect()
       V1     V2  V3  CATEGORY
    0   B  2.620   0         1
    1   B  2.875   0         1
    2   A  2.320   1         1
    3   A  3.215   2         0
    4   B  3.440   3         0
    5   B  3.460   0         0
    6   A  3.570   1         0
    7   B  3.190   2         0
    8   A  3.150   3         0
    9   B  3.440   0         0
    10  B  3.440   1         0
    11  A  4.070   3         0
    12  A  3.730   1         0
    13  B  3.780   2         0
    14  B  5.250   2         0
    15  A  5.424   3         0
    16  A  5.345   0         0
    17  B  2.200   1         1
    18  B  1.615   2         1
    19  A  1.835   0         1
    20  B  2.465   3         0
    21  A  3.520   1         0
    22  A  3.435   0         0
    23  B  3.840   2         0
    24  B  3.845   3         0
    25  A  1.935   1         1
    26  B  2.140   0         1
    27  B  1.513   1         1
    28  A  3.170   3         1
    29  B  2.770   0         1
    30  B  3.570   0         1
    31  A  2.780   3         1

    Create LogisticRegression instance and call fit:

    >>> lr = linear_model.LogisticRegression(cc, solver='newton',
    ...                                      thread_ratio=0.1, max_iter=1000,
    ...                                      pmml_export='single-row',
    ...                                      stat_inf=True, tol=0.000001)
    >>> lr.fit(df,
    ...        features=['V1', 'V2', 'V3'],
    ...        label='CATEGORY',
    ...        categorical_variable=['V3'])
    >>> lr.coef_.collect()
                                           VARIABLE_NAME  COEFFICIENT
    0                                  __PAL_INTERCEPT__    17.044785
    1                                 V1__PAL_DELIMIT__A     0.000000
    2                                 V1__PAL_DELIMIT__B    -1.464903
    3                                                 V2    -4.819740
    4                                 V3__PAL_DELIMIT__0     0.000000
    5                                 V3__PAL_DELIMIT__1    -2.794139
    6                                 V3__PAL_DELIMIT__2    -4.807858
    7                                 V3__PAL_DELIMIT__3    -2.780918
    8  {"CONTENT":"{\"impute_model\":{\"column_statis...          NaN
    >>> pred_df.collect()
        ID V1     V2  V3
    0    0  B  2.620   0
    1    1  B  2.875   0
    2    2  A  2.320   1
    3    3  A  3.215   2
    4    4  B  3.440   3
    5    5  B  3.460   0
    6    6  A  3.570   1
    7    7  B  3.190   2
    8    8  A  3.150   3
    9    9  B  3.440   0
    10  10  B  3.440   1
    11  11  A  4.070   3
    12  12  A  3.730   1
    13  13  B  3.780   2
    14  14  B  5.250   2
    15  15  A  5.424   3
    16  16  A  5.345   0
    17  17  B  2.200   1

    Call predict():

    >>> result = lgr.predict(pred_df,
    ...                      key='ID',
    ...                      categorical_variable=['V3'],
    ...                      thread_ratio=0.1)
    >>> result.collect()
        ID CLASS   PROBABILITY
    0    0     1  9.503618e-01
    1    1     1  8.485210e-01
    2    2     1  9.555861e-01
    3    3     0  3.701858e-02
    4    4     0  2.229129e-02
    5    5     0  2.503962e-01
    6    6     0  4.945832e-02
    7    7     0  9.922085e-03
    8    8     0  2.852859e-01
    9    9     0  2.689207e-01
    10  10     0  2.200498e-02
    11  11     0  4.713726e-03
    12  12     0  2.349803e-02
    13  13     0  5.830425e-04
    14  14     0  4.886177e-07
    15  15     0  6.938072e-06
    16  16     0  1.637820e-04
    17  17     1  8.986435e-01

    Input data for score():

    >>> df_score.collect()
        ID V1     V2  V3  CATEGORY
    0    0  B  2.620   0         1
    1    1  B  2.875   0         1
    2    2  A  2.320   1         1
    3    3  A  3.215   2         0
    4    4  B  3.440   3         0
    5    5  B  3.460   0         0
    6    6  A  3.570   1         1
    7    7  B  3.190   2         0
    8    8  A  3.150   3         0
    9    9  B  3.440   0         0
    10  10  B  3.440   1         0
    11  11  A  4.070   3         0
    12  12  A  3.730   1         0
    13  13  B  3.780   2         0
    14  14  B  5.250   2         0
    15  15  A  5.424   3         0
    16  16  A  5.345   0         0
    17  17  B  2.200   1         1

    Call score():

    >>> lgr.score(df_score,
    ...           key='ID',
    ...           categorical_variable=['V3'],
    ...           thread_ratio=0.1)
    0.944444
    """
    solver_map = {'auto':-1, 'newton':0, 'cyclical':2, 'lbfgs':3, 'stochastic':4, 'proximal':6}
    pmml_map_multi = {'no': 0, 'multi-row': 1}
    pmml_map_binary = {'no': 0, 'single-row': 1, 'multi-row': 2}
    resampling_method_list = ('cv', 'stratified_cv', 'bootstrap', 'stratified_bootstrap')
    valid_metric_map = {'accuracy':'ACCURACY', 'f1_score':'F1_SCORE', 'auc':'AUC', 'nll':'NLL'}
    #pylint:disable=too-many-arguments, too-many-branches, too-many-statements
    def __init__(self,
                 conn_context,
                 multi_class=False,
                 max_iter=None,
                 pmml_export=None,
                 categorical_variable=None,
                 standardize=True,
                 stat_inf=False,
                 solver=None,
                 alpha=None,
                 lamb=None,
                 tol=None,
                 epsilon=None,
                 thread_ratio=None,
                 max_pass_number=None,
                 sgd_batch_number=None,
                 precompute=None,#adding new parameters
                 handle_missing=None,
                 resampling_method=None,
                 metric=None,
                 fold_num=None,
                 repeat_times=None,
                 search_strategy=None,
                 random_search_times=None,
                 random_state=None,
                 timeout=None,
                 lamb_values=None,
                 lamb_range=None,
                 alpha_values=None,
                 alpha_range=None,#new parameters ends here
                 lbfgs_m=None,
                 class_map0=None,
                 class_map1=None,
                 progress_indicator_id=None):
        #pylint:disable=too-many-locals
        super(LogisticRegression, self).__init__(conn_context)
        self.multi_class = self._arg('multi_class', multi_class, bool)
        self.max_iter = self._arg('max_iter', max_iter, int)
        #set default value and dict for solver of each case
        if self.multi_class:
            pmml_map = self.pmml_map_multi
        else:
            pmml_map = self.pmml_map_binary
            solver = 'newton' if solver is None else solver#maybe something wrong here
        self.pmml_export = self._arg('pmml_export', pmml_export, pmml_map)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        self.categorical_variable = self._arg('categorical_variable', categorical_variable,
                                              ListOfStrings)
        self.standardize = self._arg('standardize', standardize, bool)
        self.stat_inf = self._arg('stat_inf', stat_inf, bool)
        #below is for parameters specific for binary class classification
        binary_params = {'solver': solver, 'alpha': alpha, 'lamb': lamb, 'tol': tol,
                         'epsilon': epsilon, 'thread_ratio': thread_ratio,
                         'max_pass_number': max_pass_number, 'sgd_batch_number': sgd_batch_number,
                         'lbfgs_m': lbfgs_m, 'class_map0': class_map0, 'class_map1': class_map1}
        self._check_params_for_binary(binary_params)
        self.solver = self._arg('solver', solver, self.solver_map)
        if self.solver not in (0, 2, 3, 6) and alpha is not None:
            msg = ('Parameter alpha is only applicable when solver ' +
                   'is newton, cyclical, lbfgs or proximal.')
            logger.error(msg)
            raise ValueError(msg)
        self.alpha = self._arg('alpha', alpha, float)
        if self.solver not in (0, 2, 3, 6) and lamb is not None:
            msg = ('Parameter lamb is only applicable when solver is newton, cyclical, ' +
                   'lbfgs or proximal.')
            logger.error(msg)
            raise ValueError(msg)
        self.lamb = self._arg('lamb', lamb, float)
        self.tol = self._arg('tol', tol, float)#Corresponds to EXIT_THRESHOLD
        self.epsilon = self._arg('epsilon', epsilon, float)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        if self.solver != 4 and max_pass_number is not None:
            msg = 'Parameter max_pass_number is only applicable when solver is stochastic.'
            logger.error(msg)
            raise ValueError(msg)
        self.max_pass_number = self._arg('max_pass_number', max_pass_number, int)
        if self.solver != 4 and sgd_batch_number is not None:
            msg = 'Parameter sgd_batch_number is only applicable when solver is stochastic.'
            logger.error(msg)
            raise ValueError(msg)
        self.sgd_batch_number = self._arg('sgd_batch_number', sgd_batch_number, int)
        if self.solver != 3 and lbfgs_m is not None:
            msg = ('Parameter lbfgs_m will be applicable '+
                   'only if method is lbfgs.')
            logger.error(msg)
            raise ValueError(msg)
        #New parms start here
        self.precompute = self._arg('precompute', precompute, bool)
        self.handle_missing = self._arg('handle_missing', handle_missing, bool)
        #self.exit_threshold = self._arg('exit_threshold', exit_threshold, float)
        self.resampling_method = self._arg('resampling_method', resampling_method, str)
        if self.resampling_method is not None:
            if self.resampling_method not in self.resampling_method_list:
                msg = ("Resampling method '{}' is not supported ".format(self.resampling_method) +
                       "for model evaluation or parameter selection.")
                logger.error(msg)
                raise ValueError(msg)
        self.metric = self._arg('metric', metric, self.valid_metric_map)
        self.fold_num = self._arg('fold_num', fold_num, int)
        self.repeat_times = self._arg('repeat_times', repeat_times, int)
        self.search_strategy = self._arg('search_stategy', search_strategy, str)
        if self.search_strategy is not None:
            if self.search_strategy not in ('grid', 'random'):
                msg = ("Search strategy '{}' is not available for ".format(self.search_strategy)+
                       "parameter selection.")
                logger.error(msg)
                raise ValueError(msg)
        self.random_search_times = self._arg('random_search_times', random_search_times, int)
        self.random_state = self._arg('random_state', random_state, int)
        self.timeout = self._arg('timeout', timeout, int)
        self.lamb_values = None
        lamb_vals = self._arg('lamb_values', lamb_values, list)
        #check if all lambda values are of (or can be converted to) float type
        if lamb_vals is not None:
            self.lamb_values = '{'
            count = 1
            for lamb_ in lamb_vals:
                self.lamb_values += str(lamb_)
                if count < len(lamb_vals):
                    self.lamb_values += ','
                count += 1
            self.lamb_values += '}'
        self.lamb_range = None
        lamb_rg = self._arg('alpha_range', lamb_range, list)
        if lamb_rg is not None:
            if len(lamb_rg) != 2:
                msg = ('The range param should contain '+
                       'exactly two elements.')
                logger.error(msg)
                raise ValueError(msg)
            self.lamb_range = lamb_rg
        self.alpha_values = None
        alpha_vals = self._arg('alpha_values', alpha_values, list)
        #check if all alpha values are of (or can be converted to) float type.
        if alpha_vals is not None:
            self.alpha_values = '{'
            count = 1
            for alpha_ in alpha_vals:
                self.alpha_values += str(alpha_)
                if count < len(alpha_vals):
                    self.alpha_values += ','
                count += 1
            self.alpha_values += '}'
        self.alpha_range = None
        alpha_rg = self._arg('alpha_range', alpha_range, list)
        if alpha_rg is not None:
            if len(alpha_rg) != 2:
                msg = ('The range param should contain '+
                       'exactly two elements.')
                logger.error(msg)
                raise ValueError(msg)
            self.alpha_range = alpha_rg
        self.lbfgs_m = self._arg('lbfgs_m', lbfgs_m, int)
        self.class_map0 = self._arg('class_map0', class_map0, str)
        self.class_map1 = self._arg('class_map1', class_map1, str)
        self.progress_indicator_id = self._arg('progress_indicator_id', progress_indicator_id, str)
        self.label_type = None

    #pylint:disable=too-many-locals, too-many-statements, invalid-name
    def fit(self,
            data,
            key=None,
            features=None,
            label=None,
            categorical_variable=None,
            class_map0=None,
            class_map1=None):
        r"""
        Fit the LR model when given training dataset.

        Parameters
        ----------

        data : DataFrame

            DataFrame containing the data.

        key : str, optional

            Name of the ID column. If ``key`` is not provided, it is assumed that the input has no ID column.

        features : list of str, optional

            Names of the feature columns.

            If ``features`` is not provided, it defaults to all non-ID,
            non-label columns.

        label : str, optional

            Name of the label column.
            If ``label`` is not provided, it defaults to the last column.

        categorical_variable : str or list of str, optional

            Specifies INTEGER column(s) that shoud be treated as categorical.
            Otherwise All INTEGER columns are treated as numerical.

        class_map0 : str, optional (deprecated)

            Categorical label to map to 0.
            ``class_map0`` is mandatory when ``label`` column type is VARCHAR or NVARCHAR during binary class fit and score.

            Only valid when ``multi_class`` is False.

        class_map1 : str, optional (deprecated)

            Categorical label to map to 1.
            ``class_map1`` is mandatory when ``label`` column type is VARCHAR or NVARCHAR during binary class fit and score.

            Only valid when ``multi_class`` is False.

        """
        key = self._arg('key', key, str)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        class_map0 = self._arg('class_map0', class_map0, str)
        class_map1 = self._arg('class_map1', class_map1, str)
        if class_map0 is not None:
            self.class_map0 = class_map0
        if class_map1 is not None:
            self.class_map1 = class_map1
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        categorical_variable = self._arg('categorical_variable', categorical_variable, ListOfStrings)
        if categorical_variable is not None:
            self.categorical_variable = categorical_variable
        cols = data.columns
        if key is not None:
            cols.remove(key)
        if label is None:
            label = cols[-1]
        self.label_type = data.dtypes([label])[0][1]
        cols.remove(label)
        if features is None:
            features = cols
        #check label sql type
        if not self.multi_class:
            self._check_label_sql_type(data, label)
        used_cols = [col for col in itertools.chain([key], features, [label])
                     if col is not None]
        data = data[used_cols]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()

        outputs = ['#LR_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                   for name in ['RESULT', 'PMML', 'STAT', 'OPTIM']]
        result_tbl, pmml_tbl, stat_tbl, optim_tbl = outputs
        #start with shared parameters and shared tables
        param_array = [('MAX_ITERATION', self.max_iter, None, None),
                       ('PMML_EXPORT', self.pmml_export, None, None),
                       ('HAS_ID', key is not None, None, None),
                       ('STANDARDIZE', self.standardize, None, None),
                       ('STAT_INF', self.stat_inf, None, None),
                       ('ENET_ALPHA', None, self.alpha, None),
                       ('ENET_LAMBDA', None, self.lamb, None),
                       ('EXIT_THRESHOLD', None, self.tol, None),
                       ('METHOD', self.solver, None, None),
                       ('RESAMPLING_METHOD', None, None, self.resampling_method),
                       ('EVALUATION_METRIC', None, None, self.metric),
                       ('FOLD_NUM', self.fold_num, None, None),
                       ('REPEAT_TIMES', self.repeat_times, None, None),
                       ('PARAM_SEARCH_STRATEGY', None, None, self.search_strategy),
                       ('RANDOM_SEARCH_TIMES', None, None, self.random_search_times),
                       ('SEED', self.random_state, None, None),
                       ('TIMEOUT', self.timeout, None, None),
                       ('ENET_LAMBDA_VALUES', None, None, self.lamb_values),
                       ('ENET_LAMBDA_RANGE', None, None,
                        str(self.lamb_range) if self.lamb_range is not None else None),
                       ('ENET_ALPHA_VALUES', None, None, self.alpha_values),
                       ('ENET_ALPHA_RANGE', None, None,
                        str(self.alpha_range) if self.alpha_range is not None else None),
                       ('PROGRESS_INDICATOR_ID', None, None, self.progress_indicator_id)
                      ]
        if self.categorical_variable is not None:
            param_array.extend([('CATEGORICAL_VARIABLE', None, None, col)
                                for col in self.categorical_variable])
        #if categorical_variable is not None:
        #    param_array.extend([('CATEGORICAL_VARIABLE', None, None, col)
        #                        for col in categorical_variable])
        #pmml_specs = [('ROW_INDEX', INTEGER), ('MODEL_CONTENT', NVARCHAR(5000))]
        #stat_specs = [('STAT_NAME', NVARCHAR(256)), ('STAT_VALUE', NVARCHAR(1000))]
        #optim-tbl is meaningful only in binary class, in multi-class, it's named placeholder table
        #optim_specs = [('PARAM_NAME', NVARCHAR(256)),
        #               ('INT_VALUE', INTEGER),
        #               ('DOUBLE_VALUE', DOUBLE),
        #               ('STRING_VALUE', NVARCHAR(1000))]
        if self.multi_class:
            proc_name = 'PAL_MULTICLASS_LOGISTIC_REGRESSION'
            #result_specs = [('VARIABLE_NAME', NVARCHAR(1000)),
            #                ('CLASS', NVARCHAR(100)),
            #                ('COEFFICIENT', DOUBLE),
            #                ('Z_SCORE', DOUBLE),
            #                ('P_VALUE', DOUBLE)]
            coef_list = ['VARIABLE_NAME', 'CLASS', 'COEFFICIENT']
        else:
            proc_name = "PAL_LOGISTIC_REGRESSION"
            #result_specs = [('VARIABLE_NAME', NVARCHAR(1000)),
            #                ('COEFFICIENT', DOUBLE),
            #                ('Z_SCORE', DOUBLE),
            #                ('P_VALUE', DOUBLE)]
            coef_list = ['VARIABLE_NAME', 'COEFFICIENT']
            param_array.extend([('EPSILON', None, self.epsilon, None),
                                ('THREAD_RATIO', None, self.thread_ratio, None),
                                ('MAX_PASS_NUMBER', self.max_pass_number, None, None),
                                ('SGD_BATCH_NUMBER', self.sgd_batch_number, None, None),
                                ('PRECOMPUTE', self.precompute, None, None),
                                ('HANDLE_MISSING', self.handle_missing, None, None),
                                ('LBFGS_M', self.lbfgs_m, None, None)
                                #('CLASS_MAP1', None, None, self.class_map1),
                                #('CLASS_MAP0', None, None, self.class_map0)
                               ])
            if self.label_type in ('VARCHAR', 'NVARCHAR'):
                param_array.extend([('CLASS_MAP1', None, None, self.class_map1),
                                    ('CLASS_MAP0', None, None, self.class_map0)
                                   ])
        try:
            self._call_pal_auto(proc_name,
                                data,
                                ParameterTable().with_data(param_array),
                                result_tbl,
                                pmml_tbl,
                                stat_tbl,
                                optim_tbl)
        except dbapi.Error as db_err:
            #logger.error("HANA error during LogisticRegression fit.", exc_info=True)
            logger.exception(str(db_err))
            raise
        #pylint:disable=attribute-defined-outside-init
        self.result_ = self.conn_context.table(result_tbl)
        self.coef_ = self.result_.select(coef_list)
        self.pmml_ = self.conn_context.table(pmml_tbl) if self.pmml_export else None
        self.optim_param_ = self.conn_context.table(optim_tbl)
        self.stat_ = self.conn_context.table(stat_tbl)

    def predict(self,
                data,
                key,
                features=None,
                categorical_variable=None,
                thread_ratio=None,
                class_map0=None,
                class_map1=None,
                verbose=False):
        #pylint:disable=too-many-locals
        r"""
        Predict with the dataset using the trained model.

        Parameters
        ----------

        data : DataFrame

            DataFrame containing the data.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.
            If ``features`` is not provided, it defaults to all non-ID columns.

        verbose : bool, optional

            If true, output scoring probabilities for each class.
            It is only applicable for multi-class case.

            Defaults to False.

        categorical_variable : str or list of str, optional (deprecated)

            Specifies INTEGER column(s) that shoud be treated as categorical.
            Otherwise all integer columns are treated as numerical.
            Mandatory if training data of the prediction model contains such
            data columns.

        thread_ratio : float, optional

            Controls the proportion of available threads to use.
            The value range is from 0 to 1, where 0 means only using 1 thread, and 1 means using at most all the currently available threads.
            Values outside this range tell pal to heuristically determine the number of threads to use.

            Defaults to 0.

        class_map0 : str, optional (deprecated)

            Categorical label to map to 0.
            ``class_map0`` is mandatory when ``label`` column type is varchar or nvarchar during binary class fit and score.only valid when ``multi_class`` is false.

        class_map1 : str, optional (deprecated)

            Categorical label to map to 1.
            ``class_map1`` is mandatory when ``label`` column type is varchar or nvarchar during binary class fit and score.
            Only valid when ``multi_class`` is false.

        Returns
        -------

        DataFrame
            Predicted result, structured as follows:

            - 1: ID column, with edicted class name.
            - 2: PROBABILITY, type DOUBLE

                - multi-class: probability of being predicted as the predicted class.
                - binary-class: probability of being predicted as the positive class.

           .. note ::

                predict() will pass the ``pmml_`` table to PAL as the model representation if there is a ``pmml_`` table, or the ``result_`` table otherwise.
        """
        if getattr(self, 'pmml_', None) is not None:
            model = self.pmml_
        elif hasattr(self, 'result_'):
            model = self.result_
        else:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")

        key = self._arg('arg', key, str, True)
        features = self._arg('features', features, ListOfStrings)
        verbose = self._arg('verbose', verbose, bool)
        thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()

        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols
        data = data[[key] + features]

        result_tbl = '#LR_PREDICT_RESULT_TBL_{}_{}'.format(self.id, unique_id)

        ###binary class only parameter: class_map, categorical_variable and thread_ratio
        if self.multi_class:
            #to avoid empty param table bug.
            param_array = [('VERBOSE_OUTPUT', verbose, None, None)]
            #result_specs = [parse_one_dtype(data.dtypes([key])[0]),
            #                ('CLASS', NVARCHAR(5000)),
            #                ('PROBABILITY', DOUBLE)]
            proc_name = 'PAL_MULTICLASS_LOGISTIC_REGRESSION_PREDICT'
        else:
            #0.0 is the default thread_ratio value used in prediction for binary case
            #thread_ratio = 0.0 if self.thread_ratio is None else self.thread_ratio
            param_array = [('THREAD_RATIO', None, thread_ratio, None)]
            if categorical_variable is not None:
                param_array.extend([('CATEGORICAL_VARIABLE', None, None, variable)
                                    for variable in categorical_variable])
            if self.categorical_variable is not None:
                param_array.extend([('CATEGORICAL_VARIABLE', None, None, variable)
                                    for variable in self.categorical_variable])
            if class_map0 is not None:
                param_array.extend([('CLASS_MAP0', None, None, class_map0),
                                    ('CLASS_MAP1', None, None, class_map1)])
            else:
                param_array.extend([('CLASS_MAP0', None, None, self.class_map0),
                                    ('CLASS_MAP1', None, None, self.class_map1)])
            #result_specs = [parse_one_dtype(data.dtypes([key])[0]),
            #                ('PROBABILITY', DOUBLE),
            #                ('CLASS', NVARCHAR(5000))]
            proc_name = 'PAL_LOGISTIC_REGRESSION_PREDICT'
        try:
            self._call_pal_auto(proc_name,
                                data,
                                model,
                                ParameterTable().with_data(param_array),
                                result_tbl)
        except dbapi.Error as db_err:
            #logger.error("HANA error during LogisticRegression prediction.", exc_info=True)
            logger.exception(str(db_err))
            raise
        result = self.conn_context.table(result_tbl)
        if result.has('SCORE'):
            # Anonymous block code path uses PAL's column names.
            result = result.rename_columns({'SCORE': 'PROBABILITY'})
        result = result[[key, 'CLASS', 'PROBABILITY']]
        return result

    def score(self,
              data,
              key,
              features=None,
              label=None,
              categorical_variable=None,
              thread_ratio=None,
              class_map0=None,
              class_map1=None):
        r"""
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------

        data : DataFrame

            DataFrame containing the data.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.
            If ``features`` is not provided, it defaults to all non-ID,
            non-label columns.

        label : str, optional

            Name of the label column.
            If ``label`` is not provided, it defaults to the last column.

        categorical_variable : str or list of str, optional (deprecated)

            Specifies INTEGER columns that shoud be treated as categorical, otherwise all integer columns are treated as numerical.
            Mandatory if training data of the prediction model contains such data columns.

        thread_ratio : float, optional

            Controls the proportion of available threads to use.
            The value range is from 0 to 1, where 0 means only using 1 thread, and 1 means using at most all the currently available threads.
            values outside this range tell pal to heuristically determine the number of threads to use.

            Defaults to 0.

        class_map0 : str, optional (deprecated)

            Categorical label to map to 0.
            ``class_map0`` is mandatory when ``label`` column type is varchar or nvarchar during binary class fit and score.
            Only valid when ``multi_class`` is false.

        class_map1 : str, optional (deprecated)

            Categorical label to map to 1.
            ``class_map1`` is mandatory when ``label`` column type is varchar or nvarchar during binary class fit and score.
            Only valid when ``multi_class`` is false.

        Returns
        -------

        accuracy : float

            Scalar accuracy value after comparing the predicted label
            and original label.
        """
        key = self._arg('key', key, str, True)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        categorical_variable = self._arg('categorical_variable',
                                         categorical_variable,
                                         ListOfStrings)
        thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        cols = data.columns
        cols.remove(key)
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols
        #check label sql type
        if not self.multi_class:
            self._check_label_sql_type(data, label)

        prediction = self.predict(data=data, key=key,
                                  features=features,
                                  categorical_variable=categorical_variable,
                                  thread_ratio=thread_ratio,
                                  class_map0=class_map0,
                                  class_map1=class_map1)
        prediction = prediction.select(key, 'CLASS').rename_columns(['ID_P', 'PREDICTION'])

        actual = data.select(key, label).rename_columns(['ID_A', 'ACTUAL'])
        joined = actual.join(prediction, 'ID_P=ID_A').select('ACTUAL', 'PREDICTION')

        return metrics.accuracy_score(self.conn_context, joined,
                                      label_true='ACTUAL',
                                      label_pred='PREDICTION')

    def _check_label_sql_type(self, data, label):
        label_sql_type = parse_one_dtype(data.dtypes([label])[0])[1]
        if label_sql_type.startswith("NVARCHAR") or label_sql_type.startswith("VARCHAR"):
            if self.class_map0 is None or self.class_map1 is None:
                msg = ("class_map0 and class_map1 are mandatory when `label` column type " +
                       "is VARCHAR or NVARCHAR.")
                logger.error(msg)
                raise ValueError(msg)

    def _check_params_for_binary(self, params):
        for name in params:
            msg = 'Parameter {} is only applicable for binary classification.'.format(name)
            if params[name] is not None and self.multi_class:
                logger.error(msg)
                raise ValueError(msg)
