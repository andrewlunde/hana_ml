"""
This module contains wrappers for PAL regression algorithms.

The following classes are available:

    * :class:`PolynomialRegression`
    * :class:`GLM`
    * :class:`ExponentialRegression`
    * :class:`BiVariateGeometricRegression`
    * :class:`BiVariateNaturalLogarithmicRegression`
    * :class:`CoxProportionalHazardModel`
"""

# pylint: disable=too-many-lines, line-too-long, too-many-arguments, too-many-branches, too-many-instance-attributes
import logging
import sys
import uuid
from hdbcli import dbapi
from hana_ml.ml_exceptions import FitIncompleteError
from .pal_base import (
    PALBase,
    ParameterTable,
    ListOfStrings,
)
from . import metrics
logger = logging.getLogger(__name__) #pylint: disable=invalid-name
if sys.version_info.major == 2:
    #pylint: disable=undefined-variable
    _INTEGER_TYPES = (int, long)
    _STRING_TYPES = (str, unicode)
else:
    _INTEGER_TYPES = (int,)
    _STRING_TYPES = (str,)

class PolynomialRegression(PALBase):
    r"""
    Polynomial regression is an approach to modeling the relationship between a scalar variable y and a variable denoted X. In polynomial regression,
    data is modeled using polynomial functions, and unknown model parameters are estimated from the data. Such models are called polynomial models.

    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA system.

    degree : int

        Degree of the polynomial model.

    decomposition : {'LU', 'SVD'}, optional

        Matrix factorization type to use. Case-insensitive.

          - 'LU': LU decomposition.
          - 'SVD': singular value decomposition.

        Defaults to LU decomposition.

    adjusted_r2 : bool, optional

        If true, include the adjusted R\ :sup:`2` \ value in the statistics table.

        Defaults to False.

    pmml_export : {'no', 'single-row', 'multi-row'}, optional

        Controls whether to output a PMML representation of the model,
        and how to format the PMML. Case-insensitive.

            - 'no' or not provided: No PMML model.
            - 'single-row': Exports a PMML model in a maximum of
              one row. Fails if the model doesn't fit in one row.
            - 'multi-row': Exports a PMML model, splitting it
              across multiple rows if it doesn't fit in one.

        Prediction does not require a PMML model.

    thread_ratio : float, optional

        Controls the proportion of available threads to use for prediction.
        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads. Values between 0 and 1
        will use that percentage of available threads. Values outside this
        range tell PAL to heuristically determine the number of threads to use.
        Does not affect fitting.

        Defaults to 0.

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

    Training data (based on y = x^3 - 2x^2 + 3x + 5, with noise):

    >>> df.collect()
       ID    X       Y
    0   1  0.0   5.048
    1   2  1.0   7.045
    2   3  2.0  11.003
    3   4  3.0  23.072
    4   5  4.0  49.041

    Training the model:

    >>> pr = PolynomialRegression(cc, degree=3)
    >>> pr.fit(df, key='ID')

    Prediction:

    >>> df2.collect()
       ID    X
    0   1  0.5
    1   2  1.5
    2   3  2.5
    3   4  3.5
    >>> pr.predict(df2, key='ID').collect()
       ID      VALUE
    0   1   6.157063
    1   2   8.401269
    2   3  15.668581
    3   4  33.928501

    Ideal output:

    >>> df2.select('ID', ('POWER(X, 3)-2*POWER(X, 2)+3*x+5', 'Y')).collect()
       ID       Y
    0   1   6.125
    1   2   8.375
    2   3  15.625
    3   4  33.875
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    decomposition_map = {'lu': 0, 'svd': 2}
    pmml_export_map = {'no': 0, 'single-row': 1, 'multi-row': 2}
    def __init__(self,
                 conn_context,
                 degree,
                 decomposition=None,
                 adjusted_r2=None,
                 pmml_export=None,
                 thread_ratio=None):
        super(PolynomialRegression, self).__init__(conn_context)
        self.degree = self._arg('degree', degree, int, required=True)
        self.decomposition = self._arg('decomposition', decomposition, self.decomposition_map)
        self.adjusted_r2 = self._arg('adjusted_r2', adjusted_r2, bool)
        self.pmml_export = self._arg('pmml_export', pmml_export, self.pmml_export_map)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)

    def fit(self, data, key=None, features=None, label=None):
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

            Names of the feature columns. Since the underlying
            PAL_POLYNOMIAL_REGRESSION only supports one feature,
            this list can only contain one element.
            If ``features`` is not provided, ``data`` must have exactly 1
            non-ID, non-label column, and ``features`` defaults to that
            column.

        label : str, optional

            Name of the dependent variable.

            Defaults to the last column.
            (This is not the PAL default.)
        """
        # pylint: disable=too-many-locals,too-many-statements

        key = self._arg('key', key, str)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)

        cols = data.columns
        if key is not None:
            maybe_id = [key]
            cols.remove(key)
        else:
            maybe_id = []
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols

        if len(features) != 1:
            msg = ('PAL polynomial regression requires exactly one ' +
                   'feature column.')
            logger.error(msg)
            raise TypeError(msg)

        data = data[maybe_id + [label] + features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()

        outputs = ['COEF', 'PMML', 'FITTED', 'STATS', 'OPTIMAL_PARAM']
        outputs = ['#PAL_POLYNOMIAL_REGRESSION_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                   for name in outputs]
        coef_tbl, pmml_tbl, fitted_tbl, stats_tbl, opt_param_tbl = outputs

        param_rows = [
            ('POLYNOMIAL_NUM', self.degree, None, None),
            ('ALG', self.decomposition, None, None),
            ('ADJUSTED_R2', self.adjusted_r2, None, None),
            ('PMML_EXPORT', self.pmml_export, None, None),
            ('HAS_ID', key is not None, None, None),
        ]

        #coef_spec = [
        #    ('VARIABLE_NAME', NVARCHAR(1000)),
        #    ('COEFFICIENT_VALUE', DOUBLE),
        #]
        #pmml_spec = [
        #    ('ROW_INDEX', INTEGER),
        #    ('MODEL_CONTENT', NVARCHAR(5000)),
        #]
        #fitted_spec = [
        #    # This doesn't make much sense when key is None, but it
        #    # matches what PAL does.
        #    parse_one_dtype(data.dtypes()[0]),
        #    ('VALUE', DOUBLE),
        #]
        #statistics_spec = [
        #    ('STAT_NAME', NVARCHAR(256)),
        #    ('STAT_VALUE', NVARCHAR(1000)),
        #]

        try:
            self._call_pal_auto('PAL_POLYNOMIAL_REGRESSION',
                                data,
                                ParameterTable().with_data(param_rows),
                                coef_tbl,
                                pmml_tbl,
                                fitted_tbl,
                                stats_tbl,
                                opt_param_tbl)
        except dbapi.Error as db_err:
            #msg = 'HANA error while attempting to fit polynomial regression model.'
            logger.exception(str(db_err))
            self._try_drop(outputs)
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

            Independent variable values used for prediction.

        key : str
            Name of the ID column.

        features : list of str, optional

            Names of the feature columns. Since the underlying
            PAL_POLYNOMIAL_REGRESSION only supports one feature,
            this list can only contain one element.
            If ``features`` is not provided, ``data`` must have exactly 1
            non-ID column, and ``features`` defaults to that column.

        Returns
        -------

        DataFrame

            Predicted values, structured as follows:

                - ID column, with same name and type as ``data``'s ID column.
                - VALUE, type DOUBLE, representing predicted values.

            .. note::

                predict() will pass the ``pmml_`` table to PAL as the model
                representation if there is a ``pmml_`` table, or the ``coefficients_``
                table otherwise.
        """
        # pylint: disable=too-many-locals

        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)

        if getattr(self, 'pmml_', None) is not None:
            model = self.pmml_
            model_format = 1
        elif hasattr(self, 'coefficients_'):
            model = self.coefficients_
            model_format = 0
        else:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")

        if features is None:
            cols = data.columns
            cols.remove(key)
            features = cols
        if len(features) != 1:
            msg = ('PAL polynomial regression requires exactly one ' +
                   'feature column.')
            logger.error(msg)
            raise TypeError(msg)

        data = data[[key] + features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()

        fitted_tbl = '#PAL_POLYNOMIAL_REGRESSION_FITTED_TBL_{}_{}'.format(self.id, unique_id)

        param_rows = [
            ('THREAD_RATIO', None, self.thread_ratio, None),
            ('MODEL_FORMAT', model_format, None, None),
        ]


        try:
            self._call_pal_auto('PAL_POLYNOMIAL_REGRESSION_PREDICT',
                                data,
                                model,
                                ParameterTable().with_data(param_rows),
                                fitted_tbl)
        except dbapi.Error as db_err:
            #msg = 'HANA error during polynomial regression prediction.'
            logger.exception(str(db_err))
            raise

        return self.conn_context.table(fitted_tbl)

    def score(self, data, key, features=None, label=None):
        r"""
        Returns the coefficient of determination R \ :sup:`2`\  of the prediction.

        Parameters
        ----------

        data : DataFrame

            Data on which to assess model performance.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns. Since the underlying
            PAL_POLYNOMIAL_REGRESSION_PREDICT only supports one feature,
            this list can only contain one element.
            If ``features`` is not provided, ``data`` must have exactly 1
            non-ID, non-label column, and ``features`` defaults to that
            column.

        label : str, optional

            Name of the dependent variable. Defaults to the last column.
            (This is not the PAL default.)

        Returns
        -------

        accuracy : float

            The coefficient of determination R\:sup:`2`\  of the prediction on the
            given data.
        """

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

        if len(features) != 1:
            msg = ('PAL polynomial regression requires exactly one ' +
                   'feature column.')
            logger.error(msg)
            raise TypeError(msg)

        prediction = self.predict(data, key=key, features=features)
        prediction = prediction.rename_columns(['ID_P', 'PREDICTION'])

        actual = data.select(key, label).rename_columns(['ID_A', 'ACTUAL'])

        joined = actual.join(prediction, 'ID_P=ID_A').select(
            'ACTUAL', 'PREDICTION')
        return metrics.r2_score(self.conn_context,
                                joined,
                                label_true='ACTUAL',
                                label_pred='PREDICTION')

class GLM(PALBase):
    r"""
    Regression by a generalized linear model, based on PAL_GLM. Also supports
    ordinal regression.

    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA system.

    family : {'gaussian', 'normal', 'poisson', 'binomial', 'gamma', 'inversegaussian', 'negativebinomial', 'ordinal'}, optional

        The kind of distribution the dependent variable outcomes are
        assumed to be drawn from.
        Defaults to 'gaussian'.

    link : str, optional

        GLM link function. Determines the relationship between the linear
        predictor and the predicted response. Default and allowed values
        depend on ``family``. 'inverse' is accepted as a synonym of
        'reciprocal'.

        ================ ============= ========================================
        family           default link  allowed values of link
        ================ ============= ========================================
        gaussian         identity      identity, log, reciprocal
        poisson          log           identity, log
        binomial         logit         logit, probit, comploglog, log
        gamma            reciprocal    identity, reciprocal, log
        inversegaussian  inversesquare inversesquare, identity, reciprocal, log
        negativebinomial log           identity, log, sqrt
        ordinal          logit         logit, probit, comploglog
        ================ ============= ========================================

    solver : {'irls', 'nr', 'cd'}, optional

        Optimization algorithm to use.

            - 'irls': Iteratively re-weighted least squares.
            - 'nr': Newton-Raphson.
            - 'cd': Coordinate descent. (Picking coordinate descent activates
              elastic net regularization.)

        Defaults to 'irls', except when ``family`` is 'ordinal'.
        Ordinal regression requires (and defaults to) 'nr', and Newton-Raphson
        is not supported for other values of ``family``.

    handle_missing_fit : {'skip', 'abort', 'fill_zero'}, optional

        How to handle data rows with missing independent variable values
        during fitting.

            - 'skip': Don't use those rows for fitting.
            - 'abort': Throw an error if missing independent variable values
              are found.
            - 'fill_zero': Replace missing values with 0.

        Defaults to 'skip'.

    quasilikelihood : bool, optional

        If True, enables the use of quasi-likelihood to estimate overdispersion.

        Defaults to False.

    max_iter : int, optional

        Maximum number of optimization iterations.

        Defaults to 100 for IRLS
        and Newton-Raphson.

        Defaults to 100000 for coordinate descent.

    tol : float, optional

        Stopping condition for optimization.

        Defaults to 1e-8 for IRLS,
        1e-6 for Newton-Raphson, and 1e-7 for coordinate descent.

    significance_level : float, optional

        Significance level for confidence intervals and prediction intervals.

        Defaults to 0.05.

    output_fitted : bool, optional

        If True, create the ``fitted_`` DataFrame of fitted response values
        for training data in fit.

    alpha : float, optional

        Elastic net mixing parameter. Only accepted when using coordinate
        descent. Should be between 0 and 1 inclusive.

        Defaults to 1.0.

    num_lambda : int, optional

        The number of lambda values. Only accepted when using coordinate
        descent.

        Defaults to 100.

    lambda_min_ratio : float, optional

        The smallest value of lambda, as a fraction of the maximum lambda,
        where lambda_max is the smallest value for which all coefficients
        are zero. Only accepted when using coordinate descent.

        Defaults to 0.01 when the number of observations is smaller than the number
        of covariates, and 0.0001 otherwise.

    categorical_variable : list of str, optional

        INTEGER columns specified in this list will be treated as categorical
        data. Other INTEGER columns will be treated as continuous.

    ordering : list of str or list of int, optional

        Specifies the order of categories for ordinal regression.
        The default is numeric order for ints and alphabetical order for
        strings.

    Attributes
    ----------

    statistics_ : DataFrame

        Training statistics and model information other than the
        coefficients and covariance matrix.

    coef_ : DataFrame

        Model coefficients.

    covmat_ : DataFrame

        Covariance matrix. Set to None for coordinate descent.

    fitted_ : DataFrame

        Predicted values for the training data. Set to None if
        ``output_fitted`` is False.

    Examples
    --------
    Training data:

    >>> df.collect()
       ID  Y  X
    0   1  0 -1
    1   2  0 -1
    2   3  1  0
    3   4  1  0
    4   5  1  0
    5   6  1  0
    6   7  2  1
    7   8  2  1
    8   9  2  1

    Fitting a GLM on that data:

    >>> glm = GLM(cc, solver='irls', family='poisson', link='log')
    >>> glm.fit(df, key='ID', label='Y')

    Performing prediction:

    >>> df2.collect()
       ID  X
    0   1 -1
    1   2  0
    2   3  1
    3   4  2
    >>> glm.predict(df2, key='ID')[['ID', 'PREDICTION']].collect()
       ID           PREDICTION
    0   1  0.25543735346197155
    1   2    0.744562646538029
    2   3   2.1702915689746476
    3   4     6.32608352871737
    """

    family_link_values = {
        'gaussian': ['identity', 'log', 'reciprocal', 'inverse'],
        'normal': ['identity', 'log', 'reciprocal', 'inverse'],
        'poisson': ['identity', 'log'],
        'binomial': ['logit', 'probit', 'comploglog', 'log'],
        'gamma': ['identity', 'reciprocal', 'inverse', 'log'],
        'inversegaussian': ['inversesquare', 'identity', 'reciprocal',
                            'inverse', 'log'],
        'negativebinomial': ['identity', 'log', 'sqrt'],
        'ordinal': ['logit', 'probit', 'comploglog']
    }
    solvers = ['irls', 'nr', 'cd']
    handle_missing_fit_map = {
        'abort': 0,
        'skip': 1,
        'fill_zero': 2
    }
    handle_missing_predict_map = {
        'skip': 1,
        'fill_zero': 2
    }
    def __init__(self,
                 conn_context,
                 family=None,
                 link=None,
                 solver=None,
                 handle_missing_fit=None,
                 quasilikelihood=None,
                 max_iter=None,
                 tol=None,
                 significance_level=None,
                 output_fitted=None,
                 alpha=None,
                 num_lambda=None,
                 lambda_min_ratio=None,
                 categorical_variable=None,
                 ordering=None):
        # pylint:disable=too-many-arguments
        # pylint:disable=too-many-locals
        # pylint:disable=too-many-branches
        super(GLM, self).__init__(conn_context)
        self.family = self._arg('family',
                                family,
                                {x:x for x in self.family_link_values})
        links = self.family_link_values['gaussian' if self.family is None
                                        else self.family]
        self.link = self._arg('link',
                              link,
                              {x:x for x in links})
        self.solver = self._arg('solver',
                                solver,
                                {x:x for x in self.solvers})
        self.handle_missing_fit = self._arg('handle_missing_fit',
                                            handle_missing_fit,
                                            self.handle_missing_fit_map)
        self.quasilikelihood = self._arg('quasilikelihood',
                                         quasilikelihood,
                                         bool)
        self.max_iter = self._arg('max_iter', max_iter, int)
        self.tol = self._arg('tol', tol, float)
        self.significance_level = self._arg('significance_level',
                                            significance_level,
                                            float)
        self.output_fitted = self._arg('output_fitted',
                                       output_fitted,
                                       bool)
        if self.solver != 'cd':
            if alpha is not None:
                bad = 'alpha'
            elif num_lambda is not None:
                bad = 'num_lambda'
            elif lambda_min_ratio is not None:
                bad = 'lambda_min_ratio'
            else:
                bad = None
            if bad is not None:
                msg = ("Parameter {} should not be provided when solver " +
                       "is not 'cd'.").format(bad)
                logger.error(msg)
                raise ValueError(msg)
        self.alpha = self._arg('alpha', alpha, float)
        self.num_lambda = self._arg('num_lambda', num_lambda, int)
        self.lambda_min_ratio = self._arg('lambda_min_ratio',
                                          lambda_min_ratio,
                                          float)
        self.categorical_variable = categorical_variable
        if ordering is None:
            self.ordering = ordering
        elif not ordering:
            msg = 'ordering should be nonempty.'
            logger.error(msg)
            raise ValueError(msg)
        elif all(isinstance(val, _INTEGER_TYPES) for val in ordering):
            self.ordering = ', '.join(map(str, ordering))
        elif all(isinstance(val, _STRING_TYPES) for val in ordering):
            for value in ordering:
                if ',' in value or value.strip() != value:
                    # I don't know whether this check is enough, but it
                    # covers the cases I've found to be problematic in
                    # testing.
                    # The PAL docs don't say anything about escaping.
                    msg = ("Can't have commas or leading/trailing spaces"
                           + " in the elements of the ordering list.")
                    logger.error(msg)
                    raise ValueError(msg)
            self.ordering = ', '.join(ordering)
        else:
            msg = 'ordering should be a list of ints or a list of strings.'
            logger.error(msg)
            raise ValueError(msg)

    def fit(self, data, key=None, features=None, label=None, categorical_variable=None, dependent_variable=None, excluded_feature=None):
        r"""
        Fit a generalized linear model based on training data.

        Parameters
        ----------

        data : DataFrame

            Training data.

        key : str, optional

            Name of the ID column. If ``key`` is not provided, it is assumed
            that the input has no ID column.
            Required when ``output_fitted`` is True.

        features : list of str, optional

            Names of the feature columns.

            Defaults to all non-ID, non-label
            columns.

        label : str or list of str, optional

            Name of the dependent variable. Defaults to the last column.
            (This is not the PAL default.)
            When ``family`` is 'binomial', ``label`` may be either a single
            column name or a list of two column names.

        categorical_variable : list of str, optional

            INTEGER columns specified in this list will be treated as categorical
            data. Other INTEGER columns will be treated as continuous.

        dependent_variable : str, optional

            Only used when you need to indicate the dependence.

        excluded_feature : list of str, optional

            Excludes the indicated feature column.

            Defaults to None.

        """

        # pylint:disable=too-many-locals
        # pylint:disable=too-many-statements

        key = self._arg('key', key, str)
        features = self._arg('features', features, ListOfStrings)
        # label requires more complex check.
        categorical_variable = self._arg('categorical_variable', categorical_variable, ListOfStrings)
        dependent_variable = self._arg('dependent_variable', dependent_variable, str)
        excluded_feature = self._arg('excluded_feature', excluded_feature, ListOfStrings)
        if label is not None and not isinstance(label, _STRING_TYPES):
            if self.family != 'binomial':
                msg = ("When family is not 'binomial', "
                       + "label must be a single string.")
                logger.error(msg)
                raise TypeError(msg)
            if (not isinstance(label, list)
                    or len(label) != 2
                    or not all(isinstance(elem, _STRING_TYPES) for elem in label)):
                msg = "A non-string label must be a list of two strings."
                logger.error(msg)
                raise TypeError(msg)

        if key is None and self.output_fitted:
            msg = 'A key column is required when output_fitted is True.'
            logger.error(msg)
            raise TypeError(msg)

        cols_left = data.columns
        if key is None:
            maybe_id = []
        else:
            maybe_id = [key]
            cols_left.remove(key)
        if label is None:
            label = cols_left[-1]
        if isinstance(label, _STRING_TYPES):
            label = [label]
        for column in label:
            cols_left.remove(column)
        if features is None:
            features = cols_left

        data = data[maybe_id + label + features]

        outputs = stat_tbl, coef_tbl, covmat_tbl, fit_tbl = [
            '#GLM_{}_TBL_{}'.format(tab, self.id)
            for tab in ['STAT', 'COEF', 'COVMAT', 'FIT']]

        param_rows = [
            ('SOLVER', None, None, self.solver),
            ('FAMILY', None, None, self.family),
            ('LINK', None, None, self.link),
            ('HANDLE_MISSING', self.handle_missing_fit, None, None),
            ('QUASI', self.quasilikelihood, None, None),
            ('GROUP_RESPONSE', len(label) == 2, None, None),
            ('MAX_ITERATION', self.max_iter, None, None),
            ('CONVERGENCE_CRITERION', None, self.tol, None),
            ('SIGNIFICANCE_LEVEL', None, self.significance_level, None),
            ('OUTPUT_FITTED', self.output_fitted, None, None),
            ('ENET_ALPHA', None, self.alpha, None),
            ('ENET_NUM_LAMBDA', self.num_lambda, None, None),
            ('LAMBDA_MIN_RATIO', None, self.lambda_min_ratio, None),
            ('HAS_ID', key is not None, None, None),
            ('ORDERING', None, None, self.ordering),
            ('DEPENDENT_VARIABLE', None, None, dependent_variable)
        ]

        if self.categorical_variable is not None:
            param_rows.extend(('CATEGORICAL_VARIABLE', None, None, variable)
                              for variable in self.categorical_variable)
        if categorical_variable is not None:
            param_rows.extend(('CATEGORICAL_VARIABLE', None, None, variable)
                              for variable in categorical_variable)
        if excluded_feature is not None:
            param_rows.extend(('EXCLUDED_FEATURE', None, None, exc)
                              for exc in excluded_feature)

        try:
            param_t = ParameterTable(name='#PARAM_TBL')
            self._call_pal_auto('PAL_GLM',
                                data,
                                param_t.with_data(param_rows),
                                *outputs)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            self._try_drop('#PARAM_TBL')
            self._try_drop(outputs)
            raise

        # pylint: disable=attribute-defined-outside-init
        self.statistics_ = self.conn_context.table(stat_tbl)
        self.coef_ = self.conn_context.table(coef_tbl)

        # For coordinate descent, this table is empty, but PAL_GLM_PREDICT
        # still wants it.
        self._covariance = self.conn_context.table(covmat_tbl)
        self.covariance_ = self._covariance if self.solver != 'cd' else None

        self.fitted_ = (self.conn_context.table(fit_tbl) if self.output_fitted
                        else None)

    def predict(self, # pylint: disable=too-many-arguments
                data,
                key,
                features=None,
                prediction_type=None,
                significance_level=None,
                handle_missing=None):
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

            Defaults to all non-ID columns.

        prediction_type : {'response', 'link'}, optional

            Specifies whether to output predicted values of the
            response or the link function.

            Defaults to 'response'.

        significance_level : float, optional

            Significance level for confidence intervals and prediction
            intervals. If specified, overrides the value passed to the
            GLM constructor.

        handle_missing : {'skip', 'fill_zero'}, optional

            How to handle data rows with missing independent variable values.

                - 'skip': Don't perform prediction for those rows.
                - 'fill_zero': Replace missing values with 0.

            Defaults to 'skip'.

        Returns
        -------

        DataFrame

            Predicted values, structured as follows. The following two
            columns are always populated:

                - ID column, with same name and type as ``data``'s ID column.
                - PREDICTION, type NVARCHAR(100), representing predicted values.

            The following five columns are only populated for IRLS:

                - SE, type DOUBLE. Standard error, or for ordinal regression, \
                  the probability that the data point belongs to the predicted \
                  category.
                - CI_LOWER, type DOUBLE. Lower bound of the confidence interval.
                - CI_UPPER, type DOUBLE. Upper bound of the confidence interval.
                - PI_LOWER, type DOUBLE. Lower bound of the prediction interval.
                - PI_UPPER, type DOUBLE. Upper bound of the prediction interval.
        """
        # pylint:disable=too-many-locals
        if not all(hasattr(self, attr)
                   for attr in ('statistics_', 'coef_', 'covariance_')):
            raise FitIncompleteError("Model not initialized. Perform a fit first.")

        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        prediction_type = self._arg('prediction_type', prediction_type,
                                    {'response': 'response', 'link': 'link'})
        significance_level = self._arg('significance_level', significance_level, float)
        if significance_level is None:
            significance_level = self.significance_level
        handle_missing = self._arg('handle_missing',
                                   handle_missing,
                                   self.handle_missing_predict_map)

        if features is None:
            features = data.columns
            features.remove(key)
        data = data[[key] + features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()

        result_tbl = '#PAL_GLM_RESULT_TBL_{}_{}'.format(self.id, unique_id)

        param_rows = [
            ('TYPE', None, None, prediction_type),
            ('SIGNIFICANCE_LEVEL', None, significance_level, None),
            ('HANDLE_MISSING', handle_missing, None, None),
        ]

        try:
            param_t = ParameterTable(name='#PARAM_TBL')
            self._call_pal_auto('PAL_GLM_PREDICT',
                                data,
                                self.statistics_,
                                self.coef_,
                                self._covariance,
                                param_t.with_data(param_rows),
                                result_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            self._try_drop('#PARAM_TBL')
            self._try_drop(result_tbl)
            raise

        return self.conn_context.table(result_tbl)

    def score(self, # pylint: disable=too-many-arguments
              data,
              key,
              features=None,
              label=None,
              prediction_type=None,
              handle_missing=None):
        r"""
        Returns the coefficient of determination R\ :sup:`2`\  of the prediction.

        Not applicable for ordinal regression.

        Parameters
        ----------

        data : DataFrame

            Data on which to assess model performance.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.

            Defaults to all non-ID, non-label
            columns.

        label : str, optional

            Name of the dependent variable.

            Defaults to the last column.
            (This is not the PAL default.)
            Cannot be two columns, even for family='binomial'.

        prediction_type : {'response', 'link'}, optional

            Specifies whether to predict the value of the
            response or the link function. The contents of the ``label``
            column should match this choice.

            Defaults to 'response'.

        handle_missing : {'skip', 'fill_zero'}, optional

            How to handle data rows with missing independent variable values.

            - 'skip': Don't perform prediction for those rows. Those rows
              will be left out of the R\ :sup:`2`\  computation.
            - 'fill_zero': Replace missing values with 0.

            Defaults to 'skip'.

        Returns
        -------

        accuracy : float

            The coefficient of determination R \:sup:`2`\  of the prediction on the
            given data.
        """
        if self.family == 'ordinal':
            msg = "Can't compute R^2 for ordinal regression."
            logger.error(msg)
            raise TypeError(msg)

        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        # leaving prediction_type and handle_missing to predict()

        cols = data.columns
        cols.remove(key)
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols

        prediction = self.predict(data, key=key, features=features,
                                  prediction_type=prediction_type,
                                  handle_missing=handle_missing)
        prediction = prediction.select(key, 'PREDICTION')
        prediction = prediction.cast('PREDICTION', 'DOUBLE')
        prediction = prediction.rename_columns(['ID_P', 'PREDICTION'])

        actual = data.select(key, label).rename_columns(['ID_A', 'ACTUAL'])

        joined = actual.join(prediction, 'ID_P=ID_A').select(
            'ID_A', 'ACTUAL', 'PREDICTION')
        return metrics.r2_score(self.conn_context,
                                joined,
                                label_true='ACTUAL',
                                label_pred='PREDICTION')

class ExponentialRegression(PALBase):
    r"""
    Exponential regression is an approach to modeling the relationship between a scalar variable y and one or more variables denoted X. In exponential regression,
    data is modeled using exponential functions, and unknown model parameters are estimated from the data. Such models are called exponential models.

    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA system.

    decomposition : {'LU', 'SVD'}, optional

        Matrix factorization type to use. Case-insensitive.

          - 'LU': LU decomposition.
          - 'SVD': singular value decomposition.

        Defaults to LU decomposition.

    adjusted_r2 : boolean, optional

        If true, include the adjusted R \:sup:`2`\  value in the statistics table.

        Defaults to False.

    pmml_export : {'no', 'single-row', 'multi-row'}, optional

        Controls whether to output a PMML representation of the model,
        and how to format the PMML. Case-insensitive.

          - 'no' or not provided: No PMML model.

          - 'single-row': Exports a PMML model in a maximum of
            one row. Fails if the model doesn't fit in one row.

          - 'multi-row': Exports a PMML model, splitting it
            across multiple rows if it doesn't fit in one.

        Prediction does not require a PMML model.

    thread_ratio : float, optional

        Controls the proportion of available threads to use for prediction.
        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads. Values between 0 and 1
        will use that percentage of available threads. Values outside this
        range tell PAL to heuristically determine the number of threads to use.
        Does not affect fitting.

        Defaults to 0.

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

    >>> df.collect()
       ID    Y       X1      X2
       0    0.5     0.13    0.33
       1    0.15    0.14    0.34
       2    0.25    0.15    0.36
       3    0.35    0.16    0.35
       4    0.45    0.17    0.37

    Training the model:

    >>> er = ExponentialRegression(pmml_export = 'multi-row')
    >>> er.fit(df, key='ID')

    Prediction:

    >>> df2.collect()
       ID    X1       X2
       0    0.5      0.3
       1    4        0.4
       2    0        1.6
       3    0.3      0.45
       4    0.4      1.7

    >>> er.predict(df2, key='ID').collect()
       ID      VALUE
       0      0.6900598931338715
       1      1.2341502316656843
       2      0.006630664136180741
       3      0.3887970208571841
       4      0.0052106543571450266
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    decomposition_map = {'lu': 0, 'svd': 2}
    pmml_export_map = {'no': 0, 'single-row': 1, 'multi-row': 2}
    def __init__(self,
                 conn_context,
                 decomposition=None,
                 adjusted_r2=None,
                 pmml_export=None,
                 thread_ratio=None):
        super(ExponentialRegression, self).__init__(conn_context)
        self.decomposition = self._arg('decomposition', decomposition, self.decomposition_map)
        self.adjusted_r2 = self._arg('adjusted_r2', adjusted_r2, bool)
        self.pmml_export = self._arg('pmml_export', pmml_export, self.pmml_export_map)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)

    def fit(self, data, key=None, features=None, label=None):
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
        label : str, optional
            Name of the dependent variable.

            Defaults to the last column.
            (This is not the PAL default.)
        """
        # pylint: disable=too-many-locals

        key = self._arg('key', key, str)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)

        cols = data.columns
        if key is not None:
            maybe_id = [key]
            cols.remove(key)
        else:
            maybe_id = []
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols
        str1 = ','.join(str(e) for e in features)
        data = data[maybe_id + [label] + features]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        tables = ['PARAM', 'COEF', 'PMML', 'FITTED', 'STATS']
        tables = ['#PAL_EXPONENTIAL_REGRESSION_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                  for name in tables]
        param_tbl, coef_tbl, pmml_tbl, fitted_tbl, stats_tbl = tables

        param_rows = [
            ('ALG', self.decomposition, None, None),
            ('ADJUSTED_R2', self.adjusted_r2, None, None),
            ('PMML_EXPORT', self.pmml_export, None, None),
            ('SELECTED_FEATURES', None, None, str1),
            ('DEPENDENT_VARIABLE', None, None, label),
            ('HAS_ID', key is not None, None, None),
        ]
        try:
            self._call_pal_auto('PAL_EXPONENTIAL_REGRESSION',
                                data,
                                ParameterTable(param_tbl).with_data(param_rows),
                                coef_tbl,
                                fitted_tbl,
                                stats_tbl,
                                pmml_tbl)
        except dbapi.Error as db_err:
            #msg = 'HANA error while attempting to fit exponential regression model.'
            logger.exception(str(db_err))
            self._try_drop(tables)
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

            Independent variable values used for prediction.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.

        Returns
        -------

        DataFrame

            Predicted values, structured as follows:

                - ID column, with same name and type as ``data`` 's ID column.
                - VALUE, type DOUBLE, representing predicted values.

            .. note::

                predict() will pass the ``pmml_`` table to PAL as the model
                representation if there is a ``pmml_`` table, or the ``coefficients_``
                table otherwise.
        """
        # pylint: disable=too-many-locals

        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        if getattr(self, 'pmml_', None) is not None:
            model = self.pmml_
            model_format = 1
        elif hasattr(self, 'coefficients_'):
            model = self.coefficients_
            model_format = 0
        else:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")

        if features is None:
            cols = data.columns
            cols.remove(key)
            features = cols

        data = data[[key] + features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()

        tables = param_tbl, fitted_tbl = [
            '#PAL_EXPONENTIAL_REGRESSION_{}_TBL_{}_{}'.format(name, self.id, unique_id)
            for name in ['PARAM', 'FITTED']]

        param_rows = [
            ('THREAD_RATIO', None, self.thread_ratio, None),
            ('MODEL_FORMAT', model_format, None, None),
        ]

        try:
            self._call_pal_auto('PAL_EXPONENTIAL_REGRESSION_PREDICT',
                                data,
                                model,
                                ParameterTable(param_tbl).with_data(param_rows),
                                fitted_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            self._try_drop(tables)
            raise

        return self.conn_context.table(fitted_tbl)

    def score(self, data, key, features=None, label=None):
        r"""
        Returns the coefficient of determination R\ :sup:`2`\  of the prediction.

        Parameters
        ----------

        data : DataFrame

            Data on which to assess model performance.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.

        label : str, optional

            Name of the dependent variable.

            Defaults to the last column.
            (This is not the PAL default.)

        Returns
        -------

        accuracy : float

            The coefficient of determination R \:sup:`2`\  of the prediction on the
            given data.
        """

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

        actual = data.select(key, label).rename_columns(['ID_A', 'ACTUAL'])

        joined = actual.join(prediction, 'ID_P=ID_A').select(
            'ACTUAL', 'PREDICTION')
        return metrics.r2_score(self.conn_context,
                                joined,
                                label_true='ACTUAL',
                                label_pred='PREDICTION')

class BiVariateGeometricRegression(PALBase):
    r"""
    Geometric regression is an approach used to model the relationship between a scalar variable y and a variable denoted X. In geometric regression,
    data is modeled using geometric functions, and unknown model parameters are estimated from the data. Such models are called geometric models.

    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA system.

    decomposition : {'LU', 'SVD'}, optional

        Matrix factorization type to use. Case-insensitive.

          - 'LU': LU decomposition.
          - 'SVD': singular value decomposition.

        Defaults to LU decomposition.

    adjusted_r2 : bool, optional

        If true, include the adjusted R \:sup:`2`\  value in the statistics table.

        Defaults to False.

    pmml_export : {'no', 'single-row', 'multi-row'}, optional

        Controls whether to output a PMML representation of the model,
        and how to format the PMML. Case-insensitive.

          - 'no' or not provided: No PMML model.

          - 'single-row': Exports a PMML model in a maximum of
            one row. Fails if the model doesn't fit in one row.

          - 'multi-row': Exports a PMML model, splitting it
            across multiple rows if it doesn't fit in one.

        Prediction does not require a PMML model.

    thread_ratio : float, optional

        Controls the proportion of available threads to use for prediction.
        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads. Values between 0 and 1
        will use that percentage of available threads. Values outside this
        range tell PAL to heuristically determine the number of threads to use.
        Does not affect fitting.

        Defaults to 0.

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

    >>> df.collect()
    ID    Y       X1
    0    1.1      1
    1    4.2      2
    2    8.9      3
    3    16.3     4
    4    24       5


    Training the model:

    >>> gr = BiVariateGeometricRegression(pmml_export = 'multi-row')
    >>> gr.fit(df, key='ID')

    Prediction:

    >>> df2.collect()
    ID    X1
    0     1
    1     2
    2     3
    3     4
    4     5

    >>> er.predict(df2, key='ID').collect()
    ID      VALUE
    0        1
    1       3.9723699817481437
    2       8.901666037549536
    3       15.779723271893747
    4       24.60086108408644
"""

    # pylint: disable=too-many-arguments,too-many-instance-attributes

    decomposition_map = {'lu': 0, 'svd': 2}
    pmml_export_map = {'no': 0, 'single-row': 1, 'multi-row': 2}
    def __init__(self,
                 conn_context,
                 decomposition=None,
                 adjusted_r2=None,
                 pmml_export=None,
                 thread_ratio=None):
        super(BiVariateGeometricRegression, self).__init__(conn_context)
        self.decomposition = self._arg('decomposition', decomposition, self.decomposition_map)
        self.adjusted_r2 = self._arg('adjusted_r2', adjusted_r2, bool)
        self.pmml_export = self._arg('pmml_export', pmml_export, self.pmml_export_map)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)

    def fit(self, data, key=None, features=None, label=None):
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

        label : str, optional

            Name of the dependent variable.

            Defaults to the last column.
            (This is not the PAL default.)
        """
        # pylint: disable=too-many-locals


        key = self._arg('key', key, str)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)

        cols = data.columns
        if key is not None:
            maybe_id = [key]
            cols.remove(key)
        else:
            maybe_id = []
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols
        str1 = ','.join(str(e) for e in features)
        data = data[maybe_id + [label] + features]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        tables = ['PARAM', 'COEF', 'PMML', 'FITTED', 'STATS']
        tables = ['#PAL_GEOMETRIC_REGRESSION_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                  for name in tables]
        param_tbl, coef_tbl, pmml_tbl, fitted_tbl, stats_tbl = tables

        param_rows = [
            ('ALG', self.decomposition, None, None),
            ('ADJUSTED_R2', self.adjusted_r2, None, None),
            ('PMML_EXPORT', self.pmml_export, None, None),
            ('SELECTED_FEATURES', None, None, str1),
            ('DEPENDENT_VARIABLE', None, None, label),
            ('HAS_ID', key is not None, None, None),
        ]
        try:
            self._call_pal_auto('PAL_GEOMETRIC_REGRESSION',
                                data,
                                ParameterTable(param_tbl).with_data(param_rows),
                                coef_tbl,
                                fitted_tbl,
                                stats_tbl,
                                pmml_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            self._try_drop(tables)
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

            Independent variable values used for prediction.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.

        Returns
        -------

        DataFrame

            Predicted values, structured as follows:

                - ID column, with same name and type as ``data`` 's ID column.
                - VALUE, type DOUBLE, representing predicted values.

            .. note::

                predict() will pass the ``pmml_`` table to PAL as the model
                representation if there is a ``pmml_`` table, or the ``coefficients_``
                table otherwise.
        """
        # pylint: disable=too-many-locals
        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        if getattr(self, 'pmml_', None) is not None:
            model = self.pmml_
            model_format = 1
        elif hasattr(self, 'coefficients_'):
            model = self.coefficients_
            model_format = 0
        else:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")

        if features is None:
            cols = data.columns
            cols.remove(key)
            features = cols

        data = data[[key] + features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()

        tables = param_tbl, fitted_tbl = [
            '#PAL_GEOMETRIC_REGRESSION_{}_TBL_{}_{}'.format(name, self.id, unique_id)
            for name in ['PARAM', 'FITTED']]

        param_rows = [
            ('THREAD_RATIO', None, self.thread_ratio, None),
            ('MODEL_FORMAT', model_format, None, None),
        ]

        try:
            self._call_pal_auto('PAL_GEOMETRIC_REGRESSION_PREDICT',
                                data,
                                model,
                                ParameterTable(param_tbl).with_data(param_rows),
                                fitted_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            self._try_drop(tables)
            raise

        return self.conn_context.table(fitted_tbl)

    def score(self, data, key, features=None, label=None):
        r"""
        Returns the coefficient of determination R \ :sup:`2`\  of the prediction.

        Parameters
        ----------

        data : DataFrame

            Data on which to assess model performance.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.

        label : str, optional

            Name of the dependent variable.

            Defaults to the last column.
            (This is not the PAL default.)

        Returns
        -------

        accuracy : float

            The coefficient of determination R\ :sup:`2`\  of the prediction on the
            given data.
        """

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

        actual = data.select(key, label).rename_columns(['ID_A', 'ACTUAL'])

        joined = actual.join(prediction, 'ID_P=ID_A').select(
            'ACTUAL', 'PREDICTION')
        return metrics.r2_score(self.conn_context,
                                joined,
                                label_true='ACTUAL',
                                label_pred='PREDICTION')

class BiVariateNaturalLogarithmicRegression(PALBase):
    r"""
    Bi-variate natural logarithmic regression is an approach to modeling the relationship between a scalar variable y and one variable denoted X. In natural logarithmic regression,
    data is modeled using natural logarithmic functions, and unknown model parameters are estimated from the data.
    Such models are called natural logarithmic models.

    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA system.

    decomposition : {'LU', 'SVD'}, optional

        Matrix factorization type to use. Case-insensitive.

          - 'LU': LU decomposition.
          - 'SVD': singular value decomposition.

        Defaults to LU decomposition.

    adjusted_r2 : boolean, optional

        If true, include the adjusted R\ :sup:`2`\  value in the statistics table.

        Defaults to False.

    pmml_export : {'no', 'single-row', 'multi-row'}, optional

        Controls whether to output a PMML representation of the model,
        and how to format the PMML. Case-insensitive.

          - 'no' or not provided: No PMML model.

          - 'single-row': Exports a PMML model in a maximum of
            one row. Fails if the model doesn't fit in one row.

          - 'multi-row': Exports a PMML model, splitting it
            across multiple rows if it doesn't fit in one.

        Prediction does not require a PMML model.

    thread_ratio : float, optional

        Controls the proportion of available threads to use for prediction.
        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads. Values between 0 and 1
        will use that percentage of available threads. Values outside this
        range tell PAL to heuristically determine the number of threads to use.
        Does not affect fitting.

        Defaults to 0.

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

    >>> df.collect()
       ID    Y       X1
       0    10       1
       1    80       2
       2    130      3
       3    180      5
       4    190      6


    Training the model:

    >>> gr = BiVariateNaturalLogarithmicRegression(pmml_export = 'multi-row')
    >>> gr.fit(df, key='ID')

    Prediction:

    >>> df2.collect()
       ID    X1
       0     1
       1     2
       2     3
       3     4
       4     5



    >>> er.predict(df2, key='ID').collect()
       ID      VALUE
       0     14.86160299
       1     82.9935329364932
       2     122.8481570569525
       3     151.1254628829864
       4     173.05904529166017
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes

    decomposition_map = {'lu': 0, 'svd': 2}
    pmml_export_map = {'no': 0, 'single-row': 1, 'multi-row': 2}
    def __init__(self,
                 conn_context,
                 decomposition=None,
                 adjusted_r2=None,
                 pmml_export=None,
                 thread_ratio=None):
        super(BiVariateNaturalLogarithmicRegression, self).__init__(conn_context)
        self.decomposition = self._arg('decomposition', decomposition, self.decomposition_map)
        self.adjusted_r2 = self._arg('adjusted_r2', adjusted_r2, bool)
        self.pmml_export = self._arg('pmml_export', pmml_export, self.pmml_export_map)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)

    def fit(self, data, key=None, features=None, label=None):
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

        label : str, optional

            Name of the dependent variable.

            Defaults to the last column.
            (This is not the PAL default.)
        """
        # pylint: disable=too-many-locals

        key = self._arg('key', key, str)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)

        cols = data.columns
        if key is not None:
            maybe_id = [key]
            cols.remove(key)
        else:
            maybe_id = []
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols
        str1 = ','.join(str(e) for e in features)
        data = data[maybe_id + [label] + features]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        tables = ['PARAM', 'COEF', 'PMML', 'FITTED', 'STATS']
        tables = ['#PAL_LOGARITHMIC_REGRESSION_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                  for name in tables]
        param_tbl, coef_tbl, pmml_tbl, fitted_tbl, stats_tbl = tables

        param_rows = [
            ('ALG', self.decomposition, None, None),
            ('ADJUSTED_R2', self.adjusted_r2, None, None),
            ('PMML_EXPORT', self.pmml_export, None, None),
            ('SELECTED_FEATURES', None, None, str1),
            ('DEPENDENT_VARIABLE', None, None, label),
            ('HAS_ID', key is not None, None, None),
        ]
        try:
            self._call_pal_auto('PAL_LOGARITHMIC_REGRESSION',
                                data,
                                ParameterTable(param_tbl).with_data(param_rows),
                                coef_tbl,
                                fitted_tbl,
                                stats_tbl,
                                pmml_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            self._try_drop(tables)
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

            Independent variable values used for prediction.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.

        Returns
        -------

        DataFrame

            Predicted values, structured as follows:

                - ID column, with same name and type as ``data`` 's ID column.
                - VALUE, type DOUBLE, representing predicted values.

            .. note::

                predict() will pass the ``pmml_`` table to PAL as the model
                representation if there is a ``pmml_`` table, or the ``coefficients_``
                table otherwise.
        """
        # pylint: disable=too-many-locals

        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        if getattr(self, 'pmml_', None) is not None:
            model = self.pmml_
            model_format = 1
        elif hasattr(self, 'coefficients_'):
            model = self.coefficients_
            model_format = 0
        else:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")

        if features is None:
            cols = data.columns
            cols.remove(key)
            features = cols

        data = data[[key] + features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()

        tables = param_tbl, fitted_tbl = [
            '#PAL_LOGARITHMIC_REGRESSION_{}_TBL_{}_{}'.format(name, self.id, unique_id)
            for name in ['PARAM', 'FITTED']]

        param_rows = [
            ('THREAD_RATIO', None, self.thread_ratio, None),
            ('MODEL_FORMAT', model_format, None, None),
        ]

        try:
            self._call_pal_auto('PAL_LOGARITHMIC_REGRESSION_PREDICT',
                                data,
                                model,
                                ParameterTable(param_tbl).with_data(param_rows),
                                fitted_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            self._try_drop(tables)
            raise

        return self.conn_context.table(fitted_tbl)

    def score(self, data, key, features=None, label=None):
        r"""
        Returns the coefficient of determination R \ :sup:`2`\  of the prediction.

        Parameters
        ----------

        data : DataFrame

            Data on which to assess model performance.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.

        label : str, optional

            Name of the dependent variable.

            Defaults to the last column.
            (This is not the PAL default.)

        Returns
        -------

        accuracy : float

            The coefficient of determination R \ :sup:`2`\  of the prediction on the
            given data.
        """

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

        actual = data.select(key, label).rename_columns(['ID_A', 'ACTUAL'])

        joined = actual.join(prediction, 'ID_P=ID_A').select(
            'ACTUAL', 'PREDICTION')
        return metrics.r2_score(self.conn_context,
                                joined,
                                label_true='ACTUAL',
                                label_pred='PREDICTION')

class CoxProportionalHazardModel(PALBase):
    r"""
    Cox proportional hazard model (CoxPHM) is a special generalized linear model.
    It is a well-known realization-of-survival model that demonstrates failure or death at a certain time.

    Parameters
    ----------

    conn_context : ConnectionContext
        Connection to the HANA system.

    tie_method : {'breslow', 'efron'}, optional
        The method to deal with tied events.

        Defaults to 'efron'.
    status_col : bool, optional
        If a status column is defined for right-censored data:

        - 'False' : No status column. All response times are failure/death.
        - 'True' : The 3rd column of the data input table is a status column,
                 of which 0 indicates right-censored data and 1 indicates
                 failure/death.

        Defaults to 'True'.

    max_iter : int, optional
        Maximum number of iterations for numeric optimization.

    convergence_criterion : float, optional
        Convergence criterion of coefficients for numeric optimization.

        Defaults to 0.

    significance_level : float, optional
        Significance level for the confidence interval of estimated coefficients.

        Defaults to 0.05.

    calculate_hizard : bool, optional
        Controls whether to calculate hazard function as well as survival function.

        - 'False' : Does not calculate hazard function.
        - 'True': Calculates hazard function.

        Defaults to 'True'.

    output_fitted : bool, optional
        Controls whether to output the fitted response:

        - 'False' : Does not output the fitted response.
        - 'True': Outputs the fitted response.

        Defaults to 'False'.

    type_kind : str, optional
        The prediction type:

        - 'risk': Predicts in risk space
        - 'lp': Predicts in linear predictor space

        Default Value 'risk'

    Attributes
    ----------

    statistics_ : DataFrame
        Regression-related statistics, such as r-square, log-likelihood, aic.

    coefficient\_ : DataFrame
        Fitted regression coefficients.

    covariance_variance : DataFrame
        Co-Variance related data.

    hazard\_ : DataFrame
        Statistics related to Time, Hazard, Survival.

    fitted\_ : DataFrame
        Predicted dependent variable values for training data.
        Set to None if the training data has no row IDs.

    Examples
    ----------

    >>> df1.collect()
        ID	TIME	STATUS	X1	X2
        1	  4	         1	 0	 0
        2	  3	         1	 2 	 0
        3	  1	         1	 1	 0
        4	  1	         0	 1	 0
        5	  2	         1	 1	 1
        6	  2	         1	 0	 1
        7	  3	         0	 0	 1


    Training the model:

    >>> cox = CoxProportionalHazardModel(conn_context=connection_context,
    significance_level= 0.05, calculate_hazard='yes', type_kind='risk')
    >>> cox.fit(df1, key='ID', features=['STATUS', 'X1', 'X2'], label='TIME')

    Prediction:

    >>> df2.collect()
        ID	X1	X2
        1	0	0
        2	2	0
        3	1	0
        4	1	0
        5	1	1
        6	0	1
        7	0	1

    >>> cox.predict(full_tbl, key='ID',features=['STATUS', 'X1', 'X2']).collect()
        ID	 PREDICTION	   SE	      CI_LOWER	   CI_UPPER
        1	0.383590423	0.412526262	0.046607574	3.157032199
        2	1.829758442	1.385833778	0.414672719	8.073875617
        3	0.837781484	0.400894077	0.32795551	2.140161678
        4	0.837781484	0.400894077	0.32795551	2.140161678

    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    map = {'no':0, 'yes': 1}

    def __init__(self,
                 conn_context,
                 tie_method=None,
                 status_col=None,
                 max_iter=None,
                 convergence_criterion=None,
                 significance_level=None,
                 calculate_hazard=None,
                 output_fitted=None,
                 type_kind=None):
        super(CoxProportionalHazardModel, self).__init__(conn_context)
        self.tie_method = self._arg('tie_method', tie_method, str)
        self.status_col = self._arg('status_col', status_col, bool)
        self.max_iter = self._arg('max_iter', max_iter, int)
        self.convergence_criterion = self._arg('convergence_criterion', convergence_criterion, float)
        self.significance_level = self._arg('significance_level', significance_level, float)
        self.calculate_hazard = self._arg('calculate_hazard', calculate_hazard, bool)
        self.output_fitted = self._arg('output_fitted', output_fitted, bool)
        self.type_kind = self._arg('type_kind', type_kind, str)

    def fit(self, data, key=None, features=None, label=None):
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

        label : str, optional
            Name of the dependent variable.

            Defaults to the last column.
            (This is not the PAL default.)
        """

        # pylint: disable=too-many-locals
        key = self._arg('key', key, str)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)

        cols = data.columns
        if key is not None:
            maybe_id = [key]
            cols.remove(key)
        else:
            maybe_id = []
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols
        data = data[maybe_id + [label] + features]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        tables = ['PARAM', 'COEF', 'CO_VARIANCE', 'FITTED', 'STATS', 'HAZARD']
        tables = ['#PAL_COXPH_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                  for name in tables]
        param_tbl, coef_tbl, co_variance_tbl, fitted_tbl, stats_tbl, hazard_tbl = tables

        param_rows = [
            ('TIE_METHOD', None, None, self.tie_method),
            ('STATUS_COL', self.status_col, None, None),
            ('MAX_ITERATION', self.max_iter, None, None),
            ('CONVERGENCE_CRITERION', self.convergence_criterion, None, None),
            ('SIGNIFICANCE_LEVEL', None, self.significance_level, None),
            ('CALCULATE_HAZARD', self.calculate_hazard, None, None),
            ('OUTPUT_FITTED', self.output_fitted, None, None),
        ]
        try:
            self._call_pal_auto('PAL_COXPH',
                                data,
                                ParameterTable(param_tbl).with_data(param_rows),
                                stats_tbl,
                                coef_tbl,
                                co_variance_tbl,
                                hazard_tbl,
                                fitted_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            self._try_drop(tables)
            raise

        # pylint: disable=attribute-defined-outside-init
        conn = self.conn_context
        self.coefficients_ = conn.table(coef_tbl)
        self.covariance_variance_ = conn.table(co_variance_tbl)
        self.hazard_ = conn.table(hazard_tbl) if self.calculate_hazard is not None else None
        self.fitted_ = conn.table(fitted_tbl) if key is not None else None
        self.statistics_ = conn.table(stats_tbl)

    def predict(self, data, key, features=None):
        r"""
        Predict dependent variable values based on fitted model.

        Parameters
        ----------

        data : DataFrame
            Independent variable values used for prediction.

        key : str
            Name of the ID column.

        features : list of str, optional
            Names of the feature columns.

        Returns
        -------
        DataFrame
            Predicted values, structured as follows:

              - ID column, with same name and type as ``data`` 's ID column.
              - VALUE, type DOUBLE, representing predicted values.

        """

        # pylint: disable=too-many-locals
        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)

        if features is None:
            cols = data.columns
            cols.remove(key)
            features = cols

        data = data[[key] + features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()

        tables = param_tbl, fitted_tbl = [
            '#PAL_COXPH_{}_TBL_{}_{}'.format(name, self.id, unique_id)
            for name in ['PARAM', 'FITTED']]

        param_rows = [
            ('TYPE', None, None, self.type_kind),
            ('SIGNIFICANCE_LEVEL', None, self.significance_level, None),
        ]

        try:
            self._call_pal_auto('PAL_COXPH_PREDICT',
                                data,
                                self.statistics_,
                                self.coefficients_,
                                self.covariance_variance_,
                                ParameterTable(param_tbl).with_data(param_rows),
                                fitted_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            self._try_drop(tables)
            raise

        return self.conn_context.table(fitted_tbl)

    def score(self, data, key, features=None, label=None):
        r"""
        Returns the coefficient of determination R \:sup:`2`\  of the prediction.

        Parameters
        ----------

        data : DataFrame
            Data on which to assess model performance.

        key : str
            Name of the ID column.

        features : list of str, optional
            Names of the feature columns.

        label : str, optional
            Name of the dependent variable.

            Defaults to the last column.
            (This is not the PAL default.)

        Returns
        -------

        accuracy : float
            The coefficient of determination R \:sup:`2`\  of the prediction on the
            given data.
        """

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
        prediction = prediction.select(key, 'PREDICTION')
        prediction = prediction.cast('PREDICTION', 'DOUBLE')
        prediction = prediction.rename_columns(['ID_P', 'PREDICTION'])

        actual = data.select(key, label).rename_columns(['ID_A', 'ACTUAL'])

        joined = actual.join(prediction, 'ID_P=ID_A').select(
            'ACTUAL', 'PREDICTION')
        return metrics.r2_score(self.conn_context,
                                joined,
                                label_true='ACTUAL',
                                label_pred='PREDICTION')
