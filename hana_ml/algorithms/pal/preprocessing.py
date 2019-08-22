"""
This module contains PAL wrappers for preprocessing algorithms.

The following classes are available:

    * :class:`FeatureNormalizer`
    * :class:`KBinsDiscretizer`
    * :class:`Imputer`
"""

#pylint: disable=line-too-long, unused-variable
import logging
import uuid

from hdbcli import dbapi
from hana_ml.ml_exceptions import FitIncompleteError
from .pal_base import (
    PALBase,
    ParameterTable,
    INTEGER,
    DOUBLE,
    NVARCHAR,
    ListOfStrings,
    ListOfTuples
)

logger = logging.getLogger(__name__)#pylint: disable=invalid-name
#pylint:disable=too-many-lines
class FeatureNormalizer(PALBase):#pylint: disable=too-many-instance-attributes
    """
    Normalize a dataframe.

    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA system.

    method : {'min-max', 'z-score', 'decimal'}

        Scaling methods:

            - 'min-max': Min-max normalization.
            - 'z-score': Z-Score normalization.
            - 'decimal': Decimal scaling normalization.

    z_score_method : {'mean-standard', 'mean-mean', 'median-median'}, optional

        Only valid when ``method`` is 'z-score'.

            - 'mean-standard': Mean-Standard deviation
            - 'mean-mean': Mean-Mean deviation
            - 'median-median': Median-Median absolute deviation

    new_max : float, optional

        The new maximum value for min-max normalization.

        Only valid when ``method`` is 'min-max'.

    new_min : float, optional

        The new minimum value for min-max normalization.

        Only valid when ``method`` is 'min-max'.

    thread_ratio : float, optional

        Controls the proportion of available threads to use.
        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads. Values between 0 and 1
        will use that percentage of available threads. Values outside this range
        tell PAL to heuristically determine the number of threads to use.
        Does not affect transform().

        Defaults to 0.

    Attributes
    ----------

    result_ : DataFrame

        Scaled dataset from fit() and fit_transform().

    model_ :

        Trained model content.

    Examples
    --------

    Input dataframe for training:

    >>> df1.head(4).collect()
        ID    X1    X2
    0    0   6.0   9.0
    1    1  12.1   8.3
    2    2  13.5  15.3
    3    3  15.4  18.7

    Creating FeatureNormalizer instance:

    >>> fn = FeatureNormalizer(cc, method="min-max", new_max=1.0, new_min=0.0)

    Performing fit() on given dataframe:

    >>> fn.fit(df1, key='ID')
    >>> fn.result_.head(4).collect()
        ID        X1        X2
    0    0  0.000000  0.033175
    1    1  0.186544  0.000000
    2    2  0.229358  0.331754
    3    3  0.287462  0.492891

    Input dataframe for transforming:

    >>> df2.collect()
       ID  S_X1  S_X2
    0   0   6.0   9.0
    1   1   6.0   7.0
    2   2   4.0   4.0
    3   3   1.0   2.0
    4   4   9.0  -2.0
    5   5   4.0   5.0

    Performing transform() on given dataframe:

    >>> result = fn.transform(df2, key='ID')
    >>> result.collect()
       ID      S_X1      S_X2
    0   0  0.000000  0.033175
    1   1  0.000000 -0.061611
    2   2 -0.061162 -0.203791
    3   3 -0.152905 -0.298578
    4   4  0.091743 -0.488152
    5   5 -0.061162 -0.156398
    """

    method_map = {'min-max': 0, 'z-score': 1, 'decimal': 2}
    z_score_method_map = {'mean-standard': 0, 'mean-mean': 1, 'median-median': 2}

    def __init__(self,#pylint: disable=too-many-arguments
                 conn_context,
                 method,
                 z_score_method=None,
                 new_max=None,
                 new_min=None,
                 thread_ratio=None):
        super(FeatureNormalizer, self).__init__(conn_context)
        self.method = self._arg('method', method, self.method_map, required=True)
        self.z_score_method = self._arg('z_score_method', z_score_method, self.z_score_method_map)
        self.new_max = self._arg('new_max', new_max, float)
        self.new_min = self._arg('new_min', new_min, float)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)

        if z_score_method is not None:
            if method.lower() != 'z-score':
                msg = 'z_score_method is not applicable when scale method is not z-score.'
                logger.error(msg)
                raise ValueError(msg)
        else:
            if method.lower() == 'z-score':
                msg = 'z_score_method must be provided when scale method is z-score.'
                logger.error(msg)
                raise ValueError(msg)

        if method.lower() == 'min-max':
            if new_min is None or new_max is None:
                msg = 'new_min and new_max must be provided when scale method is min-max.'
                logger.error(msg)
                raise ValueError(msg)

        if method.lower() != 'min-max':
            if new_min is not None or new_max is not None:
                msg = 'new_min or new_max is not applicable when scale method is not min-max.'
                logger.error(msg)
                raise ValueError(msg)

    def fit(self, data, key, features=None):#pylint:disable=invalid-name, too-many-locals
        """
        Normalize input data and generate a scaling model using one of the three
        scaling methods: min-max normalization, z-score normalization and
        normalization by decimal scaling.

        Parameters
        ----------

        data : DataFrame

            DataFrame to be normalized.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.
            If ``features`` is not provided, it defaults to all non-ID columns.
        """
        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)

        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols

        data = data[[key] + features]

        outputs = ['RESULT', 'MODEL', 'STATISTIC', 'PLACEHOLDER']
        outputs = ['#PAL_FN_{}_TBL_{}'.format(name, self.id)
                   for name in outputs]
        result_tbl, model_tbl, statistic_tbl, placeholder_tbl = outputs

        param_rows = [
            ('SCALING_METHOD', self.method, None, None),
            ('Z-SCORE_METHOD', self.z_score_method, None, None),
            ('THREAD_RATIO', None, self.thread_ratio, None),
            ('NEW_MAX', None, self.new_max, None),
            ('NEW_MIN', None, self.new_min, None)
            ]

        try:
            param_t = ParameterTable(name='#PARAM_TBL')
            self._call_pal_auto('PAL_SCALE',
                                data,
                                param_t.with_data(param_rows),
                                *outputs)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            self._try_drop('#PARAM_TBL')
            self._try_drop(outputs)
            raise

        # pylint: disable=attribute-defined-outside-init
        self.result_ = self.conn_context.table(result_tbl)
        self.model_ = self.conn_context.table(model_tbl)

    def fit_transform(self, data, key, features=None):#pylint:disable=invalid-name
        """
        Fit with the dataset and return the results.

        Parameters
        ----------

        data : DataFrame
            DataFrame to be normalized.
        key : str
            Name of the ID column.
        features : list of str, optional
            Names of the feature columns.
            If ``features`` is not provided, it defaults to all non-ID columns.

        Returns
        -------

        DataFrame
            Normalized result, with the same structure as ``data``.
        """
        self.fit(data, key, features)
        return self.result_

    def transform(self, data, key, features=None):#pylint:disable=invalid-name
        """
        Scales data based on the previous scaling model.

        Parameters
        ----------

        data : DataFrame
            DataFrame to be normalized.
        key : str
            Name of the ID column.
        features : list of str, optional
            Names of the feature columns.
            If ``features`` is not provided, it defaults to all non-ID columns.

        Returns
        -------

        DataFrame
            Normalized result, with the same structure as ``data``.
        """
        if self.model_ is None:
            raise FitIncompleteError("Model not initialized.")

        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)

        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols

        data = data[[key] + features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()

        result_tbl = '#PAL_FN_RESULT_TBL_{}_{}'.format(self.id, unique_id)

        try:
            param_t = ParameterTable(name='#PARAM_TBL')
            self._call_pal_auto('PAL_SCALE_WITH_MODEL',
                                data,
                                self.model_,
                                param_t,
                                result_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            self._try_drop('#PARAM_TBL')
            self._try_drop(result_tbl)
            raise

        return self.conn_context.table(result_tbl)


class KBinsDiscretizer(PALBase):
    """
    Bin continuous data into number of intervals and perform local smoothing.

    Parameters
    ----------

    conn_context : ConnectionContext
        Connection to the HANA system.
    strategy : {'uniform_number', 'uniform_size', 'quantile', 'sd'}
        Binning methods:
            - 'uniform_number': Equal widths based on the number of bins.
            - 'uniform_size': Equal widths based on the bin size.
            - 'quantile': Equal number of records per bin.
            - 'sd': Bins are divided based on the distance from the mean.
              Most bins are one standard deviation wide, except that the
              center bin contains all values within one standard deviation
              from the mean, and the leftmost and rightmost bins contain
              all values more than ``n_sd`` standard deviations from the
              mean in the corresponding directions.
    smoothing : {'means', 'medians', 'boundaries'}
        Smoothing methods:
            - 'means': Each value within a bin is replaced by the average of
              all the values belonging to the same bin.
            - 'medians': Each value in a bin is replaced by the median of all
              the values belonging to the same bin.
            - 'boundaries': The minimum and maximum values in a given bin are
              identified as the bin boundaries. Each value in the bin is then
              replaced by its closest boundary value. When the distance is
              equal to both sides, it will be replaced by the front boundary
              value.

        Values used for smoothing are not re-calculated during transform().
    n_bins : int, optional
        The number of bins.
        Only valid when ``strategy`` is 'uniform_number' or 'quantile'.

        Defaults to 2.
    bin_size : int, optional
        The interval width of each bin.
        Only valid when ``strategy`` is 'uniform_size'.

        Defaults to 10.
    n_sd : int, optional
        The leftmost bin contains all values located further than n_sd
        standard deviations lower than the mean, and the rightmost bin
        contains all values located further than n_sd standard deviations
        above the mean.
        Only valid when ``strategy`` is 'sd'.

        Defaults to 1.

    Attributes
    ----------
    result_ : DataFrame
        Binned dataset from fit() and fit_transform().
    model_ :
        Binning model content.

    Examples
    --------
    Input dataframe for fitting:

    >>> df1.collect()
        ID  DATA
    0    0   6.0
    1    1  12.0
    2    2  13.0
    3    3  15.0
    4    4  10.0
    5    5  23.0
    6    6  24.0
    7    7  30.0
    8    8  32.0
    9    9  25.0
    10  10  38.0

    Creating KBinsDiscretizer instance:

    >>> binning = KBinsDiscretizer(cc, strategy='uniform_size',
    ...                          smoothing='means',
    ...                          bin_size=10)

    Performing fit() on the given dataframe:

    >>> binning.fit(df1, key='ID')
    >>> binning.result_.collect()
        ID  BIN_INDEX       DATA
    0    0          1   8.000000
    1    1          2  13.333333
    2    2          2  13.333333
    3    3          2  13.333333
    4    4          1   8.000000
    5    5          3  25.500000
    6    6          3  25.500000
    7    7          3  25.500000
    8    8          4  35.000000
    9    9          3  25.500000
    10  10          4  35.000000

    Input dataframe for transforming:

    >>> df2.collect()
       ID  DATA
    0   0   6.0
    1   1  67.0
    2   2   4.0
    3   3  12.0
    4   4  -2.0
    5   5  40.0

    Performing transform() on the given dataframe:

    >>> result = binning.transform(df2, key='ID')
    >>> result.collect()
       ID  BIN_INDEX       DATA
    0   0          1   8.000000
    1   1         -1  67.000000
    2   2          1   8.000000
    3   3          2  13.333333
    4   4          1   8.000000
    5   5          4  35.000000
    """

    strategy_map = {'uniform_number': 0, 'uniform_size': 1, 'quantile': 2, 'sd': 3}
    smooth_map = {'means': 0, 'medians': 1, 'boundaries': 2}

    def __init__(self,#pylint: disable=too-many-arguments
                 conn_context,
                 strategy,
                 smoothing,
                 n_bins=None,
                 bin_size=None,
                 n_sd=None):
        super(KBinsDiscretizer, self).__init__(conn_context)
        self.strategy = self._arg('strategy', strategy, self.strategy_map, required=True)
        self.smoothing = self._arg('smoothing', smoothing, self.smooth_map, required=True)
        self.n_bins = self._arg('n_bins', n_bins, int)
        self.bin_size = self._arg('bin_size', bin_size, int)
        self.n_sd = self._arg('n_sd', n_sd, int)
        #following checks are based on PAL docs, pal example has 'sd' with uniform_size
        #tested that pal ignores SD in actual executions
        if (strategy.lower() != 'uniform_number' and strategy.lower() != 'quantile'
                and n_bins is not None):
            msg = "n_bins is only applicable when strategy is uniform_number or quantile."
            logger.error(msg)
            raise ValueError(msg)
        if strategy.lower() != 'uniform_size' and bin_size is not None:
            msg = "bin_size is only applicable when strategy is uniform_size."
            logger.error(msg)
            raise ValueError(msg)
        if strategy.lower() != 'sd' and n_sd is not None:
            msg = "n_sd is only applicable when strategy is sd."
            logger.error(msg)
            raise ValueError(msg)

    def fit(self, data, key, features=None):#pylint: disable=too-many-locals
        """
        Bin input data into number of intervals and smooth.

        Parameters
        ----------

        data : DataFrame
            DataFrame to be discretized.
        key : str
            Name of the ID column.
        features : list of str, optional
            Names of the feature columns. Since the underlying PAL_BINNING
            only supports one feature, this list can only contain one element.
            If ``features`` is not provided, ``data`` must have exactly 1
            non-ID column, and ``features`` defaults to that column.
        """

        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)

        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols

        if len(features) != 1:
            msg = ('PAL binning requires exactly one ' +
                   'feature column.')
            logger.error(msg)
            raise TypeError(msg)

        data = data[[key] + features]
        #PAL_BINNING requires stats and placeholder table which is not mentioned in PAL doc
        outputs = ['RESULT', 'MODEL', 'STATISTIC', 'PLACEHOLDER']
        outputs = ['#PAL_BINNING_{}_TBL_{}'.format(name, self.id)
                   for name in outputs]
        result_tbl, model_tbl, statistic_tbl, placeholder_tbl = outputs

        param_rows = [
            ('BINNING_METHOD', self.strategy, None, None),
            ('SMOOTH_METHOD', self.smoothing, None, None),
            ('BIN_NUMBER', self.n_bins, None, None),
            ('BIN_DISTANCE', self.bin_size, None, None),
            ('SD', self.n_sd, None, None)
            ]

        try:
            param_t = ParameterTable(name='#PARAM_TBL')
            self._call_pal_auto('PAL_BINNING',
                                data,
                                param_t.with_data(param_rows),
                                *outputs)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            self._try_drop('#PARAM_TBL')
            self._try_drop(outputs)
            raise

        # pylint: disable=attribute-defined-outside-init
        self.result_ = self.conn_context.table(result_tbl)
        self.model_ = self.conn_context.table(model_tbl)

    def fit_transform(self, data, key, features=None):#pylint:disable=invalid-name
        """
        Fit with the dataset and return the results.

        Parameters
        ----------

        data : DataFrame
            DataFrame to be binned.
        key : str
            Name of the ID column.
        features : list of str, optional
            Names of the feature columns. Since the underlying PAL_BINNING
            only supports one feature, this list can only contain one element.
            If ``features`` is not provided, ``data`` must have exactly 1
            non-ID column, and ``features`` defaults to that column.

        Returns
        -------

        DataFrame
            Binned result, structured as follows:

              - DATA_ID column: with same name and type as ``data``'s ID column.
              - BIN_INDEX: type INTEGER, assigned bin index.
              - BINNING_DATA column: smoothed value, with same name and
                type as ``data``'s feature column.

        """

        self.fit(data, key, features)
        return self.result_

    def transform(self, data, key, features=None):#pylint:disable=invalid-name
        """
        Bin data based on the previous binning model.

        Parameters
        ----------

        data : DataFrame
            DataFrame to be binned.
        key : str
            Name of the ID column.
        features : list of str, optional
            Names of the feature columns. Since the underlying
            PAL_BINNING_ASSIGNMENT only supports one feature, this list can
            only contain one element.
            If ``features`` is not provided, ``data`` must have exactly 1
            non-ID column, and ``features`` defaults to that column.

        Returns
        -------

        DataFrame
            Binned result, structured as follows:

              - DATA_ID column: with same name and type as ``data`` 's ID column.
              - BIN_INDEX: type INTEGER, assigned bin index.
              - BINNING_DATA column: smoothed value, with same name and
                type as ``data`` 's feature column.

        """

        if self.model_ is None:
            raise FitIncompleteError("Model not initialized.")

        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)

        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols

        if len(features) != 1:
            msg = ('PAL binning assignment requires exactly one ' +
                   'feature column.')
            logger.error(msg)
            raise TypeError(msg)

        data = data[[key] + features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        result_tbl = '#PAL_BINNING_RESULT_TBL_{}_{}'.format(self.id, unique_id)

        try:
            param_t = ParameterTable(name='#PARAM_TBL')
            self._call_pal_auto('PAL_BINNING_ASSIGNMENT',
                                data,
                                self.model_,
                                param_t,
                                result_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            self._try_drop('#PARAM_TBL')
            self._try_drop(result_tbl)
            raise

        return self.conn_context.table(result_tbl)

#pylint: disable=too-many-instance-attributes, too-few-public-methods
class Imputer(PALBase):
    r"""
    Missing value imputation for dataframes.

    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA system.

    strategy : {'non', 'mean', 'median', 'zero', 'als', 'delete'}, optional
        The overall imputation strategy for all Numerical columns.

        Defaults to 'mean'.

    thread_ratio : float, optional

        Controls the proportion of available threads to use.
        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads. Values between
        0 and 1 will use up to that percentage of available threads. Values
        outside this range tell PAL to heuristically determine the number of
        threads to use.

        The following parameters all have pre-fix 'als\_', and are invoked only when
        'als' is the overall imputation strategy. Those parameters are for setting
        up the alternating-least-square(ALS) mdoel for data imputation.

        Defaults to 0.0.

    als_factors : int, optional

        Length of factor vectors in the ALS model.
        It should be less than the number of numerical columns,
        so that the imputation results would be meaningful.

        Defaults to 3.

    als_lambda : float, optional

        L2 regularization applied to the factors in the ALS model.
        Should be non-negative.

        Defaults to 0.01.

    als_maxit : int, optional

        Maximum number of iterations for solving the ALS model.

        Defaults to 20.

    als_randomstate : int, optional

        Specifies the seed of the random number generator used
        in the training of ALS model:

            0: Uses the current time as the seed,

            Others: Uses the specified value as the seed.

        Defaults to 0.

    als_exit_threshold : float, optional

        Specify a value for stopping the training of ALS nmodel.
        If the improvement of the cost function of the ALS model
        is less than this value between consecutive checks, then
        the training process will exit.
        0 means there is no checking of the objective value when
        running the algorithms, and it stops till the maximum number of
        iterations has been reached.

        Defaults to 0.

    als_exit_interval : int, optional

        Specify the number of iterations between consecutive checking of
        cost functions for the ALS model, so that one can see if the
        pre-specified exit_threshold is reached.

        Defaults to 5.

    als_linsolver : {'cholsky', 'cg'}, optional

        Linear system solver for the ALS model.
        'cholsky' is usually much faster.
        'cg' is recommended when als_factors is large.

        Defaults to 'cholsky'.

    als_maxit : int, optional

        Specifies the maximum number of iterations for cg algorithm.
        Invoked only when the 'cg' is the chosen linear system solver for ALS.

        Defaults to 3.

    als_centering : bool, optional

        Whether to center the data by column before training the ALS model.

        Defaults to True.

    als_scaling : bool, optional

        Whether to scale the data by column before training the ALS model.

        Defaults to True.

    Attributes
    ----------

    stats_model_ : DataFrame

        Statistics model content.

    Examples
    --------

    Input data:

    >>> df.head(5).collect()
       V0   V1 V2   V3   V4    V5
    0  10  0.0  D  NaN  1.4  23.6
    1  20  1.0  A  0.4  1.3  21.8
    2  50  1.0  C  NaN  1.6  21.9
    3  30  NaN  B  0.8  1.7  22.6
    4  10  0.0  A  0.2  NaN   NaN

    Create an Imputer instance using 'mean' strategy and call fit():

    >>> impute = Imputer(conn_context, strategy='mean')
    >>> result = impute.fit_transform(df, categorical_variable=['V1'], strategy_by_col=[('V1', 'categorical_const', '0')])

    >>> result.head(5).collect()
       V0  V1 V2        V3        V4         V5
    0  10   0  D  0.507692  1.400000  23.600000
    1  20   1  A  0.400000  1.300000  21.800000
    2  50   1  C  0.507692  1.600000  21.900000
    3  30   0  B  0.800000  1.700000  22.600000
    4  10   0  A  0.200000  1.469231  20.646154

    The stats_model_ content of input DataFrame:

    >>> impute.stats_model_.head(5).collect()
                STAT_NAME                   STAT_VALUE
    0  V0.NUMBER_OF_NULLS                            3
    1  V0.IMPUTATION_TYPE                         MEAN
    2    V0.IMPUTED_VALUE                           24
    3  V1.NUMBER_OF_NULLS                            2
    4  V1.IMPUTATION_TYPE  SPECIFIED_CATEGORICAL_VALUE

    The above stats_model_ content of the input DataFrame can be applied
    to imputing another DataFrame with the same data structure, e.g. consider
    the following DataFrame with missing values:

    >>> df1.collect()
       ID    V0   V1    V2   V3   V4    V5
    0   0  20.0  1.0     B  NaN  1.5  21.7
    1   1  40.0  1.0  None  0.6  1.2  24.3
    2   2   NaN  0.0     D  NaN  1.8  22.6
    3   3  50.0  NaN     C  0.7  1.1   NaN
    4   4  20.0  1.0     A  0.3  NaN  20.6

    With attribute impute.stats_model_ being obtained, one can impute the
    missing values of df1 via the following line of code, and then check
    the result:

    >>> result1, _ = impute.transform(df1, key='ID')
    >>> result1.collect()
       ID  V0  V1 V2        V3        V4         V5
    0   0  20   1  B  0.507692  1.500000  21.700000
    1   1  40   1  A  0.600000  1.200000  24.300000
    2   2  24   0  D  0.507692  1.800000  22.600000
    3   3  50   0  C  0.700000  1.100000  20.646154
    4   4  20   1  A  0.300000  1.469231  20.600000

    Create an Imputer instance using other strategies, e.g. 'als' strategy
    and then call fit():

    >>> impute = Imputer(cc, strategy='als', als_factors=2, als_randomstate=1)
    >>> result2 = impute.fit_transform(df, categorical_variable=['V1'])
    >>> result2.head(5).collect()
       V0  V1 V2        V3        V4         V5
    0  10   0  D  0.306957  1.400000  23.600000
    1  20   1  A  0.400000  1.300000  21.800000
    2  50   1  C  0.930689  1.600000  21.900000
    3  30   0  B  0.800000  1.700000  22.600000
    4  10   0  A  0.200000  1.333668  21.371753

    """

    overall_imputation_map = {'non':0, 'mean':1, 'median':2,
                              'zero':3, 'als':4, 'delete':5}
    column_imputation_map = {'non':0, 'delete':1, 'most_frequent':100,
                             'categorical_const':101,
                             'mean':200, 'median':201,
                             'numerical_const':203,
                             'als':204}
    dtype_escp = {'INT':INTEGER, 'DOUBLE':DOUBLE,
                  'NVARCHAR':NVARCHAR(5000), 'VARCHAR':NVARCHAR(256)}
    solver_map = {'cholsky':0, 'cg':1}
    #pylint:disable=too-many-arguments
    def __init__(self, conn_context,
                 strategy=None,
                 als_factors=None,
                 als_lambda=None,
                 als_maxit=None,
                 als_randomstate=None,
                 als_exit_threshold=None,
                 als_exit_interval=None,
                 als_linsolver=None,
                 als_cg_maxit=None,
                 als_centering=None,
                 als_scaling=None,
                 thread_ratio=None):
        super(Imputer, self).__init__(conn_context)
        self.strategy = self._arg('strategy', strategy, self.overall_imputation_map)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.als_factors = self._arg('als_factors', als_factors, int)
        self.als_lambda = self._arg('als_lambda', als_lambda, float)
        self.als_maxit = self._arg('als_maxit', als_maxit, int)
        self.als_randomstate = self._arg('als_randomstate', als_randomstate, int)
        self.als_exit_threshold = self._arg('als_exit_threshold', als_exit_threshold,
                                            float)
        self.als_exit_interval = self._arg('als_exit_interval', als_exit_interval, int)
        self.als_linsolver = self._arg('als_linsolver', als_linsolver, self.solver_map)
        self.als_cg_maxit = self._arg('als_cg_maxit', als_cg_maxit, int)
        self.als_centering = self._arg('als_centering', als_centering, bool)
        self.als_scaling = self._arg('als_scaling', als_scaling, bool)
        self.stats_model_ = None

    #pylint:disable=attribute-defined-outside-init, too-many-locals
    def fit_transform(self, data, key=None,
                      categorical_variable=None,
                      strategy_by_col=None):
        """
        Inpute the missing values of a DataFrame, return the result,
        and collect the related statistics/model info for imputation.

        Parameters
        ----------

        data : DataFrame
            Input data with missing values.
        key : str, optional
            Name of the ID column.
            Assume no ID column if key not provided.
        categorical_variable : str, optional
            Names of columns with INTEGER data type that should actually
            be treated as categorical.
            By default, columns of INTEGER and DOUBLE type are all treated
            numerical, while columns of VARCHAR or NVARCHAR type are treated
            as categorical.
        strategy_by_col : ListOfTuples, optional
            Specifies the imputation strategy for a set of columns, which
            overrides the overall strategy for data imputation.
            Each tuple in the list should contain at least two elements,
            such that:
            The first element is the name of a column;
            the second element is the imputation strategy of that column.
            If the imputation strategy is 'categorical_const' or 'numerical_const',
            then a third element must be included in the tuple, which specifies
            the constant value to be used to substitute the detected missing values
            in the column.

            An illustrative example:
                [('V1', 'categorical_const', '0'), ('V5','median')]

        Returns
        -------

        DataFrame
            Imputed result using specified strategy, with the same data structure,
            i.e. column names and data types same as ``data``.
        """
        key = self._arg('key', key, str)
        self.categorical_variable = self._arg('categorical_variable',
                                              categorical_variable, ListOfStrings)
        self.strategy_by_col = self._arg('strategy_by_col',
                                         strategy_by_col, ListOfTuples)
        if self.strategy_by_col is not None:
            for col_strategy in self.strategy_by_col:
                if col_strategy[0] not in data.columns:
                    msg = ('{} is not a column name'.format(col_strategy[0]) +
                           ' of the input dataframe.')
                    logger.error(msg)
                    raise ValueError(msg)

        param_rows = [('IMPUTATION_TYPE', self.strategy, None, None),
                      ('ALS_FACTOR_NUMBER', self.als_factors, None, None),
                      ('ALS_REGULARIZATION', None, self.als_lambda, None),
                      ('ALS_MAX_ITERATION', self.als_maxit, None, None),
                      ('ALS_SEED', self.als_randomstate, None, None),
                      ('ALS_EXIT_THRESHOLD', None, self.als_exit_threshold, None),
                      ('ALS_EXIT_INTERVAL', self.als_exit_interval, None, None),
                      ('ALS_LINEAR_SYSTEM_SOLVER', self.als_linsolver, None, None),
                      ('ALS_CG_MAX_ITERATION', self.als_cg_maxit, None, None),
                      ('ALS_CENTERING', self.als_centering, None, None),
                      ('ALS_SCALING', self.als_scaling, None, None),
                      ('THREAD_RATIO', None, self.thread_ratio, None),
                      ('HAS_ID', key is not None, None, None)]

        if self.categorical_variable is not None:
            param_rows.extend([('CATEGORICAL_VARIABLE', None, None, str(var))
                               for var in self.categorical_variable])
        #override the overall imputation methods for specified columns
        if self.strategy_by_col is not None:
            for col_imp_type in self.strategy_by_col:
                imp_type = self._arg('imp_type', col_imp_type[1], self.column_imputation_map)
                if len(col_imp_type) == 2:
                    param_rows.extend([('{}_IMPUTATION_TYPE'.format(col_imp_type[0]),
                                        imp_type, None, None)])
                elif len(col_imp_type) == 3:
                    if imp_type == 101:
                        param_rows.extend([('{}_IMPUTATION_TYPE'.format(col_imp_type[0]),
                                            imp_type, None, str(col_imp_type[2]))])
                    else:
                        param_rows.extend([('{}_IMPUTATION_TYPE'.format(col_imp_type[0]),
                                            imp_type, col_imp_type[2], None)])
                else:
                    continue

        outputs = ['RESULT', 'STATS_MODEL']
        outputs = ['#PAL_IMPUTATION_{}_TBL_{}'.format(name, self.id)
                   for name in outputs]
        result_tbl, stats_model_tbl = outputs

        try:
            param_t = ParameterTable(name='#PARAM_TBL')
            self._call_pal_auto('PAL_MISSING_VALUE_HANDLING',
                                data,
                                param_t.with_data(param_rows),
                                *outputs)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            self._try_drop('#PARAM_TBL')
            self._try_drop(outputs)
            raise

        conn = self.conn_context
        self.stats_model_ = conn.table(stats_model_tbl)
        return conn.table(result_tbl)

    def transform(self, data, key=None, thread_ratio=None):
        """
        The function imputes missing values a DataFrame using
        statistic/model info collected from another DataFrame.

        Parameters
        ----------

        data : DataFrame
           Input DataFrame.
        key : str, optional
           Name of ID cclumn. Assumed no ID column if not provided.
        thread_ratio : float, optional
           Controls the proportion of available threads to use.
           The value range is from 0 to 1, where 0 indicates a single thread,
           and 1 indicates up to all available threads. Values between
           0 and 1 will use up to that percentage of available threads. Values
           outside this range tell PAL to heuristically determine the number of
           threads to use.

           Defaults to 0.0.

        Returns
        -------

        DataFrames.
            The first DataFrame is the inputation result structured same as ``data``.
            The second DataFrame is the statistics for the imputation result,
            structured as

            - STAT_NAME: type NVACHAR(256), statistics name.
            - STATE_VALUE: type NVACHAR(5000), statistics value.

        """

        if self.stats_model_ is None:
            raise FitIncompleteError("Stats/model not initialized. "+
                                     "Perform a fit_transform first.")
        key = self._arg('key', key, str)
        thread_ratio = self._arg('thread_ratio', thread_ratio, float)

        param_rows = [('HAS_ID', key is not None, None, None),
                      ('THREAD_RATIO', None, thread_ratio, None)]

        outputs = ['RESULT', 'STATS']
        outputs = ['#PAL_IMPUTE_PREDICT_{}_TBL_{}'.format(name, self.id)
                   for name in outputs]
        result_tbl, stats_tbl = outputs

        try:
            param_t = ParameterTable(name='#PARAM_TBL')
            self._call_pal_auto('PAL_MISSING_VALUE_HANDLING_WITH_MODEL',
                                data,
                                self.stats_model_,
                                param_t.with_data(param_rows),
                                *outputs)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            self._try_drop('#PARAM_TBL')
            self._try_drop(outputs)
            raise

        conn = self.conn_context
        return conn.table(result_tbl), conn.table(stats_tbl)
