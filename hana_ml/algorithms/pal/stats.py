"""
This module contains PAL wrappers for statistics algorithms.

The following functions are available:

    * :func:`chi_squared_goodness_of_fit`
    * :func:`chi_squared_independence`
    * :func:`ttest_1samp`
    * :func:`ttest_ind`
    * :func:`ttest_paired`
    * :func:`f_oneway`
    * :func:`f_oneway_repeated`
    * :func:`univariate_analysis`
    * :func:`covariance_matrix`
    * :func:`pearsonr_matrix`
    * :func:`iqr`
"""

#pylint:disable=too-many-lines, line-too-long
import logging
import uuid

from hdbcli import dbapi
from .pal_base import (
    ParameterTable,
    arg,
    try_drop,
    ListOfStrings,
    require_pal_usable,
    call_pal_auto
)

logger = logging.getLogger(__name__)#pylint: disable=invalid-name

def _ttest_base(conn_context, data, mu=None, test_type=None, paired=None, var_equal=None,#pylint: disable=too-many-arguments, too-many-locals, invalid-name
                conf_level=None):
    require_pal_usable(conn_context)

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    mu_final = arg('mu', mu, float)
    test_type_map = {'two_sides':0, 'less':1, 'greater':2}
    test_type_final = arg('test_type', test_type, test_type_map)
    paired_final = arg('paired', paired, bool)
    var_equal_final = arg('var_equal', var_equal, bool)
    conf_level_final = arg('conf_level', conf_level, float)
    stat_tbl = '#TTEST_STAT_TBL_{}'.format(unique_id)
    param_array = [('TEST_TYPE', test_type_final, None, None),
                   ('MU', None, mu_final, None),
                   ('PAIRED', paired_final, None, None),
                   ('VAR_EQUAL', var_equal_final, None, None),
                   ('CONF_LEVEL', None, conf_level_final, None)]
    try:
        call_pal_auto(conn_context,
                      'PAL_T_TEST',
                      data,
                      ParameterTable().with_data(param_array),
                      stat_tbl)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn_context, stat_tbl)
        raise
    return conn_context.table(stat_tbl)

def chi_squared_goodness_of_fit(conn_context, data, key, observed_data=None, expected_freq=None):
    r"""
    Perform the chi-squared goodness-of fit test to tell whether or not an \
    observed distribution differs from an expected chi-squared distribution.

    Parameters
    ----------

    conn_context : ConnectionContext

        Database connection object.

    data : DataFrame

        Input data.

    key : str

        Name of the ID column.

    observed_data : str, optional

        Name of column for counts of actual observations belonging to each \
        category. \
        If not given, the input dataframe must only have three columns. \
        The first of the non-ID columns will be ``observed_data``.

    expected_freq : str, optional

        Name of the expected frequency column. \
        If not given, the input dataframe must only have three columns. \
        The second of the non-ID columns will be ``expected_freq``.

    Returns
    -------

    count_comparison_df : DataFrame

        Comparsion between the actual counts and the expected \
        counts, structured as follows:

            - ID column, with same name and type as ``data`` 's ID column.
            - Observed data column, with same name as ``data``'s observed_data \
              column, but always with type DOUBLE.
            - EXPECTED, type DOUBLE, expected count in each category.
            - RESIDUAL, type DOUBLE, the difference between the observed \
              counts and the expected counts.

    stat_df : DataFrame

        Statistical outputs, including the calculated chi-squared value, \
        degrees of freedom and p-value, structured as follows:

            - STAT_NAME, type NVARCHAR(100), name of statistics.
            - STAT_VALUE, type DOUBLE, value of statistics.

    Examples
    --------

    Data to test:

    >>> df = cc.table('PAL_CHISQTESTFIT_DATA_TBL')
    >>> df.collect()
       ID  OBSERVED    P
    0   0     519.0  0.3
    1   1     364.0  0.2
    2   2     363.0  0.2
    3   3     200.0  0.1
    4   4     212.0  0.1
    5   5     193.0  0.1

    Perform chi_squared_goodness_of_fit:

    >>> res, stat = chi_squared_goodness_of_fit(cc, df, 'ID')
    >>> res.collect()
       ID  OBSERVED  EXPECTED  RESIDUAL
    0   0     519.0     555.3     -36.3
    1   1     364.0     370.2      -6.2
    2   2     363.0     370.2      -7.2
    3   3     200.0     185.1      14.9
    4   4     212.0     185.1      26.9
    5   5     193.0     185.1       7.9
    >>> stat.collect()
               STAT_NAME  STAT_VALUE
    0  Chi-squared Value    8.062669
    1  degree of freedom    5.000000
    2            p-value    0.152815
    """
    require_pal_usable(conn_context)

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    key = arg('key', key, str, True)
    observed_data = arg('observed_data', observed_data, str)
    if observed_data is None:
        if len(data.columns) != 3:
            msg = ("If 'observed_data' is not given, the input dataframe " +
                   "must only have three columns.")
            logger.error(msg)
            raise ValueError(msg)
        else:
            observed_data = data.columns[1]
    expected_freq = arg('expected_freq', expected_freq, str)
    if expected_freq is None:
        if len(data.columns) != 3:
            msg = ("If 'expected_freq' is not given, the input dataframe " +
                   "must only have three columns.")
            logger.error(msg)
            raise ValueError(msg)
        else:
            expected_freq = data.columns[2]
    data = data[[key, observed_data, expected_freq]]
    tables = result_tbl, stat_tbl = ["#CHI_GOODNESS_{}_{}".format(name, unique_id)
                                     for name in ['RESULT', 'STATISTICS']]

    try:
        call_pal_auto(conn_context,
                      'PAL_CHISQUARED_GOF_TEST',
                      data,
                      *tables)
        return conn_context.table(result_tbl), conn_context.table(stat_tbl)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn_context, tables)
        raise

def chi_squared_independence(conn_context, data, key, observed_data=None,#pylint: disable=too-many-locals,
                             correction=False):
    """
    Perform the chi-squared test of independence to tell whether observations of \
    two variables are independent from each other.

    Parameters
    ----------

    conn_context : ConnectionContext

        Database connection object.

    data : DataFrame

        Input data.

    key : str

        Name of the ID column.

    observed_data : list of str, optional

        Names of the observed data columns. \
        If not given, it defaults to the all the non-ID columns.

    correction : bool, optional

        If True, and the degrees of freedom is 1, apply \
        Yates's correction for continuity. The effect of \
        the correction is to adjust each observed value by \
        0.5 towards the corresponding expected value.

        Defaults to False.

    Returns
    -------

    expected_count_df : DataFrame

        The expected count table, structured as follows:

            - ID column, with same name and type as ``data``'s ID column.
            - Expected count columns, named by prepending ``Expected_`` to \
              each ``observed_data`` column name, type DOUBLE. There will be as \
              many columns here as there are ``observed_data`` columns.

    stat_df : DataFrame

        Statistical outputs, including the calculated chi-squared value, \
        degrees of freedom and p-value, structured as follows:

            - STAT_NAME, type NVARCHAR(100), name of statistics.
            - STAT_VALUE, type DOUBLE, value of statistics.

    Examples
    --------

    Data to test:

    >>> df = cc.table('PAL_CHISQTESTIND_DATA_TBL')
    >>> df.collect()
           ID  X1    X2  X3    X4
    0    male  25  23.0  11  14.0
    1  female  41  20.0  18   6.0

    Perform chi-squared test of independence:

    >>> res, stats = chi_squared_independence(cc, df, 'ID')
    >>> res.collect()
           ID  EXPECTED_X1  EXPECTED_X2  EXPECTED_X3  EXPECTED_X4
    0    male    30.493671    19.867089    13.398734     9.240506
    1  female    35.506329    23.132911    15.601266    10.759494
    >>> stats.collect()
               STAT_NAME  STAT_VALUE
    0  Chi-squared Value    8.113152
    1  degree of freedom    3.000000
    2            p-value    0.043730
    """
    require_pal_usable(conn_context)

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    key = arg('key', key, str, True)
    observed_data = arg('observed_data', observed_data, ListOfStrings)
    cols = data.columns
    cols.remove(key)
    if observed_data is None:
        observed_data = cols
    used_cols = [key] + observed_data
    data = data[used_cols]
    tables = ["#CHI_INDEPENDENCE_{}_{}".format(name, unique_id)
              for name in ['RESULT', 'STATS']]
    result_tbl, stat_tbl = tables
    param_array = [('CORRECTION_TYPE', correction, None, None)]

    try:
        call_pal_auto(conn_context,
                      'PAL_CHISQUARED_IND_TEST',
                      data,
                      ParameterTable().with_data(param_array),
                      *tables)
        return conn_context.table(result_tbl), conn_context.table(stat_tbl)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn_context, tables)
        raise

def ttest_1samp(conn_context, data, col=None, mu=0, test_type='two_sides', conf_level=0.95):#pylint: disable=too-many-arguments, invalid-name
    """
    Perform the t-test to determine whether a sample of observations
    could have been generated by a process with a specific mean.

    Parameters
    ----------

    conn_context : ConnectionContext

        Database connection object.

    data : DataFrame

        DataFrame containing the data.

    col : str, optional

        Name of the column for sample. \
        If not given, the input dataframe must only have one column.

    mu : float, optional

        Hypothesized mean of the population underlying the sample.

        Default value: 0

    test_type : {'two_sides', 'less', 'greater'}, optional

        The alternative hypothesis type.

        Default value: two_sides

    conf_level : float, optional

        Confidence level for alternative hypothesis confidence interval.

        Default value: 0.95

    Returns
    -------

    stat_df : DataFrame

        DataFrame containing the statistics results from the t-test.

    Examples
    --------

    Original data:

    >>> df.collect()
        X1
    0  1.0
    1  2.0
    2  4.0
    3  7.0
    4  3.0

    Perform One Sample T-Test

    >>> ttest_1samp(conn, df).collect()
               STAT_NAME  STAT_VALUE
    0            t-value    3.302372
    1  degree of freedom    4.000000
    2            p-value    0.029867
    3      _PAL_MEAN_X1_    3.400000
    4   confidence level    0.950000
    5         lowerLimit    0.541475
    6         upperLimit    6.258525
    """

    col = arg('col', col, str)
    if col is None:
        if len(data.columns) > 1:
            msg = "If 'col' is not given, the input dataframe must only have one column."
            logger.error(msg)
            raise ValueError(msg)
        else:
            col = data.columns[0]
    data = data.select(col)
    return _ttest_base(conn_context, data, mu=mu, test_type=test_type, conf_level=conf_level)

def ttest_ind(conn_context, data, col1=None, col2=None, mu=0, test_type='two_sides',#pylint: disable=too-many-arguments, invalid-name
              var_equal=False, conf_level=0.95):
    """
    Perform the T-test for the mean difference of two independent samples.

    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA system.

    data : DataFrame

        DataFrame containing the data.

    col1 : str, optional

        Name of the column for sample1. \
        If not given, the input dataframe must only have two columns. \
        The first of the columns will be col1.

    col2 : str, optional

        Name of the column for sample2. \
        If not given, the input dataframe must only have two columns. \
        The second of the columns will be col2.

    mu : float, optional

        Hypothesized difference between the two underlying population means.

        Default value: 0

    test_type : {'two_sides', 'less', 'greater'}, optional

        The alternative hypothesis type.

        Default value: two_sides

    var_equal : bool, optional

        Controls whether to assume that the two samples have equal variance.

        Default value: False

    conf_level : float, optional

        Confidence level for alternative hypothesis confidence interval.

        Default value: 0.95

    Returns
    -------

    stat_df : DataFrame
        DataFrame containing the statistics results from the t-test.

    Examples
    --------

    Original data:

    >>> df.collect()
        X1    X2
    0  1.0  10.0
    1  2.0  12.0
    2  4.0  11.0
    3  7.0  15.0
    4  NaN  10.0

    Perform Independent Sample T-Test

    >>> ttest_ind(conn, df).collect()
               STAT_NAME  STAT_VALUE
    0            t-value   -5.013774
    1  degree of freedom    5.649757
    2            p-value    0.002875
    3      _PAL_MEAN_X1_    3.500000
    4      _PAL_MEAN_X2_   11.600000
    5   confidence level    0.950000
    6         lowerLimit  -12.113278
    7         upperLimit   -4.086722
    """

    missing_msg = ("If 'col1' or 'col2' is not given, " +
                   "the input dataframe must only have two columns.")
    col1 = arg('col1', col1, str)
    col2 = arg('col2', col2, str)

    if col1 is None:
        if len(data.columns) != 2:
            logger.error(missing_msg)
            raise ValueError(missing_msg)
        else:
            col1 = data.columns[0]
    if col2 is None:
        if len(data.columns) != 2:
            logger.error(missing_msg)
            raise ValueError(missing_msg)
        else:
            col2 = data.columns[1]
    if col1 == col2:
        diff_msg = "'col1' and 'col2' must be different."
        logger.error(diff_msg)
        raise ValueError(diff_msg)
    data = data[[col1, col2]]
    return _ttest_base(conn_context, data, mu=mu, test_type=test_type, paired=False,
                       var_equal=var_equal, conf_level=conf_level)

def ttest_paired(conn_context, data, col1=None, col2=None, mu=0, test_type='two_sides',#pylint: disable=too-many-arguments, invalid-name
                 conf_level=0.95):
    """
    Perform the t-test for the mean difference of two sets of paired samples.

    Parameters
    ----------

    conn_context : ConnectionContext
        Database connection object.

    data : DataFrame

        DataFrame containing the data.

    col1 : str, optional

        Name of the column for sample1. \
        If not given, the input dataframe must only have two columns. \
        The first of two columns will be col1.

    col2 : str, optional

        Name of the column for sample2. \
        If not given, the input dataframe must only have two columns. \
        The second of the two columns will be col2.

    mu : float, optional

        Hypothesized difference between two underlying population means.

        Default value: 0

    test_type : {'two_sides', 'less', 'greater'}, optional

        The alternative hypothesis type.

        Default value: two_sides

    conf_level : float, optional

        Confidence level for alternative hypothesis confidence interval.

        Default value: 0.95

    Returns
    -------

    stat_df : DataFrame
        DataFrame containing the statistics results from the t-test.

    Examples
    --------

    Original data:

    >>> df.collect()
        X1    X2
    0  1.0  10.0
    1  2.0  12.0
    2  4.0  11.0
    3  7.0  15.0
    4  3.0  10.0

    perform Paired Sample T-Test

    >>> ttest_paired(conn, df).collect()
                    STAT_NAME  STAT_VALUE
    0                 t-value  -14.062884
    1       degree of freedom    4.000000
    2                 p-value    0.000148
    3  _PAL_MEAN_DIFFERENCES_   -8.200000
    4        confidence level    0.950000
    5              lowerLimit   -9.818932
    6              upperLimit   -6.581068
    """
    missing_msg = ("If 'col1' or 'col2' is not given, " +
                   "the input dataframe must only have two columns.")
    col1 = arg('col1', col1, str)
    col2 = arg('col2', col2, str)
    if col1 is None:
        if len(data.columns) != 2:
            logger.error(missing_msg)
            raise ValueError(missing_msg)
        else:
            col1 = data.columns[0]
    if col2 is None:
        if len(data.columns) != 2:
            logger.error(missing_msg)
            raise ValueError(missing_msg)
        else:
            col2 = data.columns[1]
    if col1 == col2:
        diff_msg = "'col1' and 'col2' must be different."
        logger.error(diff_msg)
        raise ValueError(diff_msg)
    data = data[[col1, col2]]
    return _ttest_base(conn_context, data, mu=mu, test_type=test_type, paired=True,
                       conf_level=conf_level)

#pylint: disable=too-many-arguments, too-many-locals
def f_oneway(conn_context, data, group=None, sample=None,
             multcomp_method=None, significance_level=None):
    r"""
    Performs a 1-way ANOVA.

    The purpose of one-way ANOVA is to determine whether there is any \
    statistically significant difference between the means of three \
    or more independent groups.

    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA system.

    data : DataFrame

        Input data.

    group : str

        Name of the group column. \
        If ``group`` is not provided, defaults to the first column.

    sample : str, optional

        Name of the sample measurement column. \
        If ``sample`` is not provided, ``data`` must have exactly 1 non-group \
        column and ``sample`` defaults to that column.

    multcomp_method : {'tukey-kramer', 'bonferroni', 'dunn-sidak', 'scheffe', 'fisher-lsd'},  str, optional

        Method used to perform multiple comparison tests.

        Defaults to tukey-kramer.

    significance_level : float, optional

        The significance level when the function calculates the confidence \
        interval in multiple comparison tests. \
        Values must be greater than 0 and less than 1.

        Defaults to 0.05.

    Returns
    -------

    statistics_df : DataFrame

        Statistics for each group, structured as follows:

            - GROUP, type NVARCHAR(256), group name.
            - VALID_SAMPLES, type INTEGER, number of valid samples.
            - MEAN, type DOUBLE, group mean.
            - SD, type DOUBLE, group standard deviation.

    ANOVA_df : DataFrame

        Computed results for ANOVA, structured as follows:

            - VARIABILITY_SOURCE, type NVARCHAR(100), source of variability, \
              including between groups, within groups (error) and total.

            - SUM_OF_SQUARES, type DOUBLE, sum of squares.

            - DEGREES_OF_FREEDOM, type DOUBLE, degrees of freedom.

            - MEAN_SQUARES, type DOUBLE, mean squares.

            - F_RATIO, type DOUBLE, calculated as mean square between groups \
              divided by mean square of error.

            - P_VALUE, type DOUBLE, associated p-value from the F-distribution.

    multiple_comparison_df : DataFrame

        Multiple comparison results, structured as follows:

            - FIRST_GROUP, type NVARCHAR(256), the name of the first group to \
              conduct pairwise test on.

            - SECOND_GROUP, type NVARCHAR(256), the name of the second group
              to conduct pairwise test on.

            - MEAN_DIFFERENCE, type DOUBLE, mean difference between the two \
              groups.

            - SE, type DOUBLE, standard error computed from all data.

            - P_VALUE, type DOUBLE, p-value.

            - CI_LOWER, type DOUBLE, the lower limit of the confidence interval.

            - CI_UPPER, type DOUBLE, the upper limit of the confidence interval.

    Examples
    --------

    Samples for One Way ANOVA test:

    >>> df.collect()
       GROUP  DATA
    0      A   4.0
    1      A   5.0
    2      A   4.0
    3      A   3.0
    4      A   2.0
    5      A   4.0
    6      A   3.0
    7      A   4.0
    8      B   6.0
    9      B   8.0
    10     B   4.0
    11     B   5.0
    12     B   4.0
    13     B   6.0
    14     B   5.0
    15     B   8.0
    16     C   6.0
    17     C   7.0
    18     C   6.0
    19     C   6.0
    20     C   7.0
    21     C   5.0

    Perform one-way ANOVA test:

    >>> stats, anova, mult_comp= f_oneway(conn, df,
    ...                                   multcomp_method='Tukey-Kramer',
    ...                                   significance_level=0.05)

    Outputs:

    >>> stats.collect()
       GROUP  VALID_SAMPLES      MEAN        SD
    0      A              8  3.625000  0.916125
    1      B              8  5.750000  1.581139
    2      C              6  6.166667  0.752773
    3  Total             22  5.090909  1.600866
    >>> anova.collect()
      VARIABILITY_SOURCE  SUM_OF_SQUARES  DEGREES_OF_FREEDOM  MEAN_SQUARES  \
    0              Group       27.609848                 2.0     13.804924
    1              Error       26.208333                19.0      1.379386
    2              Total       53.818182                21.0           NaN
         F_RATIO   P_VALUE
    0  10.008021  0.001075
    1        NaN       NaN
    2        NaN       NaN
    >>> mult_comp.collect()
      FIRST_GROUP SECOND_GROUP  MEAN_DIFFERENCE        SE   P_VALUE  CI_LOWER  \
    0           A            B        -2.125000  0.587236  0.004960 -3.616845
    1           A            C        -2.541667  0.634288  0.002077 -4.153043
    2           B            C        -0.416667  0.634288  0.790765 -2.028043
       CI_UPPER
    0 -0.633155
    1 -0.930290
    2  1.194710
    """

    require_pal_usable(conn_context)

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()

    multcomp_method_map = {'tukey-kramer':0, 'bonferroni':1,
                           'dunn-sidak':2, 'scheffe':3, 'fisher-lsd':4}

    group = arg('group', group, str)
    sample = arg('sample', sample, str)
    multcomp_method = arg('multcomp_method', multcomp_method, multcomp_method_map)
    significance_level = arg('significance_level', significance_level, float)
    if significance_level is not None and not 0 < significance_level < 1:
        msg = "significance_level {!r} is out of bounds.".format(significance_level)
        logger.error(msg)
        raise ValueError(msg)

    cols = data.columns
    if group is None:
        group = cols[0]
    cols.remove(group)
    if sample is None:
        if len(cols) != 1:
            msg = ('PAL One Way ANOVA requires exactly one ' +
                   'data column.')
            logger.error(msg)
            raise TypeError(msg)
        sample = cols[0]

    data = data[[group] + [sample]]

    tables = ['STATISTICS', 'ANOVA', 'MULTIPLE_COMPARISON']
    tables = ['#PAL_ANOVA_{}_TBL_{}'.format(name, unique_id) for name in tables]

    stats_tbl, anova_tbl, multi_comparison_tbl = tables

    param_rows = [
        ('MULTCOMP_METHOD', multcomp_method, None, None),
        ('SIGNIFICANCE_LEVEL', None, significance_level, None)]

    #stats_spec = [
    #    ('GROUP', NVARCHAR(256)),
    #    ('VALID_SAMPLES', INTEGER),
    #    ('MEAN', DOUBLE),
    #    ('SD', DOUBLE)]

    #anova_spec = [
    #    ('VARIABILITY_SOURCE', NVARCHAR(100)),
    #    ('SUM_OF_SQUARES', DOUBLE),
    #    ('DEGREES_OF_FREEDOM', DOUBLE),
    #    ('MEAN_SQUARES', DOUBLE),
    #    ('F_RATIO', DOUBLE),
    #    ('P_VALUE', DOUBLE)]

    #multi_comparison_spec = [
    #    ('FIRST_GROUP', NVARCHAR(256)),
    #    ('SECOND_GROUP', NVARCHAR(256)),
    #    ('MEAN_DIFFERENCE', DOUBLE),
    #    ('SE', DOUBLE),
    #    ('P_VALUE', DOUBLE),
    #    ('CI_LOWER', DOUBLE),
    #    ('CI_UPPER', DOUBLE),
    #    ]

    try:
        #materialize(conn_context, data_tbl, data)
        #create(conn_context, ParameterTable(param_tbl).with_data(param_rows))
        #create(conn_context, Table(stats_tbl, stats_spec))
        #create(conn_context, Table(anova_tbl, anova_spec))
        #create(conn_context, Table(multi_comparison_tbl, multi_comparison_spec))
        call_pal_auto(conn_context,
                      'PAL_ONEWAY_ANOVA',
                      data,
                      ParameterTable().with_data(param_rows),
                      *tables)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn_context, tables)
        raise

    return (conn_context.table(stats_tbl),
            conn_context.table(anova_tbl),
            conn_context.table(multi_comparison_tbl))

def f_oneway_repeated(conn_context, data, subject_id, measures=None,
                      multcomp_method=None, significance_level=None, se_type=None):
    r"""
    Performs one-way repeated measures analysis of variance, along with \
    Mauchly's Test of Sphericity and post hoc multiple comparison tests.

    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA system.

    data : DataFrame

        Input data.

    subject_id : str

        Name of the subject ID column. \
        The algorithm treats each row of the data table as a different subject. \
        Hence there should be no duplicate subject IDs in this column.

    measures : list of str, optional

        Names of the groups (measures). \
        If ``measures`` is not provided, defaults to all non-subject_id columns. \

    multcomp_method : {'tukey-kramer', 'bonferroni', 'dunn-sidak', 'scheffe', 'fisher-lsd'}, optional

        Method used to perform multiple comparison tests.

        Defaults to bonferroni.

    significance_level : float, optional

        The significance level when the function calculates the confidence \
        interval in multiple comparison tests. \
        Values must be greater than 0 and less than 1.

        Defaults to 0.05.

    se_type : {'all-data', 'two-group'}

        Type of standard error used in multiple comparison tests.

            - 'all-data': computes the standard error from all data. It has \
              more power if the assumption of sphericity is true, especially \
              with small data sets.
            - 'two-group': computes the standard error from only the two groups \
              being compared. It doesn't assume sphericity.

        Defaults to two-group.

    Returns
    -------

    statistics_df : DataFrame

        Statistics for each group, structured as follows:

            - GROUP, type NVARCHAR(256), group name.
            - VALID_SAMPLES, type INTEGER, number of valid samples.
            - MEAN, type DOUBLE, group mean.
            - SD, type DOUBLE, group standard deviation.

    Mauchly_test_df : DataFrame

        Mauchly test results, structured as follows:

            - STAT_NAME, type NVARCHAR(100), names of test result quantities.
            - STAT_VALUE, type DOUBLE, values of test result quantities.

    ANOVA_df : DataFrame

        Computed results for ANOVA, structured as follows:

            - VARIABILITY_SOURCE, type NVARCHAR(100), source of variability, \
              divided into group, error and subject portions.
            - SUM_OF_SQUARES, type DOUBLE, sum of squares.
            - DEGREES_OF_FREEDOM, type DOUBLE, degrees of freedom.
            - MEAN_SQUARES, type DOUBLE, mean squares.
            - F_RATIO, type DOUBLE, calculated as mean square between groups \
              divided by mean square of error.
            - P_VALUE, type DOUBLE, associated p-value from the F-distribution.
            - P_VALUE_GG, type DOUBLE, p-value of Greehouse-Geisser correction.
            - P_VALUE_HF, type DOUBLE, p-value of Huynh-Feldt correction.
            - P_VALUE_LB, type DOUBLE, p-value of lower bound correction.

    multiple_comparison_df : DataFrame

        Multiple comparison results, structured as follows:

          - FIRST_GROUP, type NVARCHAR(256), the name of the first group to \
            conduct pairwise test on.
          - SECOND_GROUP, type NVARCHAR(256), the name of the second group \
            to conduct pairwise test on.
          - MEAN_DIFFERENCE, type DOUBLE, mean difference between the two \
            groups.
          - SE, type DOUBLE, standard error computed from all data or \
            compared two groups, depending on ``se_type``.
          - P_VALUE, type DOUBLE, p-value.
          - CI_LOWER, type DOUBLE, the lower limit of the confidence interval.
          - CI_UPPER, type DOUBLE, the upper limit of the confidence interval.

    Examples
    --------

    Samples for One Way Repeated ANOVA test:

    >>> df.collect()
      ID  MEASURE1  MEASURE2  MEASURE3  MEASURE4
    0  1       8.0       7.0       1.0       6.0
    1  2       9.0       5.0       2.0       5.0
    2  3       6.0       2.0       3.0       8.0
    3  4       5.0       3.0       1.0       9.0
    4  5       8.0       4.0       5.0       8.0
    5  6       7.0       5.0       6.0       7.0
    6  7      10.0       2.0       7.0       2.0
    7  8      12.0       6.0       8.0       1.0


    Perform one-way repeated measures ANOVA test:

    >>> stats, mtest, anova, mult_comp = f_oneway_repeated(
    ...     conn,
    ...     df,
    ...     subject_id='ID',
    ...     multcomp_method='bonferroni',
    ...     significance_level=0.05,
    ...     se_type='two-group')

    Outputs:

    >>> stats.collect()
          GROUP  VALID_SAMPLES   MEAN        SD
    0  MEASURE1              8  8.125  2.232071
    1  MEASURE2              8  4.250  1.832251
    2  MEASURE3              8  4.125  2.748376
    3  MEASURE4              8  5.750  2.915476
    >>> mtest.collect()
                        STAT_NAME  STAT_VALUE
    0                 Mauchly's W    0.136248
    1                  Chi-Square   11.405981
    2                          df    5.000000
    3                      pValue    0.046773
    4  Greenhouse-Geisser Epsilon    0.532846
    5         Huynh-Feldt Epsilon    0.665764
    6         Lower bound Epsilon    0.333333
    >>> anova.collect()
      VARIABILITY_SOURCE  SUM_OF_SQUARES  DEGREES_OF_FREEDOM  MEAN_SQUARES  \
    0              Group          83.125                 3.0     27.708333
    1            Subject          17.375                 7.0      2.482143
    2              Error         153.375                21.0      7.303571
        F_RATIO  P_VALUE  P_VALUE_GG  P_VALUE_HF  P_VALUE_LB
    0  3.793806  0.02557    0.062584    0.048331    0.092471
    1       NaN      NaN         NaN         NaN         NaN
    2       NaN      NaN         NaN         NaN         NaN
    >>> mult_comp.collect()
      FIRST_GROUP SECOND_GROUP  MEAN_DIFFERENCE        SE   P_VALUE  CI_LOWER  \
    0    MEASURE1     MEASURE2            3.875  0.811469  0.012140  0.924655
    1    MEASURE1     MEASURE3            4.000  0.731925  0.005645  1.338861
    2    MEASURE1     MEASURE4            2.375  1.792220  1.000000 -4.141168
    3    MEASURE2     MEASURE3            0.125  1.201747  1.000000 -4.244322
    4    MEASURE2     MEASURE4           -1.500  1.336306  1.000000 -6.358552
    5    MEASURE3     MEASURE4           -1.625  1.821866  1.000000 -8.248955
       CI_UPPER
    0  6.825345
    1  6.661139
    2  8.891168
    3  4.494322
    4  3.358552
    5  4.998955
    """

    require_pal_usable(conn_context)

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()

    multcomp_method_map = {'tukey-kramer':0, 'bonferroni':1,
                           'dunn-sidak':2, 'scheffe':3, 'fisher-lsd':4}
    se_type_map = {'all-data':0, 'two-group':1}

    subject_id = arg('subject_id', subject_id, str, required=True)
    measures = arg('measures', measures, ListOfStrings)
    multcomp_method = arg('multcomp_method', multcomp_method, multcomp_method_map)
    significance_level = arg('significance_level', significance_level, float)
    if significance_level is not None and not 0 < significance_level < 1:
        msg = "significance_level {!r} is out of bounds.".format(significance_level)
        logger.error(msg)
        raise ValueError(msg)
    se_type = arg('se_type', se_type, se_type_map)

    cols = data.columns
    cols.remove(subject_id)
    if measures is None:
        measures = cols

    data = data[[subject_id] + measures]

    tables = ['STATISTICS', 'MAUCHLY_TEST', 'ANOVA', 'MULTIPLE_COMPARISON']
    tables = ['#PAL_ANOVA_REPEATED_{}_TBL_{}'.format(name, unique_id) for name in tables]

    stats_tbl, mauchly_test_tbl, anova_tbl, multi_comparison_tbl = tables

    param_rows = [
        ('MULTCOMP_METHOD', multcomp_method, None, None),
        ('SIGNIFICANCE_LEVEL', None, significance_level, None),
        ('SE_TYPE', se_type, None, None)]

    try:
        call_pal_auto(conn_context,
                      'PAL_ONEWAY_REPEATED_MEASURES_ANOVA',
                      data,
                      ParameterTable().with_data(param_rows),
                      *tables)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn_context, tables)
        raise

    return (conn_context.table(stats_tbl),
            conn_context.table(mauchly_test_tbl),
            conn_context.table(anova_tbl),
            conn_context.table(multi_comparison_tbl))

def univariate_analysis(conn_context, data,
                        key=None, cols=None,
                        categorical_variable=None,
                        significance_level=None,
                        trimmed_percentage=None,):
    """
    Provides an overview of the dataset. For continuous columns, it provides \
    the count of valid observations, min, lower quartile, median, upper \
    quartile, max, mean, confidence interval for the mean (lower and upper \
    bound), trimmed mean, variance, standard deviation, skewness, and kurtosis. \
    For discrete columns, it provides the number of occurrences and the \
    percentage of the total data in each category.

    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA system.

    data : DataFrame

        Input data.

    key : str, optional

        Name of the ID column. If ``key`` is not provided, it is assumed \
        that the input has no ID column.

    cols : list of str, optional

        List of column names to analyze. If ``cols`` is not provided, it \
        defaults to all non-ID columns.

    categorical_variable : list of str, optional

        INTEGER columns specified in this list will be treated as categorical \
        data. By default, INTEGER columns are treated as continuous.

    significance_level : float, optional

        The significance level when the function calculates the confidence \
        interval of the sample mean. \
        Values must be greater than 0 and less than 1.

        Defaults to 0.05.

    trimmed_percentage : float, optional

        The ratio of data at both head and tail that will be dropped in the \
        process of calculating the trimmed mean. \
        Value range is from 0 to 0.5.

        Defaults to 0.05.

    Returns
    -------

    continuous_result : DataFrame

        Statistics for continuous variables, structured as follows:

            - VARIABLE_NAME, type NVARCHAR(256), variable names.
            - STAT_NAME, type NVARCHAR(100), names of statistical quantities, \
              including the count of valid observations, min, lower quartile, \
              median, upper quartile, max, mean, confidence interval for the \
              mean (lower and upper bound), trimmed mean, variance, standard \
              deviation, skewness, and kurtosis (14 quantities in total).
            - STAT_VALUE, type DOUBLE, values for the corresponding \
              statistical quantities.

    categorical_result : DataFrame

        Statistics for categorical variables, structured as follows:

          - VARIABLE_NAME, type NVARCHAR(256), variable names.
          - CATEGORY, type NVARCHAR(256), category names of the corresponding \
            variables. Null is also treated as a category.
          - STAT_NAME, type NVARCHAR(100), names of statistical quantities: \
            number of observations, percentage of total data points falling \
            in the current category for a variable (including null).
          - STAT_VALUE, type DOUBLE, values for the corresponding \
            statistical quantities.

    Examples
    --------

    Dataset to be analyzed:

    >>> df.collect()
          X1    X2  X3 X4
    0    1.2  None   1  A
    1    2.5  None   2  C
    2    5.2  None   3  A
    3  -10.2  None   2  A
    4    8.5  None   2  C
    5  100.0  None   3  B

    Perform univariate analysis:

    >>> continuous, categorical = univariate_analysis(
    ...     conn,
    ...     df,
    ...     categorical_variable=['X3'],
    ...     significance_level=0.05,
    ...     trimmed_percentage=0.2)

    Outputs:

    >>> continuous.collect()
       VARIABLE_NAME                 STAT_NAME   STAT_VALUE
    0             X1        valid observations     6.000000
    1             X1                       min   -10.200000
    2             X1            lower quartile     1.200000
    3             X1                    median     3.850000
    4             X1            upper quartile     8.500000
    5             X1                       max   100.000000
    6             X1                      mean    17.866667
    7             X1  CI for mean, lower bound   -24.879549
    8             X1  CI for mean, upper bound    60.612883
    9             X1              trimmed mean     4.350000
    10            X1                  variance  1659.142667
    11            X1        standard deviation    40.732575
    12            X1                  skewness     1.688495
    13            X1                  kurtosis     1.036148
    14            X2        valid observations     0.000000

    >>> categorical.collect()
       VARIABLE_NAME      CATEGORY      STAT_NAME  STAT_VALUE
    0             X3  __PAL_NULL__          count    0.000000
    1             X3  __PAL_NULL__  percentage(%)    0.000000
    2             X3             1          count    1.000000
    3             X3             1  percentage(%)   16.666667
    4             X3             2          count    3.000000
    5             X3             2  percentage(%)   50.000000
    6             X3             3          count    2.000000
    7             X3             3  percentage(%)   33.333333
    8             X4  __PAL_NULL__          count    0.000000
    9             X4  __PAL_NULL__  percentage(%)    0.000000
    10            X4             A          count    3.000000
    11            X4             A  percentage(%)   50.000000
    12            X4             B          count    1.000000
    13            X4             B  percentage(%)   16.666667
    14            X4             C          count    2.000000
    15            X4             C  percentage(%)   33.333333
    """

    require_pal_usable(conn_context)

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()

    key = arg('key', key, str)
    cols = arg('cols', cols, ListOfStrings)
    categorical_variable = arg('categorical_variable',
                               categorical_variable, ListOfStrings)
    significance_level = arg('significance_level', significance_level, float)
    if significance_level is not None and not 0 < significance_level < 1:
        msg = "significance_level {!r} is out of bounds.".format(significance_level)
        logger.error(msg)
        raise ValueError(msg)
    trimmed_percentage = arg('trimmed_percentage', trimmed_percentage, float)
    if trimmed_percentage is not None and not 0 <= trimmed_percentage < 0.5:
        msg = "trimmed_percentage {!r} is out of bounds.".format(trimmed_percentage)
        logger.error(msg)
        raise ValueError(msg)
    all_cols = data.columns
    if key is not None:
        id_col = [key]
        all_cols.remove(key)
    else:
        id_col = []
    if cols is None:
        cols = all_cols

    data = data[id_col + cols]

    tables = ['CONTINUOUS', 'CATEGORICAL']
    tables = ['#PAL_UNIVARIATE_{}_TBL_{}'.format(name, unique_id) for name in tables]

    continuous_tbl, categorical_tbl = tables

    param_rows = [
        ('SIGNIFICANCE_LEVEL', None, significance_level, None),
        ('TRIMMED_PERCENTAGE', None, trimmed_percentage, None),
        ('HAS_ID', key is not None, None, None)
        ]
    #PAL documentation is inconsistent with example, tests confirmed that the following
    #parameter works as expected
    if categorical_variable is not None:
        param_rows.extend(('CATEGORICAL_VARIABLE', None, None, variable)
                          for variable in categorical_variable)

    try:
        call_pal_auto(conn_context,
                      'PAL_UNIVARIATE_ANALYSIS',
                      data,
                      ParameterTable().with_data(param_rows),
                      *tables)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn_context, tables)
        raise

    return (conn_context.table(continuous_tbl),
            conn_context.table(categorical_tbl))

def _multivariate_analysis(conn_context, data,
                           cols=None, result_type=None):
    require_pal_usable(conn_context)

    result_type_map = {'covariance': 0, 'pearsonr': 1}

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()

    cols = arg('cols', cols, ListOfStrings)
    result_type = arg('result_type', result_type, result_type_map)

    if cols is not None:
        data = data[cols]

    result_tbl = '#PAL_MULTIVARIATE_RESULT_TBL_{}'.format(unique_id)

    param_rows = [
        ('RESULT_TYPE', result_type, None, None)
        ]

    try:
        call_pal_auto(conn_context,
                      'PAL_MULTIVARIATE_ANALYSIS',
                      data,
                      ParameterTable().with_data(param_rows),
                      result_tbl)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn_context, result_tbl)
        raise

    return conn_context.table(result_tbl)

def covariance_matrix(conn_context, data, cols=None):
    """
    Computes the covariance matrix.

    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA system.

    data : DataFrame

        Input data.

    cols : list of str, optional

        List of column names to analyze. If ``cols`` is not provided, it \
        defaults to all columns.

    Returns
    -------

    covariance_matrix : DataFrame

        Covariance between any two data samples (columns).

          - ID, type NVARCHAR. The values of this column are the column names \
            from ``cols``.
          - Covariance columns, type DOUBLE, named after the columns in ``cols``. \
            The covariance between variables X and Y is in column X, in the \
            row with ID value Y.

    Examples
    --------

    Dataset to be analyzed:

    >>> df.collect()
        X     Y
    0   1   2.4
    1   5   3.5
    2   3   8.9
    3  10  -1.4
    4  -4  -3.5
    5  11  32.8

    Compute the covariance matrix:

    >>> result = covariance_matrix(conn, df)

    Outputs:

    >>> result.collect()
      ID          X           Y
    0  X  31.866667   44.473333
    1  Y  44.473333  176.677667
    """

    return _multivariate_analysis(conn_context, data, cols, result_type='covariance')

def pearsonr_matrix(conn_context, data, cols=None):
    """
    Computes a correlation matrix using Pearson's correlation coefficient.

    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA system.

    data : DataFrame

        Input data.

    cols : list of str, optional

        List of column names to analyze. If ``cols`` is not provided, it \
        defaults to all columns.

    Returns
    -------

    pearsonr_matrix : DataFrame

        Pearson's correlation coefficient between any two data samples
        (columns).

          - ID, type NVARCHAR. The values of this column are the column names \
            from ``cols``.
          - Correlation coefficient columns, type DOUBLE, named after the \
            columns in ``cols``. The correlation coefficient between variables \
            X and Y is in column X, in the row with ID value Y.

    Examples
    --------

    Dataset to be analyzed:

    >>> df.collect()
        X     Y
    0   1   2.4
    1   5   3.5
    2   3   8.9
    3  10  -1.4
    4  -4  -3.5
    5  11  32.8

    Compute the Pearson's correlation coefficient matrix:

    >>> result = pearsonr_matrix(conn, df)

    Outputs:

    >>> result.collect()
      ID               X               Y
    0  X               1  0.592707653621
    1  Y  0.592707653621               1
    """

    return _multivariate_analysis(conn_context, data, cols, result_type='pearsonr')

def iqr(conn_context, data, key, col=None, multiplier=None):
    """
    Perform the inter-quartile range (IQR) test to find the outliers of the \
    data. The inter-quartile range (IQR) is the difference between the third \
    quartile (Q3) and the first quartile (Q1) of the data. Data points will be \
    marked as outliers if they fall outside the range from \
    Q1 - ``multiplier`` * IQR to Q3 + ``multiplier`` * IQR.


    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA system.

    data : DataFrame

        DataFrame containing the data.

    key : str

        Name of the ID column.

    col : str, optional

        Name of the data column that needs to be tested. \
        If not given, the input dataframe must only have two columns including \
        the ID column. The non-ID column will be ``col``.

    multiplier : float, optional

        The multiplier used to calculate the value range during the IQR test. \

        Upper-bound = Q3 + ``multiplier`` * IQR.

        Lower-bound = Q1 - ``multiplier`` * IQR.

        Q1 is equal to 25th percentile and Q3 is equal to 75th percentile.

        Defaults to 1.5.

    Returns
    -------

    res_df : DataFrame

        Test results, structured as follows:

            - ID, with same name and type as ``data``'s ID column.
            - IS_OUT_OF_RANGE, type INTEGER, containing the test results from \
            the IQR test that determine whether each data sample is in the \
            range or not:

                - 0: a value is in the range.
                - 1: a value is out of range.

    stat_df : DataFrame

        Statistical outputs, including Upper-bound and Lower-bound from the \
        IQR test, structured as follows:

            - STAT_NAME, type NVARCHAR(256), statistics name.
            - STAT_VALUE, type DOUBLE, statistics value.


    Examples
    --------
    Original data:

    >>> df.collect()
         ID   VAL
    0    P1  10.0
    1    P2  11.0
    2    P3  10.0
    3    P4   9.0
    4    P5  10.0
    5    P6  24.0
    6    P7  11.0
    7    P8  12.0
    8    P9  10.0
    9   P10   9.0
    10  P11   1.0
    11  P12  11.0
    12  P13  12.0
    13  P14  13.0
    14  P15  12.0

    Perform the IQR test:

    >>> res, stat = iqr(cc, df, 'ID', 'VAL', 1.5)
    >>> res.collect()
             ID  IS_OUT_OF_RANGE
    0    P1                0
    1    P2                0
    2    P3                0
    3    P4                0
    4    P5                0
    5    P6                1
    6    P7                0
    7    P8                0
    8    P9                0
    9   P10                0
    10  P11                1
    11  P12                0
    12  P13                0
    13  P14                0
    14  P15                0
    >>> stat.collect()
            STAT_NAME  STAT_VALUE
    0  lower quartile        10.0
    1  upper quartile        12.0
    """

    require_pal_usable(conn_context)
    multiplier = arg('multiplier', multiplier, float)
    if multiplier is not None and multiplier < 0:
        msg = 'Parameter multiplier should be greater than or equal to 0.'
        logger.error(msg)
        raise ValueError(msg)
    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    key = arg('key', key, str)
    cols = data.columns
    cols.remove(key)
    col = arg('col', col, str)
    if col is None:
        if len(cols) != 1:
            msg = ("If 'col' is not given, the input dataframe must only "+
                   "have one column except the ID column")
            logger.error(msg)
            raise ValueError(msg)
        else:
            col = cols[0]
    data = data.select(key, col)
    tables = ['#IQR_{}_TBL_{}'.format(name, unique_id) for name in ['RESULT',
                                                                    'STAT']]
    result_tbl, stat_tbl = tables
    param_rows = [('MULTIPLIER', None, multiplier, None)]

    try:
        call_pal_auto(conn_context,
                      'PAL_IQR',
                      data,
                      ParameterTable().with_data(param_rows),
                      result_tbl,
                      stat_tbl)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        raise
    return conn_context.table(result_tbl), conn_context.table(stat_tbl)
